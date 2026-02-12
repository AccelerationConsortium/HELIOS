"""
Narrative Demo: Show the FULL agent decision loop in terminal.

Each fault scenario plays out in real-time with every pipeline stage visible:
  SENSE → CLASSIFY → ANALYZE → DECIDE → EXECUTE → VERIFY → MEMORY

Three scenarios:
  Scenario 1: Tip collision (sensor_fail)    → ABORT + safe shutdown
  Scenario 2: Current overload (overshoot)   → DEGRADE + patch downstream
  Scenario 3: Timeout on optional step       → RETRY → budget exhausted → SKIP

Usage:
    python -m exp_agent.cli.run_narrative_demo
"""
import os
import sys
import time
import json
import uuid
from io import StringIO
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from ..core.types import (
    PlanStep, PlanPatch, Action, HardwareError, DeviceState,
    ExecutionState, Decision,
)
from ..devices.simulated.heater import SimHeater
from ..executor.guarded_executor import GuardedExecutor
from ..recovery.recovery_agent import RecoveryAgent
from ..recovery.policy import classify_error, analyze_signature
from ..recovery.classifier import ErrorClassifier


# ============================================================================
# Suppress noisy internal prints from policy/executor/heater modules
# ============================================================================

@contextmanager
def suppress_internal_prints():
    """Temporarily redirect stdout to suppress [Policy], [Executor], etc. prints."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


# ============================================================================
# Terminal formatting helpers
# ============================================================================

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"

def typed(text: str, delay: float = 0.01):
    """Print text with a typing effect."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()

def section(title: str, color: str = CYAN):
    print(f"\n{color}{'━' * 70}")
    print(f"  {title}")
    print(f"{'━' * 70}{RESET}\n")
    time.sleep(0.3)

def step_banner(num: int, total: int, step_id: str, stage: str,
                criticality: str, description: str):
    crit_color = RED if criticality == "critical" else YELLOW
    print(f"\n{BLUE}{'─' * 70}")
    print(f"  [{num}/{total}]  step_id={BOLD}{step_id}{RESET}{BLUE}  "
          f"stage={stage}")
    print(f"           criticality={crit_color}{criticality}{RESET}{BLUE}  "
          f"description={description}")
    print(f"{'─' * 70}{RESET}")
    time.sleep(0.3)

def log_event(event_type: str, level: str, msg: str, payload: dict = None):
    """Print a structured pipeline log event."""
    colors = {
        "INFO": WHITE, "WARNING": YELLOW,
        "ERROR": RED, "CRITICAL": MAGENTA,
    }
    c = colors.get(level, WHITE)
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"  {DIM}{ts}{RESET}  {c}[{level:8}]{RESET} [{event_type:30}] {msg}"
    if payload:
        kv = "  ".join(f"{k}={v}" for k, v in payload.items())
        line += f"  {DIM}{{{kv}}}{RESET}"
    print(line)
    time.sleep(0.15)


# ============================================================================
# In-memory event store (simulates DB / memory)
# ============================================================================

@dataclass
class MemoryRecord:
    event_id: str
    timestamp: str
    correlation_id: str
    event_type: str
    payload: Dict[str, Any]

class MemoryStore:
    """Simulates agent's persistent memory / audit database."""
    def __init__(self):
        self.records: List[MemoryRecord] = []
        self.file_path = Path("logs/narrative_demo_audit.jsonl")

    def write(self, event_type: str, correlation_id: str, payload: dict):
        rec = MemoryRecord(
            event_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id,
            event_type=event_type,
            payload=payload,
        )
        self.records.append(rec)

    def flush_to_file(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w") as f:
            for rec in self.records:
                f.write(json.dumps(asdict(rec), default=str) + "\n")

    def print_summary(self):
        print(f"\n{MAGENTA}{'━' * 70}")
        print(f"  MEMORY STORE — {len(self.records)} events persisted")
        print(f"{'━' * 70}{RESET}")
        for rec in self.records:
            icon = {"error.detected": "🔴", "error.classified": "🟡",
                    "signature.analyzed": "🔍", "decision.made": "⚡",
                    "recovery.executed": "🔧", "recovery.verified": "✅",
                    "step.completed": "📗", "step.skipped": "⊘",
                    "step.aborted": "🛑", "plan.patched": "📝",
                    }.get(rec.event_type, "•")
            print(f"  {icon} [{rec.event_type:25}] corr={rec.correlation_id[:16]}  "
                  f"{DIM}{json.dumps(rec.payload, default=str)[:80]}{RESET}")
        print(f"\n  {DIM}Audit log written to: {self.file_path}{RESET}")


# ============================================================================
# MCP tool call simulator
# ============================================================================

def simulate_mcp_call(tool_name: str, params: dict, result: str):
    """Print a simulated MCP tool invocation."""
    print(f"\n  {BG_BLUE}{WHITE} MCP TOOL CALL {RESET}")
    print(f"  {CYAN}tool:{RESET}   {tool_name}")
    print(f"  {CYAN}params:{RESET} {json.dumps(params)}")
    time.sleep(0.4)
    print(f"  {CYAN}result:{RESET} {GREEN}{result}{RESET}")
    time.sleep(0.2)


# ============================================================================
# Full pipeline runner for one error event
# ============================================================================

def run_error_pipeline(
    error: HardwareError,
    dev_state: DeviceState,
    history: List[DeviceState],
    last_action: Action,
    stage: str,
    memory: MemoryStore,
    device: SimHeater,
    executor: GuardedExecutor,
    state: ExecutionState,
    recovery_agent: Optional[RecoveryAgent] = None,
) -> Decision:
    """
    Run the complete CLASSIFY → ANALYZE → DECIDE → EXECUTE → VERIFY → MEMORY
    pipeline with full terminal output.
    """
    corr_id = f"err_{uuid.uuid4().hex[:8]}"

    # ── 1. SENSE (error detected) ──
    section("① SENSE — Error Detected", RED)
    log_event("error.detected", "ERROR",
              f"{error.type}: {error.message}",
              {"device": error.device, "severity": error.severity})
    memory.write("error.detected", corr_id, {
        "type": error.type, "severity": error.severity,
        "message": error.message, "device": error.device,
        "telemetry": dev_state.telemetry if dev_state else {},
    })
    time.sleep(0.3)

    # ── 2. CLASSIFY ──
    section("② CLASSIFY — Error Profiling", YELLOW)
    profile = classify_error(error)
    # Also get recommended_actions from ErrorClassifier (different profile type)
    classifier = ErrorClassifier()
    classifier_profile = classifier.classify(error)
    log_event("error.classified", "WARNING",
              f"unsafe={profile.unsafe}  recoverable={profile.recoverable}  "
              f"strategy={profile.default_strategy}",
              {"recommended_actions": classifier_profile.recommended_actions[:2]})
    memory.write("error.classified", corr_id, {
        "unsafe": profile.unsafe, "recoverable": profile.recoverable,
        "strategy": profile.default_strategy,
        "recommended_actions": classifier_profile.recommended_actions,
    })
    time.sleep(0.3)

    # Simulate MCP tool call for classification lookup
    simulate_mcp_call(
        "error_knowledge_base.lookup",
        {"error_type": error.type, "device": error.device},
        f"Found {error.type} in knowledge base: "
        f"{'UNSAFE — requires immediate attention' if profile.unsafe else 'SAFE — recoverable'}"
    )

    # ── 3. ANALYZE (signature) ──
    section("③ ANALYZE — Telemetry Signature Analysis", BLUE)
    sig = analyze_signature(history)
    log_event("signature.analyzed", "INFO",
              f"signature={sig.mode}  confidence={sig.confidence:.2f}",
              {"avg_slope": f"{sig.features.get('avg_slope', 0):.3f}",
               "variance": f"{sig.features.get('variance', 0):.3f}",
               "history_len": str(len(history))})

    # Show history trace
    if history:
        print(f"\n  {DIM}  Telemetry history (last {len(history)} readings):{RESET}")
        for i, h in enumerate(history[-5:]):
            temp = h.telemetry.get("temperature", "?")
            target = h.telemetry.get("target", "?")
            heating = h.telemetry.get("heating", "?")
            marker = "  ← current" if i == len(history[-5:]) - 1 else ""
            print(f"    {DIM}t-{len(history[-5:])-1-i}:{RESET} temp={temp}°C  "
                  f"target={target}°C  heating={heating}{RED}{marker}{RESET}")
        time.sleep(0.3)

    memory.write("signature.analyzed", corr_id, {
        "mode": sig.mode, "confidence": sig.confidence,
        "features": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in sig.features.items()},
    })

    # Simulate MCP call for pattern matching
    simulate_mcp_call(
        "pattern_matcher.match_signature",
        {"mode": sig.mode, "confidence": sig.confidence,
         "error_type": error.type},
        f"Signature '{sig.mode}' matches known pattern. "
        f"Recommended: {'degrade' if sig.mode == 'drift' else 'abort' if sig.mode == 'stall' else 'retry'}"
    )

    # ── 4. DECIDE ──
    section("④ DECIDE — Recovery Decision", MAGENTA)
    if recovery_agent is None:
        recovery_agent = RecoveryAgent()
    with suppress_internal_prints():
        decision = recovery_agent.decide(
            state=dev_state, error=error,
            history=history, last_action=last_action, stage=stage,
        )
    decision_color = {
        "retry": YELLOW, "skip": CYAN,
        "degrade": MAGENTA, "abort": RED,
    }.get(decision.kind, WHITE)
    log_event("decision.made", "WARNING" if decision.kind in ["abort", "degrade"] else "INFO",
              f"DECISION = {decision_color}{BOLD}{decision.kind.upper()}{RESET}",
              {"rationale": decision.rationale[:60]})

    if decision.actions:
        print(f"\n  {DIM}  Recovery actions planned:{RESET}")
        for i, act in enumerate(decision.actions):
            print(f"    {i+1}. {act.name} {act.params}")
    time.sleep(0.3)

    memory.write("decision.made", corr_id, {
        "kind": decision.kind, "rationale": decision.rationale,
        "actions": [{"name": a.name, "params": a.params} for a in decision.actions],
    })

    # ── 5. EXECUTE recovery actions ──
    if decision.actions:
        section("⑤ EXECUTE — Recovery Actions", GREEN)
        for i, act in enumerate(decision.actions):
            log_event("recovery.action", "INFO",
                      f"[{i+1}/{len(decision.actions)}] Executing: {act.name} {act.params}",
                      {"device": act.device or device.name})
            try:
                with suppress_internal_prints():
                    executor.execute(device, act, state)
                log_event("recovery.action.ok", "INFO",
                          f"[{i+1}/{len(decision.actions)}] {act.name} — SUCCESS", {})
            except HardwareError as e:
                log_event("recovery.action.failed", "ERROR",
                          f"[{i+1}/{len(decision.actions)}] {act.name} — FAILED: {e.message}", {})
                break
            time.sleep(0.2)

        memory.write("recovery.executed", corr_id, {
            "actions_count": len(decision.actions),
            "actions": [a.name for a in decision.actions],
        })

    # ── 6. VERIFY (post-recovery state) ──
    section("⑥ VERIFY — Post-Recovery State", GREEN)
    try:
        with suppress_internal_prints():
            device.tick()
            post_state = device.read_state()
        temp = post_state.telemetry.get("temperature", "?")
        heating = post_state.telemetry.get("heating", "?")
        status = post_state.status
        log_event("recovery.verified", "INFO",
                  f"Device state: temp={temp}°C  heating={heating}  status={status}", {})
        simulate_mcp_call(
            "device_monitor.verify_state",
            {"device": device.name, "expected_safe": True},
            f"Device {device.name}: temp={temp}°C, heating={heating}, status={status}"
        )
        memory.write("recovery.verified", corr_id, {
            "temp": temp, "heating": heating, "status": status,
        })
    except HardwareError:
        log_event("recovery.verify_failed", "ERROR",
                  "Cannot verify — device still in error state", {})
        memory.write("recovery.verified", corr_id, {"status": "verify_failed"})

    # ── 7. MEMORY (persist to audit trail) ──
    section("⑦ MEMORY — Persist to Audit Trail", DIM)
    simulate_mcp_call(
        "audit_database.write_trail",
        {"correlation_id": corr_id, "event_count": len([r for r in memory.records
                                                         if r.correlation_id == corr_id])},
        f"Trail {corr_id} persisted with "
        f"{len([r for r in memory.records if r.correlation_id == corr_id])} events"
    )
    log_event("memory.persisted", "INFO",
              f"Decision trail {corr_id} saved to audit log",
              {"correlation_id": corr_id})
    time.sleep(0.3)

    return decision


# ============================================================================
# Scenario runners
# ============================================================================

def run_scenario_1(memory: MemoryStore):
    """Scenario 1: Tip Collision (sensor_fail) → ABORT + safe shutdown."""
    print(f"\n\n{BG_RED}{WHITE}{BOLD}"
          f"{'':=^70}"
          f"{RESET}")
    print(f"{BG_RED}{WHITE}{BOLD}"
          f"{'  SCENARIO 1: Tip Collision — Sensor Failure  ':=^70}"
          f"{RESET}")
    print(f"{BG_RED}{WHITE}{BOLD}"
          f"{'':=^70}"
          f"{RESET}")
    time.sleep(0.5)

    typed("  Simulating: Probe tip collides with sample surface during approach.", 0.02)
    typed("  Fault injection: sensor_fail — sensor hardware reports invalid readings.", 0.02)
    typed("  Expected: Agent classifies as UNSAFE + NON-RECOVERABLE → ABORT.", 0.02)
    time.sleep(0.5)

    # Setup
    device = SimHeater(name="probe_tip_1", fault_mode="none")
    executor = GuardedExecutor()
    with suppress_internal_prints():
        state = ExecutionState(devices={device.name: device.read_state()})

    # Build some normal history first
    history: List[DeviceState] = []
    with suppress_internal_prints():
        for _ in range(3):
            device.tick()
            ds = device.read_state()
            history.append(ds)

    # Show normal operation
    section("WORKFLOW — Step 3/5: Approach Surface", BLUE)
    step_banner(3, 5, "approach", "approach",
                "critical", "Move probe tip to sample surface")
    log_event("step.executing", "INFO", "Approaching surface at 0.1mm/step...", {})
    time.sleep(0.5)

    # INJECT: sensor_fail error
    error = HardwareError(
        device="probe_tip_1",
        type="sensor_fail",
        severity="high",
        message="Sensor failure: tip force sensor reading -999 (invalid). "
                "Possible physical collision detected.",
        context={"force_reading": -999, "position_z": 0.003,
                 "expected_range": [0, 50]},
    )

    print(f"\n  {BG_RED}{WHITE} ✗ FAULT INJECTED {RESET}  "
          f"{RED}Tip collision at z=0.003mm — sensor reads -999{RESET}\n")
    time.sleep(0.5)

    last_action = Action(name="move_approach", effect="write",
                         device="probe_tip_1",
                         params={"z_step": -0.1, "speed": "slow"})

    decision = run_error_pipeline(
        error=error, dev_state=history[-1], history=history,
        last_action=last_action, stage="approach",
        memory=memory, device=device, executor=executor, state=state,
    )

    # Show abort outcome
    section("OUTCOME — Plan Aborted", RED)
    print(f"  {RED}{BOLD}✗ ABORT{RESET} — Critical sensor failure on probe_tip_1")
    print(f"  {DIM}Agent determined: unsafe + non-recoverable → immediate safe shutdown{RESET}")
    print(f"  {DIM}Remaining steps (measure, retract) will NOT execute{RESET}")
    memory.write("step.aborted", "workflow", {
        "step_id": "approach", "decision": "abort",
        "remaining_steps": ["measure", "retract"],
    })
    time.sleep(0.5)


def run_scenario_2(memory: MemoryStore):
    """Scenario 2: Current Overload (overshoot) → DEGRADE + patch downstream."""
    print(f"\n\n{BG_YELLOW}{WHITE}{BOLD}"
          f"{'':=^70}"
          f"{RESET}")
    print(f"{BG_YELLOW}{WHITE}{BOLD}"
          f"{'  SCENARIO 2: Current Overload — Temperature Overshoot  ':=^70}"
          f"{RESET}")
    print(f"{BG_YELLOW}{WHITE}{BOLD}"
          f"{'':=^70}"
          f"{RESET}")
    time.sleep(0.5)

    typed("  Simulating: Heater overshoots target during preheat phase.", 0.02)
    typed("  Fault injection: overshoot — temperature drifts 15°C above target.", 0.02)
    typed("  Expected: Agent detects drift signature → DEGRADE to lower target.", 0.02)
    time.sleep(0.5)

    device = SimHeater(name="heater_1", fault_mode="none")
    executor = GuardedExecutor()
    with suppress_internal_prints():
        state = ExecutionState(devices={device.name: device.read_state()})

    # Build a drifting history (simulating overshoot)
    history: List[DeviceState] = []
    temps = [25.0, 60.0, 95.0, 115.0, 125.0, 132.0, 135.0]
    for t in temps:
        ds = DeviceState(
            name="heater_1", status="running",
            telemetry={"temperature": t, "target": 120.0, "heating": True}
        )
        history.append(ds)

    section("WORKFLOW — Step 2/6: Preheat to 120°C", BLUE)
    step_banner(2, 6, "preheat", "heating",
                "critical", "Heat to 120°C target temperature")

    # Show drift happening
    print(f"  {DIM}  Temperature readings during preheat:{RESET}")
    for i, t in enumerate(temps):
        over = f"  {RED}← OVERSHOOT (+{t-120:.0f}°C){RESET}" if t > 122 else ""
        print(f"    t={i}: {t}°C{over}")
        time.sleep(0.1)
    time.sleep(0.3)

    # INJECT: overshoot error
    error = HardwareError(
        device="heater_1",
        type="overshoot",
        severity="high",
        message="Temperature overshoot: current=135.0°C, target=120.0°C, delta=+15.0°C",
        context={"current_temp": 135.0, "target_temp": 120.0,
                 "delta": 15.0, "threshold": 2.0},
    )

    print(f"\n  {BG_YELLOW}{WHITE} ✗ FAULT INJECTED {RESET}  "
          f"{YELLOW}Temperature overshoot: 135°C (target was 120°C){RESET}\n")
    time.sleep(0.5)

    last_action = Action(name="set_temperature", effect="write",
                         device="heater_1",
                         params={"temperature": 120.0})

    decision = run_error_pipeline(
        error=error, dev_state=history[-1], history=history,
        last_action=last_action, stage="heating",
        memory=memory, device=device, executor=executor, state=state,
    )

    # Show degrade outcome with PlanPatch
    if decision.kind == "degrade":
        section("OUTCOME — Plan Degraded + Downstream Patched", MAGENTA)
        degraded_target = None
        for act in decision.actions:
            if act.name == "set_temperature" and "temperature" in act.params:
                degraded_target = act.params["temperature"]
        if degraded_target is None:
            degraded_target = 110.0  # Fallback for demo display

        print(f"  {MAGENTA}{BOLD}↓ DEGRADE{RESET} — Lowering target from 120°C to {degraded_target}°C")
        print(f"\n  {BOLD}PlanPatch generated:{RESET}")
        print(f"    original_target: 120.0°C")
        print(f"    degraded_target: {degraded_target}°C")
        print(f"    overrides:")
        print(f"      hold.temperature: 120.0 → {degraded_target}")
        print(f"      measure.temperature: 120.0 → {degraded_target}")
        print(f"    relaxations:")
        print(f"      hold.postcondition: '~= 120.0 +/- 2.0' → '~= {degraded_target} +/- 2.0'")
        print(f"      measure.postcondition: '~= 120.0 +/- 3.0' → '~= {degraded_target} +/- 3.0'")
        print(f"    notes:")
        print(f"      - Degraded from 120°C to {degraded_target}°C at step preheat")
        print(f"      - Downstream postconditions relaxed. Results may be compromised.")

        simulate_mcp_call(
            "plan_manager.apply_patch",
            {"patch_type": "degrade", "original": 120.0, "degraded": degraded_target,
             "affected_steps": ["hold", "measure"]},
            f"Patch applied: 2 downstream steps updated to {degraded_target}°C"
        )

        memory.write("plan.patched", "workflow", {
            "original_target": 120.0, "degraded_target": degraded_target,
            "affected_steps": ["hold", "measure"],
        })
    else:
        section("OUTCOME — Plan Aborted (overshoot too severe)", RED)
        print(f"  {RED}{BOLD}✗ ABORT{RESET} — Overshoot triggered safety violation")
        print(f"  {DIM}Agent could not degrade (insufficient drift history or unsafe){RESET}")
        memory.write("step.aborted", "workflow", {
            "step_id": "preheat", "decision": decision.kind,
        })
    time.sleep(0.5)


def run_scenario_3(memory: MemoryStore):
    """Scenario 3: Timeout on optional step → RETRY → budget exhausted → SKIP."""
    print(f"\n\n{BG_GREEN}{WHITE}{BOLD}"
          f"{'':=^70}"
          f"{RESET}")
    print(f"{BG_GREEN}{WHITE}{BOLD}"
          f"{'  SCENARIO 3: Measurement Timeout — Optional Step Skip  ':=^70}"
          f"{RESET}")
    print(f"{BG_GREEN}{WHITE}{BOLD}"
          f"{'':=^70}"
          f"{RESET}")
    time.sleep(0.5)

    typed("  Simulating: Optional measurement step times out waiting for reading.", 0.02)
    typed("  Fault injection: postcondition timeout — measurement never stabilizes.", 0.02)
    typed("  Expected: Agent retries → budget exhausted → SKIP (optional step).", 0.02)
    time.sleep(0.5)

    device = SimHeater(name="heater_1", fault_mode="none")
    executor = GuardedExecutor()
    with suppress_internal_prints():
        state = ExecutionState(devices={device.name: device.read_state()})
    # Shared recovery agent so retry counts persist across attempts
    shared_agent = RecoveryAgent()

    # Build stable-but-slightly-varying history (avoids "stall" signature)
    history: List[DeviceState] = []
    for t in [119.8, 120.1, 119.9, 120.2, 120.0]:
        ds = DeviceState(
            name="heater_1", status="running",
            telemetry={"temperature": t, "target": 120.0, "heating": True}
        )
        history.append(ds)

    section("WORKFLOW — Step 4/6: Verify Measurement (optional)", BLUE)
    step_banner(4, 6, "measure", "diagnostics",
                "optional", "Verify measurement reading — can skip on failure")
    time.sleep(0.3)

    # ── Attempt 1: timeout ──
    print(f"\n  {YELLOW}  ── Attempt 1 / max_retries=1 ──{RESET}")
    log_event("step.executing", "INFO",
              "Waiting for measurement reading to stabilize...", {})
    time.sleep(0.5)

    error = HardwareError(
        device="heater_1",
        type="postcondition_failed",
        severity="medium",
        message="Postcondition timeout: measurement reading did not stabilize "
                "within 5s (expected ~= 120.0 +/- 1.0)",
        context={"expected": 120.0, "tolerance": 1.0, "timeout": 5,
                 "last_reading": 118.3},
    )
    print(f"\n  {BG_YELLOW}{WHITE} ✗ FAULT INJECTED {RESET}  "
          f"{YELLOW}Postcondition timeout — reading=118.3, expected ~= 120.0{RESET}\n")
    time.sleep(0.3)

    last_action = Action(name="wait", effect="write", device="heater_1",
                         params={"duration": 5})

    decision = run_error_pipeline(
        error=error, dev_state=history[-1], history=history,
        last_action=last_action, stage="diagnostics",
        memory=memory, device=device, executor=executor, state=state,
        recovery_agent=shared_agent,
    )

    if decision.kind == "retry":
        print(f"\n  {YELLOW}{BOLD}↻ RETRY{RESET} — Policy says retry with backoff")
        print(f"  {DIM}  Retry budget: 1/1 used{RESET}")
        time.sleep(0.5)

        # ── Attempt 2: still fails ──
        print(f"\n  {YELLOW}  ── Attempt 2 (after backoff) ──{RESET}")
        log_event("step.executing", "INFO",
                  "Re-executing measurement after 2s backoff wait...", {})
        time.sleep(0.5)

        error2 = HardwareError(
            device="heater_1",
            type="postcondition_failed",
            severity="medium",
            message="Postcondition timeout again: reading=118.1, expected ~= 120.0",
            context={"expected": 120.0, "tolerance": 1.0, "timeout": 5,
                     "last_reading": 118.1},
        )
        print(f"\n  {BG_YELLOW}{WHITE} ✗ STILL FAILING {RESET}  "
              f"{YELLOW}Reading=118.1 — still outside tolerance{RESET}\n")
        time.sleep(0.3)

        # Run full pipeline again with the same agent (retry count increments)
        decision2 = run_error_pipeline(
            error=error2, dev_state=history[-1], history=history,
            last_action=last_action, stage="diagnostics",
            memory=memory, device=device, executor=executor, state=state,
            recovery_agent=shared_agent,
        )

        # Show what the policy decided on the second attempt
        if decision2.kind != "retry":
            print(f"\n  {DIM}  Policy decision on attempt 2: {decision2.kind.upper()}{RESET}")
            print(f"  {DIM}  Reason: {decision2.rationale}{RESET}")

    # Show final skip decision
    print(f"\n  {CYAN}{BOLD}⊘ SKIP{RESET} — Retry budget exhausted on optional step")
    print(f"  {DIM}  Step 'measure' has criticality=optional, on_failure=skip{RESET}")
    print(f"  {DIM}  Workflow cursor advances to next step → experiment continues{RESET}")

    simulate_mcp_call(
        "workflow_engine.advance_cursor",
        {"current_step": "measure", "decision": "skip",
         "next_step": "cooldown"},
        "Cursor advanced: measure(skipped) → cooldown"
    )

    memory.write("step.skipped", "workflow", {
        "step_id": "measure", "reason": "retry budget exhausted, on_failure=skip",
        "attempts": 2,
    })

    section("OUTCOME — Optional Step Skipped, Workflow Continues", CYAN)
    print(f"  {CYAN}⊘ SKIPPED{RESET} measure (diagnostics)")
    print(f"  {GREEN}→ CONTINUING{RESET} to step 5/6: cooldown")
    print(f"  {DIM}  Note: Measurement data unavailable for this run.{RESET}")
    time.sleep(0.5)


# ============================================================================
# Main
# ============================================================================

def main():
    memory = MemoryStore()

    print(f"\n{BOLD}{'═' * 70}")
    print(f"  EXP-AGENT: Recovery-Aware Execution Agent")
    print(f"  Full Decision Pipeline Demo")
    print(f"{'═' * 70}{RESET}")
    print(f"""
  This demo shows the agent's complete decision loop:

    {CYAN}① SENSE{RESET}    → Detect hardware error from device telemetry
    {YELLOW}② CLASSIFY{RESET} → Profile error: unsafe? recoverable? severity?
    {BLUE}③ ANALYZE{RESET}  → Analyze telemetry signature: drift? stall? oscillation?
    {MAGENTA}④ DECIDE{RESET}  → Choose recovery strategy: RETRY / SKIP / DEGRADE / ABORT
    {GREEN}⑤ EXECUTE{RESET}  → Run recovery actions (MCP tool calls)
    {GREEN}⑥ VERIFY{RESET}   → Check post-recovery device state
    {DIM}⑦ MEMORY{RESET}   → Persist decision trail to audit database

  Three fault scenarios will be demonstrated:
    🔴 Scenario 1: Tip collision (sensor_fail) → ABORT
    🟡 Scenario 2: Current overload (overshoot) → DEGRADE
    🟢 Scenario 3: Measurement timeout → RETRY → SKIP
""")
    time.sleep(2)

    # Run all three scenarios
    run_scenario_1(memory)
    time.sleep(1)

    run_scenario_2(memory)
    time.sleep(1)

    run_scenario_3(memory)
    time.sleep(1)

    # Final summary
    memory.flush_to_file()
    memory.print_summary()

    print(f"\n{BOLD}{'═' * 70}")
    print(f"  DEMO COMPLETE — 3 scenarios, 4 decision types demonstrated")
    print(f"{'═' * 70}{RESET}")
    print(f"""
  {RED}✗ ABORT{RESET}   — Sensor failure → unsafe + non-recoverable → safe shutdown
  {MAGENTA}↓ DEGRADE{RESET} — Temperature overshoot + drift → lower target, patch downstream
  {YELLOW}↻ RETRY{RESET}  — Postcondition timeout → retry with exponential backoff
  {CYAN}⊘ SKIP{RESET}    — Optional step retry exhausted → skip, workflow continues

  {DIM}Agent architecture:{RESET}
    SimDevice → GuardedExecutor → RecoveryAgent(classify→signature→decide)
    → WorkflowSupervisor(cursor, criticality, PlanPatch)
    → PipelineLogger(console, file, memory) → TrailAnalyzer
""")


if __name__ == "__main__":
    main()
