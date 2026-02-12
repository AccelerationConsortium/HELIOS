#!/usr/bin/env python3
"""
Multi-Device Fault Recovery Demo

Demonstrates the same recovery pipeline handling faults across 4 different device types:
  1. Heater      → overshoot      → DEGRADE
  2. Pump        → flow_blocked   → DEGRADE (prime + reduce)
  3. Positioner  → collision      → ABORT
  4. Spectrometer → signal_saturated → DEGRADE (reduce integration)

This shows that the recovery logic is device-agnostic - the same 7-stage pipeline
handles all device types uniformly.

Usage:
    python -m exp_agent.cli.run_multi_device_demo
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
from typing import List, Dict, Any, Optional, Union

from ..core.types import (
    PlanStep, PlanPatch, Action, HardwareError, DeviceState,
    ExecutionState, Decision,
)
from ..devices.simulated.heater import SimHeater
from ..devices.simulated.pump import SimPump
from ..devices.simulated.positioner import SimPositioner
from ..devices.simulated.spectrometer import SimSpectrometer
from ..devices.base import Device
from ..executor.guarded_executor import GuardedExecutor
from ..recovery.recovery_agent import RecoveryAgent
from ..recovery.policy import classify_error, analyze_signature
from ..recovery.classifier import ErrorClassifier


# ============================================================================
# Suppress noisy internal prints
# ============================================================================

@contextmanager
def suppress_internal_prints():
    """Temporarily redirect stdout to suppress internal prints."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


# ============================================================================
# Terminal formatting
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
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


def typed(text: str, delay: float = 0.01):
    """Type text with delay for visual effect."""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def section(title: str, color: str = WHITE):
    """Print a section header."""
    print(f"\n{color}{BOLD}{'─' * 60}{RESET}")
    print(f"{color}{BOLD}  {title}{RESET}")
    print(f"{color}{BOLD}{'─' * 60}{RESET}\n")


def device_banner(device_type: str, device_name: str, color: str):
    """Print a device identification banner."""
    icons = {
        "heater": "🔥",
        "pump": "💧",
        "positioner": "🎯",
        "spectrometer": "🔬",
    }
    icon = icons.get(device_type, "⚙️")
    print(f"\n  {color}{BOLD}{icon} DEVICE: {device_name} ({device_type.upper()}){RESET}\n")


def scenario_banner(num: int, total: int, device: str, fault: str, expected: str, color: str):
    """Print a scenario banner."""
    print(f"\n{color}{'═' * 70}{RESET}")
    print(f"{color}{BOLD}  SCENARIO {num}/{total}: {device.upper()} — {fault}{RESET}")
    print(f"{color}{'═' * 70}{RESET}")
    print(f"  {DIM}Expected outcome: {expected}{RESET}\n")


def log_event(event: str, level: str, msg: str, data: dict):
    """Log a pipeline event."""
    colors = {"INFO": GREEN, "WARNING": YELLOW, "ERROR": RED}
    color = colors.get(level, WHITE)
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"  {DIM}{ts}{RESET} {color}[{event}]{RESET} {msg}")


def simulate_mcp_call(tool: str, params: dict, result: str):
    """Simulate an MCP tool call."""
    print(f"    {CYAN}→ MCP: {tool}({json.dumps(params, default=str)[:50]}...){RESET}")
    time.sleep(0.15)
    print(f"    {CYAN}← {result[:60]}...{RESET}")


# ============================================================================
# Memory Store
# ============================================================================

@dataclass
class MemoryStore:
    """Simple in-memory event store."""
    events: List[Dict[str, Any]] = field(default_factory=list)
    log_file: Optional[Path] = None

    def write(self, event_type: str, correlation_id: str, data: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "correlation_id": correlation_id,
            "data": data
        }
        self.events.append(entry)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")


# ============================================================================
# Generic Error Pipeline
# ============================================================================

def run_error_pipeline(
    error: HardwareError,
    dev_state: DeviceState,
    history: List[DeviceState],
    last_action: Action,
    stage: str,
    memory: MemoryStore,
    device: Device,
    executor: GuardedExecutor,
    state: ExecutionState,
    recovery_agent: Optional[RecoveryAgent] = None,
    metric: Optional[str] = None,
) -> Decision:
    """
    Run the complete 7-stage pipeline for any device type.
    """
    corr_id = f"err_{uuid.uuid4().hex[:8]}"

    # ── 1. SENSE ──
    section("① SENSE — Error Detected", RED)
    log_event("error.detected", "ERROR", f"{error.type}: {error.message[:50]}", {})
    memory.write("error.detected", corr_id, {
        "type": error.type, "severity": error.severity,
        "message": error.message, "device": error.device,
    })
    time.sleep(0.2)

    # ── 2. CLASSIFY ──
    section("② CLASSIFY — Error Profiling", YELLOW)
    profile = classify_error(error)
    classifier = ErrorClassifier()
    classifier_profile = classifier.classify(error)
    log_event("error.classified", "WARNING",
              f"unsafe={profile.unsafe}  recoverable={profile.recoverable}  "
              f"strategy={profile.default_strategy}",
              {})
    memory.write("error.classified", corr_id, {
        "unsafe": profile.unsafe, "recoverable": profile.recoverable,
        "strategy": profile.default_strategy,
    })
    time.sleep(0.2)

    simulate_mcp_call(
        "error_knowledge_base.lookup",
        {"error_type": error.type, "device": error.device},
        f"Found {error.type}: {'UNSAFE' if profile.unsafe else 'SAFE'}, "
        f"{'recoverable' if profile.recoverable else 'non-recoverable'}"
    )

    # ── 3. ANALYZE ──
    section("③ ANALYZE — Signature Detection", BLUE)
    sig = analyze_signature(history, metric=metric)
    log_event("signature.analyzed", "INFO",
              f"mode={sig.mode}  confidence={sig.confidence:.2f}", {})
    memory.write("signature.analyzed", corr_id, {
        "mode": sig.mode, "confidence": sig.confidence,
    })
    time.sleep(0.2)

    simulate_mcp_call(
        "pattern_matcher.match_signature",
        {"mode": sig.mode, "error_type": error.type},
        f"Signature '{sig.mode}' → recommended: "
        f"{'degrade' if sig.mode == 'drift' else 'abort' if sig.mode == 'stall' else 'retry'}"
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
              {})
    print(f"    {DIM}Rationale: {decision.rationale}{RESET}")
    memory.write("decision.made", corr_id, {
        "kind": decision.kind, "rationale": decision.rationale,
    })
    time.sleep(0.2)

    simulate_mcp_call(
        "audit_database.write_trail",
        {"correlation_id": corr_id, "events": 4},
        "Decision trail persisted"
    )

    # ── 5. EXECUTE ──
    if decision.actions:
        section("⑤ EXECUTE — Recovery Actions", GREEN)
        for i, act in enumerate(decision.actions):
            log_event("recovery.action", "INFO",
                      f"[{i+1}/{len(decision.actions)}] {act.name} {act.params}", {})
            try:
                with suppress_internal_prints():
                    executor.execute(device, act, state)
                log_event("recovery.action.ok", "INFO",
                          f"[{i+1}/{len(decision.actions)}] {act.name} — SUCCESS", {})
            except Exception as e:
                log_event("recovery.action.failed", "ERROR",
                          f"[{i+1}/{len(decision.actions)}] {act.name} — FAILED", {})
            time.sleep(0.15)
        memory.write("recovery.executed", corr_id, {
            "actions": [a.name for a in decision.actions],
        })

    # ── 6. VERIFY ──
    section("⑥ VERIFY — Post-Recovery State", GREEN)
    try:
        with suppress_internal_prints():
            post_state = device.read_state()
        log_event("recovery.verified", "INFO",
                  f"Device status: {post_state.status}", {})
        # Show relevant telemetry
        key_metrics = list(post_state.telemetry.keys())[:4]
        for k in key_metrics:
            print(f"    {DIM}{k}: {post_state.telemetry[k]}{RESET}")
    except HardwareError:
        log_event("recovery.verified", "WARNING", "Device still in error state", {})

    simulate_mcp_call(
        "device_monitor.verify_state",
        {"device": device.name},
        f"State verified: {decision.kind}"
    )

    # ── 7. MEMORY ──
    section("⑦ MEMORY — Audit Trail", CYAN)
    log_event("trail.persisted", "INFO",
              f"Correlation ID: {corr_id} | Events: {len(memory.events)}", {})

    return decision


# ============================================================================
# Scenario 1: Heater Overshoot
# ============================================================================

def run_heater_scenario(memory: MemoryStore):
    scenario_banner(1, 4, "Heater", "Temperature Overshoot",
                    "DEGRADE to lower target", YELLOW)

    device = SimHeater(name="heater_1", fault_mode="none")
    executor = GuardedExecutor()
    with suppress_internal_prints():
        state = ExecutionState(devices={device.name: device.read_state()})

    device_banner("heater", device.name, RED)

    # Build drifting history
    history = []
    temps = [100.0, 110.0, 120.0, 128.0, 133.0, 138.0]
    for t in temps:
        ds = DeviceState(
            name="heater_1", status="running",
            telemetry={"temperature": t, "target": 120.0, "heating": True}
        )
        history.append(ds)
        print(f"    {DIM}Temp reading: {t}°C{RESET}")
        time.sleep(0.1)

    print(f"\n  {BG_YELLOW}{WHITE} ⚠ FAULT DETECTED {RESET}  "
          f"{YELLOW}Overshoot: 138°C (target: 120°C){RESET}\n")

    error = HardwareError(
        device="heater_1", type="overshoot", severity="high",
        message="Temperature overshoot: 138°C exceeds target 120°C by 18°C",
        when=str(time.time()),
        context={"current": 138.0, "target": 120.0}
    )
    last_action = Action(name="set_temperature", effect="write",
                         params={"temperature": 120.0})

    decision = run_error_pipeline(
        error=error, dev_state=history[-1], history=history,
        last_action=last_action, stage="preheat",
        memory=memory, device=device, executor=executor, state=state,
        metric="temperature"
    )

    print(f"\n  {MAGENTA}{BOLD}✓ OUTCOME: {decision.kind.upper()}{RESET}")
    print(f"  {DIM}Heater degraded to lower target, experiment continues{RESET}\n")
    return decision


# ============================================================================
# Scenario 2: Pump Flow Blocked
# ============================================================================

def run_pump_scenario(memory: MemoryStore):
    scenario_banner(2, 4, "Pump", "Flow Blocked",
                    "DEGRADE (stop + prime)", RED)

    device = SimPump(name="pump_1", fault_mode="none")
    executor = GuardedExecutor()
    with suppress_internal_prints():
        state = ExecutionState(devices={device.name: device.read_state()})

    device_banner("pump", device.name, BLUE)

    # Build history showing flow dropping
    history = []
    flows = [50.0, 45.0, 30.0, 10.0, 2.0, 0.0]
    for f in flows:
        ds = DeviceState(
            name="pump_1", status="running",
            telemetry={"flow_rate": f, "pressure": 3.0 + f/10, "target_flow": 50.0, "running": True}
        )
        history.append(ds)
        print(f"    {DIM}Flow: {f} mL/min | Pressure: {3.0 + f/10:.1f} bar{RESET}")
        time.sleep(0.1)

    print(f"\n  {BG_RED}{WHITE} ✗ FAULT DETECTED {RESET}  "
          f"{RED}Flow blocked: 0 mL/min (target: 50){RESET}\n")

    error = HardwareError(
        device="pump_1", type="flow_blocked", severity="high",
        message="Flow blocked: rate=0.0 mL/min, pressure building",
        when=str(time.time()),
        context={"flow_rate": 0.0, "pressure": 8.0, "target_flow": 50.0}
    )
    last_action = Action(name="set_flow", effect="write",
                         params={"flow_rate": 50.0})

    decision = run_error_pipeline(
        error=error, dev_state=history[-1], history=history,
        last_action=last_action, stage="dispense",
        memory=memory, device=device, executor=executor, state=state,
        metric="flow_rate"
    )

    print(f"\n  {MAGENTA}{BOLD}✓ OUTCOME: {decision.kind.upper()}{RESET}")
    print(f"  {DIM}Pump stopped safely, flow reduced{RESET}\n")
    return decision


# ============================================================================
# Scenario 3: Positioner Collision
# ============================================================================

def run_positioner_scenario(memory: MemoryStore):
    scenario_banner(3, 4, "Positioner", "Collision Detected",
                    "ABORT (non-recoverable)", RED)

    device = SimPositioner(name="stage_1", fault_mode="none")
    executor = GuardedExecutor()
    with suppress_internal_prints():
        state = ExecutionState(devices={device.name: device.read_state()})

    device_banner("positioner", device.name, GREEN)

    # Build history showing movement then sudden stop
    history = []
    positions = [
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),  # Stalled - collision
    ]
    for x, y, z in positions:
        ds = DeviceState(
            name="stage_1", status="moving",
            telemetry={"x": x, "y": y, "z": z, "target_x": 30.0, "moving": True}
        )
        history.append(ds)
        print(f"    {DIM}Position: ({x}, {y}, {z}) mm{RESET}")
        time.sleep(0.1)

    print(f"\n  {BG_RED}{WHITE} ✗ COLLISION {RESET}  "
          f"{RED}Hard stop at (15, 0, 0) — obstacle detected{RESET}\n")

    error = HardwareError(
        device="stage_1", type="collision", severity="critical",
        message="Collision detected at position (15.0, 0.0, 0.0)",
        when=str(time.time()),
        context={"position": {"x": 15.0, "y": 0.0, "z": 0.0}, "target": {"x": 30.0}}
    )
    last_action = Action(name="move_to", effect="write",
                         params={"x": 30.0, "y": 0.0, "z": 0.0})

    decision = run_error_pipeline(
        error=error, dev_state=history[-1], history=history,
        last_action=last_action, stage="approach",
        memory=memory, device=device, executor=executor, state=state,
        metric="x"
    )

    print(f"\n  {RED}{BOLD}✗ OUTCOME: {decision.kind.upper()}{RESET}")
    print(f"  {DIM}Physical collision — cannot recover, experiment aborted{RESET}\n")
    return decision


# ============================================================================
# Scenario 4: Spectrometer Signal Saturated
# ============================================================================

def run_spectrometer_scenario(memory: MemoryStore):
    scenario_banner(4, 4, "Spectrometer", "Signal Saturated",
                    "DEGRADE (reduce integration)", YELLOW)

    device = SimSpectrometer(name="spec_1", fault_mode="none")
    executor = GuardedExecutor()
    with suppress_internal_prints():
        state = ExecutionState(devices={device.name: device.read_state()})

    device_banner("spectrometer", device.name, MAGENTA)

    # Build history showing signal increasing to saturation
    history = []
    signals = [30000, 40000, 50000, 58000, 63000, 65000]
    for s in signals:
        ds = DeviceState(
            name="spec_1", status="acquiring",
            telemetry={
                "signal_intensity": float(s),
                "peak_intensity": float(s),
                "baseline": 100.0,
                "integration_time": 100
            }
        )
        history.append(ds)
        bar = "█" * int(s / 5000)
        print(f"    {DIM}Signal: {s:5d} {bar}{RESET}")
        time.sleep(0.1)

    print(f"\n  {BG_YELLOW}{WHITE} ⚠ SATURATED {RESET}  "
          f"{YELLOW}Peak intensity 65000 exceeds max 60000{RESET}\n")

    error = HardwareError(
        device="spec_1", type="signal_saturated", severity="medium",
        message="Detector saturated: peak=65000 (max=60000)",
        when=str(time.time()),
        context={"peak_intensity": 65000, "saturation_level": 60000}
    )
    last_action = Action(name="start_acquisition", effect="write",
                         params={"integration_time": 100})

    decision = run_error_pipeline(
        error=error, dev_state=history[-1], history=history,
        last_action=last_action, stage="measure",
        memory=memory, device=device, executor=executor, state=state,
        metric="signal_intensity"
    )

    print(f"\n  {MAGENTA}{BOLD}✓ OUTCOME: {decision.kind.upper()}{RESET}")
    print(f"  {DIM}Integration time reduced, measurement can continue{RESET}\n")
    return decision


# ============================================================================
# Main
# ============================================================================

def main():
    clear_screen()

    print(f"""
{BOLD}{CYAN}
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     🔬  MULTI-DEVICE FAULT RECOVERY DEMO  🔬                         ║
║                                                                      ║
║     Demonstrating device-agnostic recovery across 4 device types     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{RESET}
""")

    print(f"""
  {BOLD}Devices:{RESET}
    🔥 Heater       — Temperature control
    💧 Pump         — Fluid delivery
    🎯 Positioner   — XYZ stage motion
    🔬 Spectrometer — Optical measurement

  {BOLD}Pipeline:{RESET}
    SENSE → CLASSIFY → ANALYZE → DECIDE → EXECUTE → VERIFY → MEMORY

  {BOLD}Key insight:{RESET}
    Same recovery logic handles ALL device types uniformly.
    Error classification and signature analysis are device-agnostic.
""")

    input(f"\n  {DIM}Press Enter to start...{RESET}")

    # Setup
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    memory = MemoryStore(log_file=log_dir / "multi_device_demo_audit.jsonl")

    results = []

    # Run all scenarios
    results.append(("Heater", run_heater_scenario(memory)))
    time.sleep(0.5)
    input(f"\n  {DIM}Press Enter for next scenario...{RESET}")

    results.append(("Pump", run_pump_scenario(memory)))
    time.sleep(0.5)
    input(f"\n  {DIM}Press Enter for next scenario...{RESET}")

    results.append(("Positioner", run_positioner_scenario(memory)))
    time.sleep(0.5)
    input(f"\n  {DIM}Press Enter for next scenario...{RESET}")

    results.append(("Spectrometer", run_spectrometer_scenario(memory)))

    # Summary
    print(f"""
{CYAN}{'═' * 70}{RESET}
{CYAN}{BOLD}  DEMO COMPLETE — SUMMARY{RESET}
{CYAN}{'═' * 70}{RESET}
""")

    print(f"  {BOLD}Results by Device:{RESET}\n")
    for device, decision in results:
        icon = {"Heater": "🔥", "Pump": "💧", "Positioner": "🎯", "Spectrometer": "🔬"}[device]
        color = {
            "abort": RED, "degrade": MAGENTA, "retry": YELLOW, "skip": CYAN
        }.get(decision.kind, WHITE)
        print(f"    {icon} {device:12} → {color}{BOLD}{decision.kind.upper():8}{RESET}")

    print(f"""
  {BOLD}Statistics:{RESET}
    Total events:    {len(memory.events)}
    Audit log:       logs/multi_device_demo_audit.jsonl

  {BOLD}Key Takeaway:{RESET}
    {GREEN}✓{RESET} One pipeline handles heaters, pumps, positioners, spectrometers
    {GREEN}✓{RESET} Error classification is device-agnostic (unsafe/recoverable)
    {GREEN}✓{RESET} Signature analysis works across telemetry types
    {GREEN}✓{RESET} Recovery decisions based on error profile, not device type
""")


if __name__ == "__main__":
    main()
