# Exp-Agent: Narrative Demo Narration Script

> Recording: `demo/recordings/exp-agent-narrative-demo.mp4` (65s)
>
> This document maps each section of the demo video to the agent architecture.

---

## Overview (0:00 - 0:02)

**Screen shows:** Pipeline overview — the 7-stage decision loop.

**Narration:**

This demo shows how the Exp-Agent handles real hardware faults during an experiment.
Every fault goes through the same 7-stage pipeline:

1. **SENSE** — detect the error from device telemetry
2. **CLASSIFY** — profile the error (unsafe? recoverable? severity?)
3. **ANALYZE** — analyze the telemetry signature (drift? stall? oscillation?)
4. **DECIDE** — choose a recovery strategy (RETRY / SKIP / DEGRADE / ABORT)
5. **EXECUTE** — run the recovery actions via MCP tool calls
6. **VERIFY** — confirm the device is in a safe state
7. **MEMORY** — persist the decision trail to the audit database

We demonstrate three fault scenarios that exercise all four decision types.

---

## Scenario 1: Tip Collision — ABORT (0:02 - 0:18)

**What happens:** A probe tip collides with the sample surface during the approach step. The force sensor reports an invalid reading (-999), indicating physical damage.

### 0:02 — Scenario banner

Red banner: "SCENARIO 1: Tip Collision — Sensor Failure". The fault mode is `sensor_fail`.

### 0:04 — Fault injection

The workflow is at step 3/5 (`approach` stage, `critical` criticality).
A red `FAULT INJECTED` tag appears: sensor reads -999 at z=0.003mm.

> **Architecture:** This simulates `HardwareError(type="sensor_fail")` from the `SimHeater` device layer.

### 0:10 — SENSE

`error.detected` event logged with `severity=high`.

> **Architecture:** `PipelineLogger.log_error_detected()` — first event in the correlation chain.

### 0:11 — CLASSIFY

`ErrorClassifier` profiles the error:
- `unsafe=True` — this is a physical hardware failure
- `recoverable=False` — sensor is broken, no software recovery possible
- `strategy=abort` — the only safe option

MCP tool call: `error_knowledge_base.lookup` confirms this is a known unsafe pattern.

> **Architecture:** `classify_error()` in `recovery/policy.py` → `ErrorProfile(unsafe=True, recoverable=False)`

### 0:12 — ANALYZE

Telemetry signature analysis shows `stall` (temp=25.0 at all timepoints — device was at ambient before collision).

MCP tool call: `pattern_matcher.match_signature` confirms stall pattern recommends abort.

> **Architecture:** `analyze_signature()` computes slope, variance, sign-changes over the temperature history.

### 0:13 — DECIDE: ABORT

`RecoveryAgent.decide()` returns `ABORT` with rationale: "Unsafe condition (sensor_fail), cannot recover."

Recovery action planned: `cool_down` (safe shutdown).

> **Architecture:** `decide_recovery()` in `policy.py` — `unsafe=True + recoverable=False → abort + cool_down`.

### 0:14-0:16 — EXECUTE + VERIFY

Recovery action `cool_down` executes successfully. Post-recovery device state: temp=25.0, heating=False, status=idle.

MCP tool call: `device_monitor.verify_state` confirms safe state.

### 0:16-0:17 — MEMORY

MCP tool call: `audit_database.write_trail` persists the 6-event decision trail.

### 0:17 — OUTCOME

Plan aborted at step `approach`. Remaining steps (measure, retract) will NOT execute.

> **Key takeaway:** `sensor_fail` → unsafe + non-recoverable → immediate ABORT + safe shutdown. The agent does NOT attempt retry because the hardware is physically damaged.

---

## Scenario 2: Current Overload — DEGRADE (0:19 - 0:37)

**What happens:** A heater overshoots its target temperature during preheat. Temperature drifts from 120 to 135. The agent detects the drift signature and degrades the target to a lower value, then patches all downstream steps.

### 0:19 — Scenario banner

Yellow banner: "SCENARIO 2: Current Overload — Temperature Overshoot". Fault mode is `overshoot`.

### 0:26 — Temperature drift visible

The demo shows 7 temperature readings during preheat:
```
t=0: 25.0°C
t=1: 60.0°C
t=2: 95.0°C
t=3: 115.0°C
t=4: 125.0°C  ← OVERSHOOT (+5°C)
t=5: 132.0°C  ← OVERSHOOT (+12°C)
t=6: 135.0°C  ← OVERSHOOT (+15°C)
```

> **Architecture:** These are `DeviceState` entries with `telemetry.temperature` values fed to `analyze_signature()`.

### 0:26 — Fault injection

Yellow `FAULT INJECTED` tag: Temperature overshoot 135°C (target was 120°C).

### 0:28 — CLASSIFY

`classify_error()` returns:
- `unsafe=True` (overshoot is a safety concern)
- `recoverable=True` (can recover by degrading target)
- `strategy=degrade`

MCP tool call: `error_knowledge_base.lookup` — "UNSAFE, requires immediate attention".

### 0:29 — ANALYZE: drift signature

Signature analysis detects `drift` with confidence=1.00 and avg_slope=18.333.
Telemetry history shows consistent upward trend.

MCP tool call: `pattern_matcher.match_signature` — "drift matches known pattern, recommended: degrade".

> **Architecture:** The slope exceeds `drift_slope_threshold=0.5`, so it's classified as drift. This is critical — drift signature enables the DEGRADE path. A stall signature here would trigger ABORT instead.

### 0:31 — DECIDE: DEGRADE

`RecoveryAgent.decide()` returns `DEGRADE` with rationale: "Unsafe condition with drift signature. Degrading to 110.0°C."

Three recovery actions planned:
1. `cool_down` — stop heating immediately
2. `set_temperature(110.0)` — set the new degraded target
3. `wait(5.0)` — stabilization time for drift mode

> **Architecture:** `decide_recovery()` — `unsafe + recoverable + drift → degrade`. `compute_degraded_target(120, "drift")` = 120 - 10 = 110.

### 0:31-0:35 — EXECUTE + VERIFY

All 3 recovery actions execute successfully. Post-recovery state shows heater ramping to new target.

### 0:36 — PlanPatch cascade

The demo shows the full PlanPatch structure:
- `original_target`: 120.0°C
- `degraded_target`: 110.0°C
- `overrides`: hold.temperature and measure.temperature → 110.0
- `relaxations`: postconditions updated from "~= 120.0" to "~= 110.0"
- `notes`: "Degraded from 120°C to 110°C at step preheat"

MCP tool call: `plan_manager.apply_patch` — "Patch applied: 2 downstream steps updated."

> **Key takeaway:** DEGRADE is not just about lowering the current step's target. The `PlanPatch` cascades to ALL downstream steps, updating both their action parameters (`overrides`) and their success criteria (`relaxations`). This prevents false failures on later steps that still reference the original 120°C target.

---

## Scenario 3: Measurement Timeout — RETRY then SKIP (0:38 - 0:65)

**What happens:** An optional measurement step times out because the reading doesn't stabilize. The agent retries once, fails again, and then skips the step because it's optional with `on_failure=skip`.

### 0:38 — Scenario banner

Green banner: "SCENARIO 3: Measurement Timeout — Optional Step Skip".

Step 4/6 (`measure` stage, `diagnostics`). Note: `criticality=optional`, `on_failure=skip`.

### 0:45 — Attempt 1: Fault injection

Yellow `FAULT INJECTED` tag: Postcondition timeout — reading=118.3, expected ~= 120.0.

> **Architecture:** `HardwareError(type="postcondition_failed")` — the post-check verified the action executed but the result didn't meet the expected tolerance.

### 0:46-0:49 — Attempt 1: Full pipeline

The full 7-stage pipeline runs:

- **CLASSIFY**: `unsafe=False`, `recoverable=True`, `strategy=retry`
- **ANALYZE**: signature=`stable` (slight variance around 120°C), confidence=0.80
- **DECIDE**: `RETRY` — "Postcondition failed again. Retry with 2s wait."

MCP tool calls show knowledge base lookup and pattern matching.

> **Architecture:** `postcondition_failed + stable + retry_count=1 → retry with 2s backoff` (policy line 448-453).

### 0:49 — RETRY displayed

"Policy says retry with backoff. Retry budget: 1/1 used."

### 0:54 — Attempt 2: Still failing

Second fault: reading=118.1, still outside tolerance.

### 0:55-0:62 — Attempt 2: Full pipeline (shared RecoveryAgent)

Pipeline runs again with the same `RecoveryAgent` instance (retry_count now increments to 2):

- **CLASSIFY**: same profile (safe, recoverable, retry)
- **ANALYZE**: same stable signature
- **DECIDE**: `ABORT` — "No matching recovery strategy for postcondition_failed"

The policy escalation logic: `retry_count=2 + no temperature target → fallback abort`.

> **Architecture:** The `RecoveryAgent` maintains `retry_counts` dict across calls. When `r >= 2` but there's no target temperature to degrade to, the policy falls through to the fallback abort. The `WorkflowSupervisor` then checks `step.on_failure == "skip"` and `criticality == "optional"` to convert this into a SKIP.

### 0:62 — SKIP displayed

"Retry budget exhausted on optional step."
"Step 'measure' has criticality=optional, on_failure=skip."
"Workflow cursor advances to next step → experiment continues."

MCP tool call: `workflow_engine.advance_cursor` — "Cursor advanced: measure(skipped) → cooldown."

> **Key takeaway:** The SKIP decision is a two-layer mechanism:
> 1. **Policy layer**: retry budget exhausted → abort (as a signal)
> 2. **Workflow layer**: `WorkflowSupervisor` interprets abort on an optional step as SKIP based on `on_failure` semantics

---

## Memory Store Summary (0:65)

27 events persisted across 3 scenarios, each with correlation IDs:
- Scenario 1 (sensor_fail → ABORT): 7 events
- Scenario 2 (overshoot → DEGRADE): 7 events + PlanPatch
- Scenario 3 (postcondition → RETRY → SKIP): 13 events (2 full pipeline passes + skip)

Icons map to event types: error detected, error classified, signature analyzed, decision made, recovery executed, recovery verified, step aborted/skipped, plan patched.

> **Architecture:** `MemoryStore` writes to `logs/narrative_demo_audit.jsonl`. In production, this would be a real database with correlation-ID-based trail reconstruction via `TrailAnalyzer`.

---

## Architecture Quick Reference

| Component | Location | Role |
|---|---|---|
| `SimHeater` | `devices/simulated/heater.py` | Simulated hardware with fault injection |
| `GuardedExecutor` | `executor/guarded_executor.py` | Safety-checked action execution |
| `classify_error()` | `recovery/policy.py` | Error profiling (unsafe/recoverable/strategy) |
| `analyze_signature()` | `recovery/policy.py` | Telemetry pattern detection (drift/stall/stable) |
| `RecoveryAgent.decide()` | `recovery/recovery_agent.py` | Recovery decision with retry tracking |
| `decide_recovery()` | `recovery/policy.py` | Policy engine — single point of truth |
| `PlanPatch` | `core/types.py` | Downstream override/relaxation cascade |
| `WorkflowSupervisor` | `orchestrator/workflow_supervisor.py` | Step cursor, criticality semantics, patch application |
| `PipelineLogger` | `logging/pipeline.py` | Structured event logging with correlation IDs |
| `TrailAnalyzer` | `logging/pipeline.py` | Decision trail reconstruction from audit log |

---

## Decision Type Summary

| Decision | Trigger | Condition | Recovery Actions |
|---|---|---|---|
| **ABORT** | sensor_fail | unsafe + non-recoverable | `cool_down` → safe shutdown |
| **DEGRADE** | overshoot + drift | unsafe + recoverable + drift signature | `cool_down → set_temperature(degraded) → wait` + PlanPatch |
| **RETRY** | postcondition_failed | safe + recoverable + budget remaining | `wait(backoff)` |
| **SKIP** | retry exhausted | optional step + on_failure=skip | cursor advance, no recovery |
