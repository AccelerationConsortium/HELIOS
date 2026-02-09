# OTbot Hardware Testing Guide

Progressive 4-stage plan for connecting OTbot to real hardware and validating end-to-end operation.

## Prerequisites

| Component | Details |
|-----------|---------|
| OT-2 Robot | Network-accessible via IP (default `localhost`) |
| PLC (Modbus TCP) | Auto-discovered by `OT_PLC_Client_Edit` |
| USB Relay | Serial port (default `COM11` / `/dev/ttyUSBx`) |
| SquidStat Potentiostat | Serial/USB connection |
| Python environment | All dependencies installed, 745+ tests passing |

## Environment Variables

```bash
# Core adapter settings
export ADAPTER_MODE="battery_lab"      # "simulated" (default) | "battery_lab"
export ADAPTER_DRY_RUN="true"          # "true" = log-only, no real hardware commands
export LLM_PROVIDER="mock"             # "mock" | "anthropic"

# Device connections
export ROBOT_IP="localhost"        # OT-2 IP address
export RELAY_PORT="COM11"              # USB relay serial port (Linux: /dev/ttyUSB0)
export SQUIDSTAT_PORT=""               # SquidStat serial port (placeholder)

# LLM (for C3 Reviewer / C5 Evolution with real LLM)
export LLM_API_KEY="sk-ant-..."        # Anthropic API key
export LLM_MODEL="claude-sonnet-4-20250514"
```

---

## Stage 1: Dry-Run Smoke Test

**Goal**: Validate the full execution pipeline (scheduler → worker → dispatcher → adapter) with hardware logging only — no real device commands sent.

**Risk**: None. All actions are logged, not executed.

### Configuration

```bash
export ADAPTER_MODE="battery_lab"
export ADAPTER_DRY_RUN="true"
export LLM_PROVIDER="mock"
```

### Steps

1. **Start the server**:
   ```bash
   python -m app.main
   ```

2. **Submit a test run via API**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/runs \
     -H "Content-Type: application/json" \
     -d '{
       "trigger_type": "manual",
       "protocol": {
         "name": "dry-run-smoke-test",
         "version": "1.0",
         "steps": [
           {"key": "home",   "primitive": "robot.home",   "params": {}},
           {"key": "wait_1", "primitive": "wait",          "params": {"seconds": 1}},
           {"key": "log_ok", "primitive": "log",           "params": {"message": "dry-run passed"}}
         ]
       },
       "inputs": {},
       "created_by": "test-user"
     }'
   ```

3. **Verify execution**:
   ```bash
   # Check run status
   curl http://localhost:8000/api/v1/runs/{run_id}

   # Expected: status = "succeeded", all steps completed
   # Logs should show "[DRY-RUN] robot.home ..." entries
   ```

### Acceptance Criteria

| Criterion | Expected |
|-----------|----------|
| Run status | `succeeded` |
| All 3 steps | `succeeded` |
| Log output | `[DRY-RUN]` prefix on all hardware actions |
| No hardware errors | Zero connection/timeout errors |
| KPI extraction | `step_duration_s` KPIs stored for each step |

---

## Stage 2: Single-Device Connection Tests

**Goal**: Connect one device at a time, verify basic commands, then disconnect. Isolate per-device issues before full integration.

**Risk**: Low. Single commands, manually supervised.

### 2A: OT-2 Robot

```bash
export ADAPTER_MODE="battery_lab"
export ADAPTER_DRY_RUN="false"
export ROBOT_IP="localhost"
```

**Test protocol** — robot-only commands:
```json
{
  "name": "ot2-connection-test",
  "version": "1.0",
  "steps": [
    {"key": "home",       "primitive": "robot.home",            "params": {}},
    {"key": "lights_on",  "primitive": "robot.set_lights",      "params": {"on": true}},
    {"key": "lights_off", "primitive": "robot.set_lights",      "params": {"on": false}},
    {"key": "load_pip",   "primitive": "robot.load_pipettes",   "params": {
      "left": "p1000_single_gen2", "right": "p300_single_gen2"
    }},
    {"key": "load_lw",    "primitive": "robot.load_labware",    "params": {
      "slot": "1", "labware": "opentrons_96_tiprack_1000ul"
    }},
    {"key": "pickup",     "primitive": "robot.pick_up_tip",     "params": {
      "pipette": "left", "location": "1:A1"
    }},
    {"key": "drop",       "primitive": "robot.drop_tip",        "params": {
      "pipette": "left"
    }}
  ]
}
```

**Verify**:
- Robot homes successfully
- Lights toggle visually confirmed
- Pipettes loaded without errors
- Tip pickup/drop completes

### 2B: PLC (Modbus TCP)

```json
{
  "name": "plc-connection-test",
  "version": "1.0",
  "steps": [
    {"key": "pump_1ml",     "primitive": "plc.dispense_ml",          "params": {"ml": 1.0}},
    {"key": "pump_timer",   "primitive": "plc.set_pump_on_timer",    "params": {"seconds": 2}},
    {"key": "ultra_timer",  "primitive": "plc.set_ultrasonic_on_timer", "params": {"seconds": 2}}
  ]
}
```

**Verify**:
- PLC connection established (check Modbus TCP handshake in logs)
- Pump dispenses ~1 mL visually confirmed
- Timer functions complete without error

### 2C: USB Relay

```bash
export RELAY_PORT="/dev/ttyUSB0"   # adjust for your system
```

```json
{
  "name": "relay-connection-test",
  "version": "1.0",
  "steps": [
    {"key": "ch1_on",   "primitive": "relay.set_channel", "params": {"channel": 1, "state": "on"}},
    {"key": "wait_2s",  "primitive": "wait",              "params": {"seconds": 2}},
    {"key": "ch1_off",  "primitive": "relay.set_channel", "params": {"channel": 1, "state": "off"}},
    {"key": "switch",   "primitive": "relay.switch_to",   "params": {"channel": 2}}
  ]
}
```

**Verify**:
- Relay channels toggle (LED indicators or multimeter)
- No serial port errors in logs

### 2D: SquidStat Potentiostat

> **Note**: `SquidstatController` class and `squidstat.get_data` handler are currently stubs.
> Before testing, implement the SquidStat integration:
> 1. Instantiate `SquidstatController` in `BatteryLabAdapter.connect()`
> 2. Wire `squidstat_port` from `config.py` to the controller
> 3. Implement `_handle_squidstat_get_data()` in dispatcher

```json
{
  "name": "squidstat-connection-test",
  "version": "1.0",
  "steps": [
    {"key": "run_eis", "primitive": "squidstat.run_experiment", "params": {
      "experiment_name": "EIS_test",
      "channel": 0,
      "elements": [{"type": "eis", "start_freq": 100000, "end_freq": 0.1, "amplitude": 0.01}],
      "num_cycles": 1,
      "csv_filename": "test_eis.csv"
    }},
    {"key": "get_data", "primitive": "squidstat.get_data", "params": {}},
    {"key": "snapshot",  "primitive": "squidstat.save_snapshot", "params": {
      "phase": "test", "cycle": 1
    }}
  ]
}
```

**Verify**:
- EIS experiment starts and completes
- Data retrieval returns measurement values
- Snapshot saved to configured folder

---

## Stage 3: Full Integration Test

**Goal**: Run a complete multi-device protocol that exercises robot + PLC + relay + SquidStat in sequence, simulating a real experiment.

**Risk**: Medium. Multi-device coordination; monitor closely.

### Configuration

```bash
export ADAPTER_MODE="battery_lab"
export ADAPTER_DRY_RUN="false"
export LLM_PROVIDER="mock"      # or "anthropic" if testing C3/C5
```

### Test Protocol

Use the built-in OER screening protocol pattern or a simplified version:

```json
{
  "name": "integration-test-v1",
  "version": "1.0",
  "steps": [
    {"key": "home",          "primitive": "robot.home",            "params": {}},
    {"key": "load_tips",     "primitive": "robot.load_labware",    "params": {
      "slot": "1", "labware": "opentrons_96_tiprack_1000ul"
    }},
    {"key": "load_plate",    "primitive": "robot.load_labware",    "params": {
      "slot": "2", "labware": "corning_96_wellplate_360ul_flat"
    }},
    {"key": "load_pip",      "primitive": "robot.load_pipettes",   "params": {
      "left": "p1000_single_gen2"
    }},
    {"key": "pickup_tip",    "primitive": "robot.pick_up_tip",     "params": {
      "pipette": "left", "location": "1:A1"
    }},
    {"key": "aspirate",      "primitive": "robot.aspirate",        "params": {
      "pipette": "left", "volume_ul": 100, "location": "2:A1"
    }},
    {"key": "dispense",      "primitive": "robot.dispense",        "params": {
      "pipette": "left", "volume_ul": 100, "location": "2:B1"
    }},
    {"key": "drop_tip",      "primitive": "robot.drop_tip",        "params": {
      "pipette": "left"
    }},
    {"key": "relay_switch",  "primitive": "relay.switch_to",       "params": {"channel": 1}},
    {"key": "wait_settle",   "primitive": "wait",                  "params": {"seconds": 5}},
    {"key": "run_eis",       "primitive": "squidstat.run_experiment", "params": {
      "experiment_name": "integration_eis",
      "channel": 0,
      "elements": [{"type": "eis", "start_freq": 100000, "end_freq": 0.1, "amplitude": 0.01}],
      "num_cycles": 1,
      "csv_filename": "integration_test.csv"
    }},
    {"key": "snapshot",      "primitive": "squidstat.save_snapshot", "params": {
      "phase": "integration_test", "cycle": 1
    }}
  ]
}
```

### Validation Checklist

| Phase | Check | Pass? |
|-------|-------|-------|
| Robot | Home, tip pickup, liquid transfer complete | |
| Relay | Channel switched, no error | |
| Wait | 5s delay observed | |
| SquidStat | EIS experiment completes, CSV generated | |
| KPIs | `volume_accuracy_pct`, `impedance_ohm`, `step_duration_s` extracted | |
| Artifacts | CSV file stored in object store | |
| Run Status | Overall status = `succeeded` | |

### Post-Run Verification

```bash
# Check run details
curl http://localhost:8000/api/v1/runs/{run_id}

# Check KPIs
curl http://localhost:8000/api/v1/runs/{run_id}/kpis

# Check artifacts
curl http://localhost:8000/api/v1/runs/{run_id}/artifacts

# Check step details
curl http://localhost:8000/api/v1/runs/{run_id}/steps
```

---

## Stage 4: Autonomous Campaign Loop

**Goal**: Run a multi-round optimization campaign where OTbot autonomously generates candidates, executes experiments, evaluates results, and evolves parameters.

**Risk**: High. Fully autonomous hardware operation. Requires active monitoring.

### Configuration

```bash
export ADAPTER_MODE="battery_lab"
export ADAPTER_DRY_RUN="false"
export LLM_PROVIDER="anthropic"    # Real LLM for C3 review + C5 evolution
export LLM_API_KEY="sk-ant-..."
```

### Create Campaign

```bash
curl -X POST http://localhost:8000/api/v1/campaigns \
  -H "Content-Type: application/json" \
  -d '{
    "name": "eis-optimization-v1",
    "cadence_seconds": 300,
    "protocol": {
      "name": "OER_SCREENING",
      "version": "1.0",
      "steps": [
        ... (full protocol steps)
      ]
    },
    "inputs": {
      "target_impedance_ohm": 50.0,
      "max_temperature_c": 80.0
    },
    "policy": {
      "max_retries": 2,
      "require_approval_above": "HAZARDOUS"
    }
  }'
```

### Campaign Loop Execution

The campaign loop (`app/services/campaign_loop.py`) operates as:

```
Goal → Generate Candidates (C4) → Execute Runs → Evaluate KPIs (C2)
  → Review (C3) → Evolve (C5) → Check Convergence → Repeat or Stop
```

**Using the programmatic API**:
```python
from app.services.campaign_loop import run_campaign, CampaignGoal
from app.services.candidate_gen import ParameterSpace, SearchDimension

goal = CampaignGoal(
    objective_kpi="impedance_ohm",
    direction="minimize",
    target_value=50.0,
    max_rounds=10,
    batch_size=3,
    strategy="prior_guided",
)

space = ParameterSpace(
    dimensions=[
        SearchDimension(name="amplitude", low=0.005, high=0.05, dim_type="continuous"),
        SearchDimension(name="num_cycles", low=1, high=10, dim_type="integer"),
    ],
    constraints=[],
)

def execute_fn(run_ids: list[str]) -> None:
    """Execute runs on real hardware via scheduler."""
    for run_id in run_ids:
        # Worker picks these up from the dispatch loop
        pass

result = run_campaign(goal, space, execute_fn, campaign_id="camp-001")
print(f"Best KPI: {result.best_kpi}, Converged: {result.converged}")
```

### Monitoring During Campaign

```bash
# Watch campaign progress
watch -n 5 'curl -s http://localhost:8000/api/v1/campaigns/{id} | python -m json.tool'

# Check latest run
curl http://localhost:8000/api/v1/runs?campaign_id={id}&limit=1

# Check convergence status (from logs)
tail -f data/logs/otbot.log | grep -i "convergence\|campaign\|round"

# Emergency stop
curl -X PATCH http://localhost:8000/api/v1/campaigns/{id} \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

### Acceptance Criteria

| Criterion | Expected |
|-----------|----------|
| Multiple rounds complete | ≥3 rounds with successful runs |
| KPI trend | Impedance values trend toward target |
| C3 reviews generated | Each run has a structured review |
| C5 evolution | At least 1 evolution proposal created |
| Convergence detection | `detect_convergence()` returns `plateau` or `improving` |
| No safety violations | Zero CRITICAL failures unhandled |
| Recovery from faults | At least 1 retry on transient failure |

---

## Known Hardware Stubs (Pre-Implementation Required)

Before Stage 2D and Stage 3, these stubs need real implementations:

| Stub | File | Issue |
|------|------|-------|
| `SquidstatController` | `app/adapters/battery_lab.py` | `self._squidstat = None` — class never instantiated |
| `_handle_squidstat_get_data()` | `app/hardware/dispatcher.py` | Returns `{"status": "ok"}` stub |
| `SshDataStreamer` | `app/hardware/dispatcher.py` | `_handle_ssh_start/stop` are stubs |
| `sample.prepare_from_csv` | `app/hardware/dispatcher.py` | CSV chemicals hardcoded to battery domain |
| `cleanup.run_full` | `app/hardware/dispatcher.py` | Stub implementation |
| `squidstat_port` wiring | `app/core/config.py` | Setting exists but not passed to controller |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ConnectionRefusedError` on OT-2 | Robot not on network | Check `ROBOT_IP`, ping robot |
| Serial port permission denied | No USB access | `sudo chmod 666 /dev/ttyUSB0` or add user to `dialout` group |
| PLC timeout | Modbus TCP not responding | Check PLC power, network cable |
| SquidStat import error | `SquidstatPyLibrary` not installed | `pip install squidstat` (if available) |
| Run stuck in `running` | Worker thread hung | Check logs, restart server; run auto-fails on next reap cycle |
| KPIs not extracted | Artifacts missing expected fields | Check artifact JSON matches KPI extractor expectations |
