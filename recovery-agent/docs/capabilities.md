# Capabilities — Device/MCP Surface Area

This file is the **single source of truth** for what the agent can do with each device.
When you add MCP tools or new devices, update this file first.

## Capability Levels
- **L0 (observe-only)**: read state/health; cannot reliably change device state.
- **L1 (safe controls)**: can force a safe state (stop / cool down / safe shutdown) and verify it.
- **L2 (full controls)**: can set targets/modes, reset, self-test, and verify outcomes.

> Your goal (“more autonomous recovery”) usually means pushing devices from **L0 → L1 → L2** safely.

## Global signal dictionary (recommendation)
Define consistent names across devices:
- `current_temp`
- `target_temp`
- `max_temp`
- `heater_power`
- `sensor_health` (ok/degraded/bad/unknown)
- `device_heartbeat` (ok/missing)
- `last_error`

## Intent → tool mapping (template)
For each device, map high-level intents to one or more tool calls.

Example intent mapping format:
- Intent: `ENTER_SAFE_STATE`
  - Preferred: `[tool_name](...)`
  - Preconditions: [...]
  - Risks: [...]
  - Verification: [signal(s) that prove it worked]
  - Fallback: [another tool or ESCALATE]

---

## Devices

### heater_1 (template)
- Level: **[L0|L1|L2]**
- Connection: [serial:/dev/ttyUSB0 | tcp:host:port]

#### Observability
- Tool: `[read_state]`
  - Outputs: `current_temp`, `target_temp`, `heater_power`, `sensor_health`, `last_error`
  - Notes: [poll interval, units]

#### Safe controls (L1)
- Intent: `ENTER_SAFE_STATE`
  - Preferred: `[stop_heating]` or `[cool_down_to](temp=[SAFE_TEMP])`
  - Verification: `heater_power == 0` AND `current_temp trending_down`

#### Full controls (L2)
- Intent: `LOWER_TARGET`
  - Preferred: `[set_target](temp=...)`
  - Verification: `target_temp updated`

- Intent: `RESET_DEVICE`
  - Preferred: `[reset]`
  - Verification: `device_heartbeat ok` AND `last_error cleared`

- Intent: `RUN_SELF_TEST`
  - Preferred: `[self_test]`
  - Verification: `self_test == pass`

---

## Missing capability checklist (to increase autonomy)
When recovery feels "too human-dependent", it’s usually missing one of these:
- A reliable **safe-stop** tool + verification signal (L1)
- A **set-target** tool + verification (L2)
- A **reset/self-test** tool + verification (L2)
- A clear **sensor_health** signal (enables safe DEGRADE vs ABORT)
