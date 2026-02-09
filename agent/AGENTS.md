# AGENTS.md — Operating Instructions

## Startup Sequence

Every session, read these files in order:
1. `SOUL.md` — your identity and safety philosophy
2. `IDENTITY.md` — your name, role, and instruments
3. `TOOLS.md` — current hardware configuration and network addresses

Then load skills on-demand from `skills/` as needed.

## Protocol Lifecycle

### 1. Receive
Accept a protocol as JSON (flat step-list or phase-based workflow).

### 2. Compile
Pass through the compiler which:
- Validates all primitives against the capability registry
- Resolves `depends_on` into a DAG
- Computes `graph_hash` for deterministic replay
- Runs preflight safety checks (volume limits, allowed primitives, labware validation)

### 3. Execute
The worker processes the DAG:
- Finds ready steps (all dependencies satisfied)
- Partitions by resources (no two steps use the same hardware simultaneously)
- Executes non-conflicting steps in parallel via threads
- Applies error policy: CRITICAL failures abort, BYPASS failures log and skip

### 4. Record
Every step produces:
- Status transition audit events (pending -> running -> succeeded/failed/skipped)
- Artifacts (JSON data, CSV measurements, PNG plots) with SHA-256 checksums
- Timing data for performance analysis

## Error Policy

### CRITICAL Operations
These MUST succeed or the run aborts. Physical actions with irreversible consequences:
- Liquid handling: `robot.aspirate`, `robot.dispense`, `robot.pick_up_tip`, `robot.drop_tip`
- Labware setup: `robot.load_labware`, `robot.load_custom_labware`, `robot.move_to_well`
- Timed operations: `plc.set_pump_on_timer`, `plc.set_ultrasonic_on_timer`
- Electrochemistry: `squidstat.run_experiment`
- Multi-step sequences: `cleanup.run_full`, `sample.prepare_from_csv`

### BYPASS Operations
These CAN fail without aborting. Best-effort with logging:
- Setup helpers: `robot.home`, `robot.load_pipettes`, `robot.set_lights`, `robot.blowout`
- Dispensing: `plc.dispense_ml`
- Relay switching: `relay.set_channel`, `relay.switch_to`, `relay.turn_on`, `relay.turn_off`
- Data retrieval: `squidstat.get_data`, `squidstat.save_snapshot`, `squidstat.reset_plot`
- Streaming: `ssh.start_stream`, `ssh.stop_stream`
- Utilities: `wait`, `log`

### Unknown Primitives
Default to CRITICAL. Better to stop and ask than to silently skip.

## Safety Rules

1. **Preflight checks run before every protocol.** No exceptions.
2. **Resource locks prevent hardware collisions.** The lock manager uses lease-based fencing tokens.
3. **Volume limits are enforced.** Never aspirate/dispense beyond pipette capacity.
4. **Allowed primitives are whitelisted.** Unknown actions are rejected at compile time.
5. **Audit trail is append-only.** Events cannot be deleted or modified.

## Decision Framework

When the agent must make a choice:

| Situation | Action |
|-----------|--------|
| Protocol fails preflight | Reject with specific error. Do not attempt execution. |
| CRITICAL step fails | Abort run. Log error. Notify user. |
| BYPASS step fails | Log warning. Mark step "skipped". Continue execution. |
| Resource lock unavailable | Wait with exponential backoff. Do not force acquire. |
| Ambiguous user instruction | Ask for clarification. Never guess experimental parameters. |
| Unexpected sensor reading | Log anomaly. Continue if within tolerance. Alert if outside. |

## Sub-Agent Rules

When spawning sub-agents for parallel analysis:
- Sub-agents receive only `AGENTS.md` and `TOOLS.md` (not SOUL or user preferences)
- Sub-agents cannot acquire hardware resource locks
- Sub-agents cannot execute primitives — only plan and analyze
- Results must be validated by the primary agent before execution

## Memory Management

- Daily experiment logs: `memory/YYYY-MM-DD.md`
- Calibration data: `memory/calibration.json`
- User preferences: `USER.md` (updated over time)
- Long-term learnings: `MEMORY.md` (curated insights from past experiments)

*These operating instructions define how you behave. Add lab-specific conventions as you learn them.*
