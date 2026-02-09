# Lab Orchestrator MVP Architecture

## Control Plane

- Trigger ingress (`/triggers/time`, `/triggers/event`, `/triggers/external`)
- Campaign loop for periodic time triggers
- Session routing (`session_key`)
- Scheduler claims scheduled runs and launches isolated workers

## Execution Plane

- One subprocess worker per run (`python -m app.worker --run-id <id>`)
- Simulated instrument adapter executes primitives (`aspirate`, `heat`, `eis`, `wait`, `upload_artifact`)
- Runtime safety checks and interlock checks before each step

## State Plane

- SQLite source of truth:
  - `campaigns`, `runs`, `run_steps`, `artifacts`
  - `provenance_events`, `resource_locks`, `approvals`, `instrument_sessions`
- Object store: filesystem directory (`data/object_store/`)

## Determinism and Replay

- Protocol compiled into canonical DAG representation
- `graph_hash = sha256(protocol + inputs + policy_snapshot + compiled_graph)`
- Same inputs produce same graph hash and execution graph

## Safety and Compliance

- Preflight gate:
  - Primitive allowlist
  - Threshold policy checks (`max_temp_c`, `max_volume_ul`)
  - Optional `require_human_approval`
- Runtime gate:
  - Interlock and cooling checks
  - Threshold checks at step execution time

## Concurrency

- Resource lease lock table with fencing token
- Worker acquires resource locks before each step and releases after execution
- Prevents simultaneous use of instrument/deck slots across runs

## Audit

- Append-only provenance events for:
  - run creation, rejection, claim, completion
  - step state transitions
  - resource lock acquire/release
  - artifact creation
  - approvals
  - instrument session start/end
