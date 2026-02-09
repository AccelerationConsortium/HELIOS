# Lab Orchestrator MVP

MVP backend that upgrades interactive lab agent behavior into a resident orchestrator.

Implemented core semantics:

- Triggering: time/event/external triggers.
- Persistent state: SQLite source of truth for runs, steps, artifacts, provenance, locks, approvals, sessions.
- Session/isolation: each scheduled run executes in an isolated worker subprocess.
- Determinism/replay: protocol is compiled into canonical DAG + `graph_hash`.
- Safety gates: preflight and runtime threshold/interlock checks.
- Concurrency: resource lease locks with fencing tokens.
- Auditability: append-only provenance events for every important state transition.

## Stack

- FastAPI
- SQLite (WAL mode)
- Local object store (`data/object_store/`)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

API base URL: `http://127.0.0.1:8000/api/v1`

## Example flow

1. Create a campaign with periodic loop:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/campaigns \
  -H 'content-type: application/json' \
  -d '{
    "name": "hourly-electrochem-loop",
    "cadence_seconds": 30,
    "created_by": "lab-admin",
    "inputs": {"instrument_id": "potentiostat-01"},
    "policy_snapshot": {
      "max_temp_c": 80,
      "max_volume_ul": 500,
      "require_human_approval": false
    },
    "protocol": {
      "steps": [
        {"id": "s1", "primitive": "aspirate", "params": {"volume_ul": 100}, "resources": ["liquid-handler-1"]},
        {"id": "s2", "primitive": "heat", "params": {"temp_c": 40}, "depends_on": ["s1"], "resources": ["heater-1"]},
        {"id": "s3", "primitive": "eis", "depends_on": ["s2"], "resources": ["potentiostat-01"]}
      ]
    }
  }'
```

2. Trigger an event run manually:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/triggers/event \
  -H 'content-type: application/json' \
  -d '{
    "actor": "qc-monitor",
    "payload": {"event": "qc_fail", "batch": "B-42"},
    "protocol": {
      "steps": [
        {"id": "retry", "primitive": "eis", "resources": ["potentiostat-01"]}
      ]
    },
    "inputs": {"instrument_id": "potentiostat-01"}
  }'
```

3. Check run and audit trail:

```bash
curl http://127.0.0.1:8000/api/v1/runs
curl http://127.0.0.1:8000/api/v1/runs/<RUN_ID>
curl http://127.0.0.1:8000/api/v1/runs/<RUN_ID>/events
curl http://127.0.0.1:8000/api/v1/runs/meta/locks
```

4. Approve a run if policy requires approval:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/runs/<RUN_ID>/approve \
  -H 'content-type: application/json' \
  -d '{"approver": "principal-investigator", "reason": "SOP-12 override accepted"}'
```

## Notes

- This is simulation-only: adapters currently execute a simulated instrument backend.
- Object store uses local filesystem paths in artifacts metadata.
- Replace simulated adapter with real drivers when you share instrument details.
