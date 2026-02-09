---
name: utility
description: "General-purpose utilities — wait, log, cleanup, sample preparation, streaming, and artifact upload"
version: "1.0.0"
instrument: null
resource_id: null
primitives:
  - name: wait
    error_class: BYPASS
    safety_class: INFORMATIONAL
    params:
      duration_seconds: {type: number, description: "Time to wait in seconds"}
    description: "Pause execution for a specified duration"
    contract:
      preconditions: []
      effects: []
    timeout:
      seconds: 3600
      retries: 0
  - name: log
    error_class: BYPASS
    safety_class: INFORMATIONAL
    params:
      message: {type: string, description: "Message to record in the audit trail"}
      level: {type: string, optional: true, default: "info", description: "Log level (debug, info, warning, error)"}
    description: "Record a message in the experiment log"
    contract:
      preconditions: []
      effects: []
    timeout:
      seconds: 5
      retries: 0
  - name: heat
    error_class: BYPASS
    safety_class: REVERSIBLE
    params: {}
    description: "Activate heating element (placeholder for thermal control)"
    contract:
      preconditions: []
      effects: []
    timeout:
      seconds: 120
      retries: 1
  - name: upload_artifact
    error_class: BYPASS
    safety_class: INFORMATIONAL
    params: {}
    description: "Upload a file artifact to the object store"
    contract:
      preconditions: []
      effects: []
    timeout:
      seconds: 30
      retries: 1
  - name: cleanup.run_full
    error_class: CRITICAL
    safety_class: CAREFUL
    params: {}
    description: "Run the full cleanup sequence (rinse, dry, reset positions)"
    contract:
      preconditions: []
      effects:
        - "set:robot_homed:true"
    timeout:
      seconds: 300
      retries: 1
  - name: sample.prepare_from_csv
    error_class: CRITICAL
    safety_class: HAZARDOUS
    params:
      proposal_file: {type: string, description: "Path to the CSV file with sample definitions"}
      row_index: {type: integer, description: "Which row (sample) to prepare from the CSV"}
    description: "Prepare a sample according to specifications from a CSV file"
    contract:
      preconditions:
        - "pipettes_loaded"
      effects: []
    timeout:
      seconds: 600
      retries: 0
  - name: ssh.start_stream
    error_class: BYPASS
    safety_class: INFORMATIONAL
    params:
      filename_prefix: {type: string, description: "Prefix for the streamed data filename"}
    description: "Start streaming data via SSH from a remote instrument"
    contract:
      preconditions: []
      effects:
        - "set:ssh_streaming:true"
    timeout:
      seconds: 10
      retries: 1
  - name: ssh.stop_stream
    error_class: BYPASS
    safety_class: INFORMATIONAL
    params: {}
    description: "Stop the current SSH data stream"
    contract:
      preconditions: []
      effects:
        - "set:ssh_streaming:false"
    timeout:
      seconds: 10
      retries: 1
---

# Utility Primitives

These primitives provide general-purpose operations that don't belong to a
specific instrument. They handle timing, logging, data management, and
multi-step sequences.

## Categories

### Timing

**`wait`** — Pause between steps. Used for:
- Settling time after liquid dispensing
- Equilibration time before measurements
- Cooldown periods between heating steps

### Logging

**`log`** — Record messages in the audit trail. Every log entry includes
a timestamp and is stored permanently. Use for:
- Marking phase transitions
- Recording observations
- Debugging step sequences

### Cleanup

**`cleanup.run_full`** (CAREFUL) — Orchestrates a complete cleanup sequence:
1. Rinse all active labware
2. Dry with air
3. Return robot to home position
4. Reset PLC pump states

This is CAREFUL because a failed cleanup means contaminated equipment
for the next experiment, but the operation can be retried.

### Sample Preparation

**`sample.prepare_from_csv`** (HAZARDOUS) — Reads a CSV file containing
sample definitions (concentrations, volumes, well assignments) and executes
the preparation protocol. This is a compound operation that internally
dispatches multiple robot and PLC primitives. HAZARDOUS because it involves
irreversible liquid transfers.

### Data Streaming

**`ssh.start_stream`** / **`ssh.stop_stream`** — Control SSH-based data
streaming from remote instruments (e.g., cameras, sensors). INFORMATIONAL
because streaming is supplementary to the core experiment.

### Artifact Management

**`upload_artifact`** — Upload a file to the artifact store. INFORMATIONAL
because data can be manually uploaded after the run if this fails.

## Resource Locking

- `wait` and `log` require no resource locks
- `cleanup.run_full` acquires multiple locks internally (robot + PLC)
- `sample.prepare_from_csv` acquires the robot lock
- `ssh.*` operations do not require instrument locks

## Error Behavior

| Primitive | Error Class | Safety Class | On Failure |
|-----------|------------|-------------|------------|
| wait | BYPASS | INFORMATIONAL | Log and continue |
| log | BYPASS | INFORMATIONAL | Log and continue |
| heat | BYPASS | REVERSIBLE | Retry (1x), log and continue |
| upload_artifact | BYPASS | INFORMATIONAL | Retry (1x), log and continue |
| cleanup.run_full | CRITICAL | CAREFUL | Retry (1x), then abort |
| sample.prepare_from_csv | CRITICAL | HAZARDOUS | Abort immediately |
| ssh.start_stream | BYPASS | INFORMATIONAL | Retry (1x), log and continue |
| ssh.stop_stream | BYPASS | INFORMATIONAL | Retry (1x), log and continue |

*8 primitives. 5 INFORMATIONAL, 1 REVERSIBLE, 1 CAREFUL, 1 HAZARDOUS.*
