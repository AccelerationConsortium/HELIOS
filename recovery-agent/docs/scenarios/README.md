# Scenario Cards (Fault Playbooks)

Each file in this folder defines how the agent handles **one fault scenario**.

## Why scenario cards?
- You can add new failure modes without rewriting core logic.
- They keep recovery policies explicit and reviewable.
- They scale with capabilities: as MCP/tools improve, the same scenario unlocks better actions.

## How to add a new scenario
1) Copy `template.md`
2) Fill: signals, trigger, allowed decisions by capability level, intents, verification, escalation.
3) Add tests/sim fault injection (if available).

## Naming
Use short, stable names:
- `overshoot.md`
- `timeout.md`
- `sensor_fail.md`
- `comms_drop.md`
- `stuck_actuator.md`

## Design rules
- Write steps as **intents**, not direct tool names.
- Always specify **required verification**.
- Always specify **when to escalate**.
