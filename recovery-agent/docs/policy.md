# Policy — Safety & Recovery (exp-agent)

This document defines **stable, non-negotiable rules** for recovery-aware execution.
Keep it short and strict. Put device-specific details in `capabilities.md` and fault logic in `scenarios/*.md`.

## Mission
- Primary: keep hardware and environment safe.
- Secondary: complete the experiment **only if** safety invariants remain satisfied.

## Hard invariants (must never be violated)
1) **Unknown == unsafe**
   - If required signals are missing/unreliable, treat state as **UNSAFE**.
2) **No risk-increasing actions in UNSAFE state**
   - When UNSAFE, only actions that *strictly reduce risk* are permitted.
3) **Guarded execution always**
   - Every action must pass: **pre-check → safety-check → post-verify**.
4) **Bounded autonomy**
   - The agent may recover autonomously **within defined budgets** (below). Exceeding budgets ⇒ escalate.

## Decision outcomes (allowed high-level decisions)
- **RETRY**: transient failure, attempt again after a backoff.
- **DEGRADE**: continue at reduced goal (lower target, slower rate, safer mode).
- **SKIP**: skip non-critical step if safe (rare in hardware control; use carefully).
- **ABORT**: enter safe state, stop the run.
- **ESCALATE**: notify human with a structured report (can accompany ABORT).

## Capability gating (what autonomy is legal)
Autonomy should scale with device/MCP capability.

- **L0 — Observe-only**: can read state/health, cannot change device state safely.
  - Allowed: ESCALATE (and optionally "wait")
  - Not allowed: RETRY/DEGRADE/ABORT (because you can’t guarantee safety transitions)

- **L1 — Safe controls**: can force safe state (stop heating / cool down / safe shutdown).
  - Allowed: ABORT autonomously; limited RETRY when clearly safe.

- **L2 — Full controls**: can adjust targets/modes, reset, self-test, and verify.
  - Allowed: RETRY + DEGRADE per scenario cards.

> Rule of thumb: **DEGRADE requires L2**, because it means “continue operating” and must be verifiable.

## Budgets (bounded recovery)
Fill these values for your lab.
- Max retries per step: **[N]**
- Backoff: **[e.g., 1s, 2s, 4s]**
- Max total recovery time per run: **[T]**
- Max number of degradations per run: **[D]**

Escalate when:
- A fault is **unknown/unclassified**
- Verification is unavailable or contradictory
- Budgets exceeded
- The same fault repeats after a successful recovery

## Required reporting (for auditability)
Every recovery attempt must record:
- fault_id + classification + confidence
- decision (RETRY/DEGRADE/ABORT/ESCALATE)
- actions taken (intents + actual tool calls)
- before/after key signals
- verification result and remaining budgets

## Canonical “intents” (keep scenarios flexible)
Scenarios should express steps as **intents**; `capabilities.md` maps intents to actual tools.
- `REDUCE_RISK_HEAT`
- `ENTER_SAFE_STATE`
- `RESTORE_SENSOR_TRUST`
- `RETRY_COMMUNICATION`
- `LOWER_TARGET`
- `SLOW_RAMP`
- `RESET_DEVICE`
- `RUN_SELF_TEST`
- `NOTIFY_HUMAN`
