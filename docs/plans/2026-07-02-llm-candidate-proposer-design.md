# LLM Candidate Proposer (shadow-first) — Design

Date: 2026-07-02 · Branch: `feat/llm-candidate-proposer`

## Context

BORA (Digital Discovery 2026 / IJCAI-25, arXiv:2501.16224) shows LLM reasoning
*hybridized with* Bayesian optimization beats BO-only, mainly by warm-starting
exploration and generating hypotheses — with a "trust score" that dynamically
decides when to lean on the LLM vs BO. The LLM proposes; it is not the authority.

That is exactly HELIOS's existing stance: Nexus and BO are pluggable *proposers*
and advisors; HELIOS keeps campaign-level decision authority. So an LLM belongs
in HELIOS as **one more plug-in proposer** — no special status over Nexus —
adopted only by HELIOS's own standard. See `docs/agent_architecture.md`.

Goal: introduce an LLM candidate proposer, shadow-first, that HELIOS's existing
credit/scoring governs, without changing routing, strategy selection, candidate
selection, or execution until real evidence justifies promotion.

## Decisions (brainstormed)

1. **Scope**: LLM proposes candidate points/regions (BORA-style), not hypotheses
   or route changes yet.
2. **Trust/decision**: reuse HELIOS's existing backend credit — the LLM is one
   more arm in `backend_memory` / `ContextualStrategyBandit`; its trust rises/falls
   from observed reward like any Nexus backend. No new trust mechanism.
3. **Trigger**: invoke only on plateau / high epistemic uncertainty (reuse
   existing `DiagnosticSignals` + plateau detection). Cost-controlled.
4. **Rollout**: full shadow-first → canary → promote.

## Architecture

New module `app/services/llm_candidate_proposer.py` (a proposer, shaped like a
backend arm):

- **Input** (read-only round context): parameter space (`dimensions`),
  best-so-far + recent observations, objective KPI/direction, and the trigger
  signals (plateau / epistemic uncertainty).
- **Call**: existing `llm_gateway.get_llm_provider()` (default `mock` → fallback);
  prompt asks for structured JSON: N candidate points + one-line hypothesis/reason
  each (BORA Comment shape). Provider injectable for tests.
- **Output**: `LLMProposal` (pydantic): proposed points, hypothesis/reason,
  model/temperature, created_at, evidence. Shadow-only artifact.

### Validation gate (deterministic, anti-hallucination) — stricter than BORA
Each proposed point must pass, in order, else rejected-with-reason (never executed):
1. **Schema / space legality** — params/types/bounds within the space (reuse
   `candidate_gen.space_from_dimensions`).
2. **Failure-zone check** — reject points in known failure zones (reuse
   `failure_zone_memory.recall_failure_zones`).
3. **Safety gate** — existing `SafetyAgent`/policy preflight.
4. **Hard constraints** — modeled physical/operational constraints.

Rejections are provenance-logged (a real calibration signal: LLM hallucination rate).

### Control flow (Phase A — shadow)
Triggered (plateau/uncertainty) → call LLM → build `LLMProposal` → run each point
through the validation gate → record proposal + validated/rejected points as an
**advisory** artifact (attached to the substrate snapshot / a proposer log) →
compare "LLM-proposed vs what HELIOS actually selected" (reuse
`shadow_trace_comparison` patterns). **Adopt nothing.** Gated by a new flag
`LLM_PROPOSER_SHADOW_ENABLED` (default off), read-only, fail-open.

## Trust / trigger / cost / reproducibility
- Trust = the LLM arm's UCB credit in `backend_memory` (canary+ only); auto-disable
  on poor reward / safety warnings.
- Trigger = plateau OR high epistemic uncertainty; per-campaign call cap; cache by
  (space fingerprint, best-so-far bucket, round bucket).
- Reproducibility = log prompt+response+model+temperature; mark advisory /
  non-deterministic; the deterministic validation gate keeps the decision core
  reproducible.

## Rollout & success criteria
- **Phase A (this branch, first increment)**: shadow proposer + validation gate +
  advisory recording + comparison. No adoption.
- **Exit-shadow criteria** (on real campaigns): high validity rate (low
  hallucination), and post-hoc "would the LLM-proposed point have scored/rewarded
  ≥ the chosen point?" (reward correlation / regret comparison).
- **Phase B (canary)**: register LLM as a low-weight bandit arm behind a flag with
  auto-disable.
- **Phase C (promote)**: standard promotion gate on positive, safe reward delta.

## Out of scope (YAGNI)
Fine-tuning; multi-model ensembling; LLM direct execution; hypothesis / route /
space-revision proposers (separate future increments — backlog B1/B7).

## Files
- New: `app/services/llm_candidate_proposer.py`, `tests/test_llm_candidate_proposer.py`
- `app/core/config.py` — add `LLM_PROPOSER_SHADOW_ENABLED` (default false)
- Later increment (shadow wiring): attach the proposal artifact where the
  substrate snapshot is recorded; extend the comparison analyzer.

## Boundaries / invariants
Shadow-only; read-only inputs; fail-open (LLM unavailable → classical path
unchanged); LLM only proposes, HELIOS's policy/credit decides; deterministic
validation gate; full provenance; no routing/strategy/candidate/execution change.

## Verification
- TDD with a deterministic mock LLM provider (injected).
- Tests: schema/failure-zone/safety/constraint rejection; trigger gating (only on
  plateau/uncertainty); fail-open on provider error; advisory recording; live
  selection unchanged in shadow mode; JSON-safe / deterministic with injected now.
- After shadow wiring: enable the flag on a dry-run/representative campaign and
  inspect proposal validity + comparison output.
