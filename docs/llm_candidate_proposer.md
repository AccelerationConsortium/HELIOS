# LLM Candidate Proposer (shadow-first)

Status: **shadow-only**. The LLM is one more pluggable *proposer* — like a Nexus
backend — with no special authority. HELIOS decides whether to adopt its
suggestions by its own standard (the existing backend credit / bandit). Nothing
here changes routing, strategy, candidate selection, or execution until real
evidence justifies promotion. Motivated by BORA (Digital Discovery 2026 /
IJCAI-25, arXiv:2501.16224); see the design note
[plans/2026-07-02-llm-candidate-proposer-design.md](plans/2026-07-02-llm-candidate-proposer-design.md).

## The LLM is a toolkit

Every provider implements the shared `LLMProvider` protocol
(`llm_gateway.py`). The decision path never depends on the vendor.

| Module | Responsibility |
|--------|----------------|
| `llm_providers/openai_compatible.py` | `OpenAICompatibleProvider` — any `/chat/completions` vendor (Moonshot/Kimi, DeepSeek, OpenAI-chat, self-hosted) via base_url + model |
| `llm_providers/replay.py` | `RecordingProvider` / `ReplayProvider` — capture real responses once, then develop & test offline and deterministically |
| `llm_providers/__init__.py` | `build_openai_compatible` (vendor presets) + `resolve_proposer_provider` (config-driven selection) |
| `llm_candidate_proposer.py` | `LLMCandidateProposer.propose` (async, fail-open), deterministic `validate_proposal` gate, trigger `should_invoke_llm_proposer`, offline `compare_llm_proposal_to_selection` |
| `llm_proposer_evidence.py` | `build_llm_proposer_evidence` — validity / overlap / novelty / rejection histogram / random-baseline comparison |
| `llm_proposer_canary.py` | Phase B: low-weight bandit arm with reward EMA + sticky auto-disable (`canary_arms`, `update_canary_state`) |

## Per-round flow (when enabled)

```
trigger (plateau / high epistemic uncertainty)
  -> LLMCandidateProposer.propose  (structured JSON: points + reasons; fail-open)
  -> validate_proposal gate:  schema/space | failure-zone | safety | constraints
  -> record LLMProposerShadow  (advisory log line "llm_proposer_shadow")
```
Offline: parse the log lines and build an evidence report against the round's
actual selection.

## Validation gate (deterministic, anti-hallucination)

Each proposed point must pass, else rejected-with-reason (never executed):
1. **schema / space** — params/types/bounds within the space,
2. **failure-zone** — `recall_failure_zones` (reject near known failed points),
3. **safety** — `make_safety_bounds_rejector` reuses `safety.evaluate_preflight`
   thresholds (max_temp_c / max_volume_ul) at the parameter level,
4. **constraints** — pluggable extra rejectors.

Rejections are provenance-logged (a real calibration signal).

## Feature flags (all default off)

| Env var | Effect |
|---------|--------|
| `LLM_PROPOSER_SHADOW_ENABLED` | Record LLM proposals as a parallel shadow track |
| `LLM_PROPOSER_CANARY_ENABLED` | Offer the LLM as a low-weight bandit arm (mechanism only; not wired to live selection) |
| `LLM_PROPOSER_PROVIDER` | Adapter to use: `kimi` / `moonshot` / `deepseek` / `openai` (empty => none) |
| `LLM_PROPOSER_MODEL` | Model id |
| `LLM_PROPOSER_BASE_URL` | Override the vendor preset base URL |
| `LLM_PROPOSER_API_KEY` | Key (falls back to `LLM_API_KEY`) |

With no provider configured the proposer fail-opens to empty proposals; the
classical path is unchanged.

## Shadow -> canary -> promote

- **Shadow (built)**: propose + validate + record + offline compare. Adopt
  nothing.
- **Exit-shadow criteria** (need a real LLM): high validity rate, overlap and
  novel-but-valid quality, and beating the random baseline.
- **Canary (mechanism built)**: `llm_proposer_canary` offers a low-weight arm
  with auto-disable; **not yet wired into live selection**.
- **Promote**: wire `canary_arms` into the backend selection and turn the flag
  on — gated on the shadow evidence above.

## What still requires a real LLM

The engineering above is complete and tested with mocks/stubs (no key). Real
numbers for validity / overlap / novelty / beating-random, and any promotion
decision, require running with a configured provider + key. The
record-then-replay providers let one real capture unlock offline, reproducible
development thereafter.
