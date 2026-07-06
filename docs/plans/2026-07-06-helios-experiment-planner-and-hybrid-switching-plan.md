# HELIOS experiment planner and hybrid switching enhancement plan

**Date:** 2026-07-06
**Status:** Proposed enhancement plan
**Related:** `2026-06-26-method-comparison-benchmark-design.md`, `2026-06-26-bomcp-backend-integration-design.md`, `../adaptive_campaign_substrate.md`

## Motivation

The useful direction from the BO / LLM discussion is not "replace Bayesian
optimization with an LLM". The useful direction is:

> Use HELIOS as a context-aware scientific campaign meta-controller that can
> recognize problem structure, detect plateau, and switch among experiment
> planning methods while keeping execution deterministic.

HELIOS already has most of the lower-level pieces:

- `CampaignSnapshot` collects round history and campaign context for strategy
  selection.
- `select_strategy()` computes diagnostics, phase posterior, action utilities,
  backend ranking, evidence, and `StrategyTrace`.
- `CampaignDecisionLayer` can route above candidate generation to validation,
  literature, human observation, recovery, or constraint tightening.
- `CampaignMode` already provides a shadow scientific-activity vocabulary:
  BO optimization, validation, calibration, failure diagnosis, literature
  seeking, human observation, safety-constraint tightening, and stop.
- LLM planning is already separated from deterministic grounding: an LLM may
  propose a plan, but `plan_grounding.py`, safety validation, compiler, and
  execution remain deterministic.

The missing layer is a first-class experiment planning standard above backend
selection:

```text
ProblemProfile
  -> StrategyRecommendation
  -> PlateauAssessment
  -> PlannerSwitchDecision
  -> existing StrategyDecision / CampaignDecisionPlan / StrategyTrace
```

This should extend the existing dynamic strategy meta-controller. It should not
create a second planner stack or let an LLM directly steer hardware.

## Current coverage

| Capability | Current HELIOS support | Gap |
|---|---|---|
| Problem structure profiling | `method_advisor.problem_profile()` provides coarse `dims / modality / noise` buckets | Does not yet cover objective count, variable types, conditionals, budget, throughput, prior data, proxy reliability, constraints, failure history, or fidelity |
| Strategy recommendation | `select_strategy()` picks `explore / exploit / refine / stabilize` plus backend; `CampaignMode` picks broad scientific activity | No standalone experiment-planning recommendation such as DoE-first, validation-first, active learning, constrained BO, LLM-assisted proposal, or hybrid policy |
| Plateau detection | `compute_diagnostics()` exposes convergence, EI decay, improvement velocity, uncertainty, noise, drift; `CampaignDecisionLayer` has plateau + missing-literature rule | Plateau signals are not bundled into one auditable `PlateauAssessment`; no candidate-diversity or LLM-repetition plateau signal |
| Hybrid planner switching | Backend reranking exists; plateau can trigger refine or literature context | No explicit BO -> LLM, LLM -> BO, BO -> validation, optimization -> diagnostic switch policy |
| Deterministic execution boundary | LLM output is grounded through deterministic validation; live round loop is LLM-free by design | New hybrid planner must preserve this boundary and treat LLM output as proposal/context only |

## Architecture position

The enhancement belongs above backend ranking and below the human-facing
campaign intent layer.

```text
round observations + campaign context
        |
        v
CampaignSnapshot + CampaignRoundContext
        |
        v
Problem Structure Profiler
        |
        v
Plateau Detector
        |
        v
Strategy Recommendation Engine
        |
        v
Hybrid Planner Switcher
        |
        v
existing StrategyDecision / CampaignDecisionPlan / StrategyTrace
        |
        v
candidate generation, grounding, safety validation, execution, provenance
```

LLM, literature, and human observation are context/proposal sources. They do not
become execution authorities. Concrete candidates and protocols still pass
through schema validation, safety gates, candidate arbitration, protocol
compilation, connector execution, data logging, and provenance.

## Proposed modules

### A. Problem Structure Profiler

**New file:** `app/services/problem_profile.py`

Input:

- `CampaignSnapshot`
- optional `CampaignRoundContext`
- optional protocol/search-space metadata

Output:

- `ProblemProfile`

Initial fields:

- `dimensionality`: `low | medium | high`
- `variable_types`: continuous / integer / categorical / boolean / conditional
- `has_conditional_space`
- `noise_regime`: `low | medium | high | unknown`
- `throughput_regime`: `low | medium | high | unknown`
- `budget_pressure`: `low | medium | high | unknown`
- `objective_count`: integer
- `objective_regime`: single-objective / multi-objective
- `constraint_regime`: unconstrained / safety-constrained / feasibility-heavy
- `prior_data_regime`: data-poor / data-moderate / data-rich
- `proxy_reliability`: reliable / uncertain / gap-high
- `failure_recurrence`: none / occasional / recurring / dense
- `fidelity_regime`: single-fidelity / multi-fidelity / unknown
- `profile_key`: stable bucket key for decision tables and replay
- `evidence`: normalized evidence items

This should reuse and generalize `method_advisor.problem_profile()` rather than
renaming it into a parallel concept. The existing method advisor can become the
coarse backend-ranking adapter over this richer profile.

### B. Plateau Detector

**New file:** `app/services/plateau_detector.py`

Input:

- `CampaignSnapshot`
- `DiagnosticSignals`
- optional `CampaignRoundContext`
- optional recent `StrategyTrace` / candidate-pool provenance

Output:

- `PlateauAssessment`

Signals:

- `objective_stagnating`: improvement delta below threshold
- `convergence_plateau`: existing convergence detector says plateau
- `ei_exhausted`: EI decay below threshold
- `uncertainty_still_high`: model uncertainty or phase entropy remains high
- `uncertainty_not_decreasing`: uncertainty trend flat or worsening
- `ood_exploration_insufficient`: low coverage or low batch spread
- `candidate_diversity_collapsing`: repeated or near-duplicate proposals
- `failure_zone_repeated`: same failure class or failed parameter region repeats
- `proxy_gap_high_or_widening`: objective proxy no longer tracks functional target
- `llm_proposals_repetitive`: repeated LLM proposal fingerprints, when LLM proposal history exists

Output should include:

- `plateau_kind`: none / exploitation_plateau / knowledge_plateau / noise_plateau /
  failure_plateau / proxy_plateau / proposal_plateau
- `confidence`
- `recommended_response`: continue / refine / explore / stabilize / query_context /
  validate / diagnose / tighten_constraints
- `evidence`

### C. Strategy Recommendation Engine

**New file:** `app/services/strategy_recommendation.py`

Input:

- `ProblemProfile`
- `PlateauAssessment`
- `DiagnosticSignals`
- `CampaignModeDecision`
- available planners/backends

Output:

- `StrategyRecommendation`

Recommended strategy families:

- `doe_first`
- `bo`
- `constrained_bo`
- `multi_objective_bo`
- `active_learning`
- `llm_assisted_proposal`
- `validation_first`
- `diagnostic`
- `calibration`
- `diversity_exploration`
- `hybrid_switching_policy`

Examples:

| Profile / signal | Recommendation |
|---|---|
| data-poor + low/medium dimension | DoE first, then BO |
| improving objective + trusted search space | Continue BO |
| BO plateau + search space still credible | Change acquisition / batch strategy / refinement backend |
| BO plateau + prior knowledge missing | Query literature or LLM context proposal |
| high safety or feasibility constraints | Constrained BO or constraint tightening |
| proxy gap high | Validation-first or calibration |
| recurring failure zone | Diagnostic mode or failure-zone avoidance |
| multi-objective | Multi-objective BO / Pareto strategy |
| LLM proposals repetitive | Switch back to BO or diversity planner |

This module should produce an advisory recommendation first. It should not
directly mutate the live campaign path in the first implementation phase.

### D. Hybrid Planner Switcher

**New file:** `app/services/hybrid_planner_switcher.py`

Input:

- `StrategyRecommendation`
- `PlateauAssessment`
- current `StrategyDecision`
- optional recent planner history

Output:

- `PlannerSwitchDecision`

Initial deterministic transition table:

| Current state | Switch decision |
|---|---|
| BO has improvement | Continue BO |
| BO plateau, uncertainty low, search space credible | Switch acquisition / refinement / batch strategy |
| BO plateau, uncertainty high or coverage low | Explore / DoE / diversity planner |
| BO plateau, prior context missing | Query literature / LLM-assisted context proposal |
| LLM proposals repetitive | Switch to BO exploration or diversity planner |
| failure-zone dense | Constraint tightening or diagnostic mode |
| proxy gap high | Validation / calibration mode |
| safety risk high | Safety-constraint tightening before any planner |

The switch decision should be recorded in evidence and trace metadata. It should
not call an LLM directly. A switch to LLM means "request a validated proposal or
context artifact", not "execute an LLM-generated experiment".

## Integration points

### Strategy selector

Add optional profiling/switching outputs to `StrategyDecision.strategy_trace`.
The first implementation should be shadow-only:

- compute profile and plateau assessment near `compute_diagnostics()`
- add evidence to `StrategyTrace`
- keep selected backend unchanged unless a feature flag is enabled

Candidate feature flag:

- `enable_experiment_planner_shadow: bool = True`
- `enable_hybrid_planner_switching: bool = False`

### Campaign decision layer

Extend `CampaignRoundContext` with optional summaries:

- `problem_profile_summary`
- `plateau_summary`
- `strategy_recommendation_summary`
- `planner_switch_summary`

Use these to improve routing evidence:

- plateau + missing context -> `QUERY_LITERATURE`
- proxy plateau -> `RUN_VALIDATION` or `REVISE_OBJECTIVE`
- failure plateau -> `TIGHTEN_CONSTRAINTS` or `RECOVER_FAILURE`
- proposal plateau -> `PROPOSE_CANDIDATES` with BO/diversity planner preference

### Campaign mode

Do not replace `CampaignMode`. Extend it only if needed.

Possible additions:

- `DIVERSITY_EXPLORATION`
- `LLM_CONTEXT_PROPOSAL`
- `CONSTRAINED_OPTIMIZATION`

Keep priority order deterministic: stop, safety, blocking failure, attribution
uncertainty, calibration/diagnosis, proxy validation, context seeking, then
optimization.

### Orchestrator data flow

Current live flow already extracts round features:

```text
candidate execution
  -> run_kpi + candidate_params
  -> round_batch_kpis / round_batch_params
  -> all_kpis / all_params / all_rounds
  -> checkpoint_kpi()
  -> next round CampaignSnapshot
  -> compute_diagnostics()
  -> select_strategy()
```

The enhancement should add profile/switch artifacts at the same point where the
snapshot is built for adaptive strategy selection. It should not add database
writes in the first phase beyond existing trace/provenance paths.

## Development phases

### Phase 1: Shadow problem profile

Build `ProblemProfile` and tests.

Files:

- `app/services/problem_profile.py`
- `tests/test_problem_profile.py`
- optional small doc update in `docs/development_progress.md`

Acceptance:

- derives low/high dimension from `CampaignSnapshot`
- distinguishes continuous/categorical/log-scale spaces
- classifies data-poor vs data-rich from observations
- surfaces proxy/failure/budget summaries when context is present
- reuses current method-advisor buckets for backward compatibility

### Phase 2: Shadow plateau assessment

Build `PlateauAssessment` and tests.

Files:

- `app/services/plateau_detector.py`
- `tests/test_plateau_detector.py`

Acceptance:

- objective stagnation maps to exploitation plateau
- high noise maps to noise plateau / stabilize response
- high proxy gap maps to proxy plateau / validation response
- recurring failure maps to failure plateau / diagnostic response
- low coverage with plateau maps to knowledge plateau / explore or context response

### Phase 3: Strategy recommendation engine

Build recommendation rules over profile + plateau.

Files:

- `app/services/strategy_recommendation.py`
- `tests/test_strategy_recommendation.py`

Acceptance:

- data-poor low-dimensional profile recommends DoE-first
- improving trusted BO profile recommends continue BO
- constrained profile recommends constrained BO or constraint tightening
- high proxy gap recommends validation-first
- failure-zone recurrence recommends diagnostic mode
- recommendation emits stable evidence and reason strings

### Phase 4: Hybrid planner switcher

Build switch table and wire it as shadow evidence.

Files:

- `app/services/hybrid_planner_switcher.py`
- `tests/test_hybrid_planner_switcher.py`
- `app/services/strategy_selector.py`
- `app/services/strategy_models.py`

Acceptance:

- BO plateau + missing context -> LLM/literature context proposal decision
- BO plateau + credible search space -> acquisition/refinement switch decision
- LLM proposal repetition -> BO/diversity switch decision
- failure-zone dense -> diagnostic/constraint decision
- proxy gap high -> validation decision
- feature flag off means no live backend selection changes

### Phase 5: Decision-layer integration

Expose planner-switch summaries to campaign decisions.

Files:

- `app/services/decision_models.py`
- `app/services/round_context.py`
- `app/services/decision_layer.py`
- `tests/test_decision_layer.py`
- `tests/test_round_context.py`

Acceptance:

- existing decision-layer priority remains intact
- safety/failure/stop still override planner suggestions
- plateau/context evidence appears in `CampaignDecisionPlan.evidence`
- output remains JSON serializable and replayable

### Phase 6: Documentation and validation

Update user-facing architecture docs and validation scenarios.

Files:

- `README.md`
- `docs/development_progress.md`
- `docs/HELIOS_ARCHITECTURE_VALIDATION.md`
- `tests/test_offline_scenario_benchmarks.py` or a focused new scenario test

Validation:

- targeted unit tests for new modules
- contextual decision tests
- strategy trace serialization tests
- `scripts/run_validation_suite.sh` for the dynamic-strategy validation bundle

## Non-goals

- Do not make the LLM a live execution controller.
- Do not replace `select_strategy()` with a separate planner.
- Do not remove existing `CampaignMode`, `CampaignDecisionLayer`, or
  `StrategyTrace` contracts.
- Do not require Nexus, BO MCP, or an LLM provider for the default path.
- Do not enable live planner switching by default before shadow evidence and
  replay checks exist.

## Success criteria

The enhancement is successful when HELIOS can explain, for each round:

1. What problem structure it believes it is solving.
2. Whether the campaign is improving, uncertain, noisy, plateaued, failing, or
   proxy-misaligned.
3. Which strategy family is recommended and why.
4. Whether the planner should continue, switch backend/acquisition, ask for
   literature/LLM/human context, validate, diagnose, or tighten constraints.
5. Why that recommendation did or did not affect the live strategy.

The research framing should remain:

> HELIOS is a context-aware scientific campaign meta-controller. Planning can be
> agentic and context-rich; execution stays deterministic, validated, auditable,
> replayable, and safe.
