# HELIOS Development Progress

Consolidated backlog of **not-yet-developed** capabilities, folded in from the
now-removed planning notes `v2.md`, `v3.md`, `data_layer.md`,
`enhancement0627.md`, and `validation.md` (deleted 2026-07-01). Delivered items
from those notes were dropped; what remains here is the outstanding work plus a
brief map of what already shipped.

Legend: **not started** · **partial** (some infra exists, not wired/proven).

---

## A. Delivered (pointers only)

- **Adaptive campaign substrate (shadow)** — ObjectiveState, distributional
  FailureAttribution, CampaignMode transition table (incl.
  SAFETY_CONSTRAINT_TIGHTENING), DynamicActionSpace, Value-of-Information,
  aggregate snapshot, shadow-trace comparison. See
  [adaptive_campaign_substrate.md](adaptive_campaign_substrate.md).
- **Dynamic strategy meta-controller** — two-layer action taxonomy
  (`CampaignIntent` + `OptimizationMode`), phase posterior, evidence-based
  scoring, safety gates. See README → Architecture.
- **Context / memory / logging** — campaign context, objective stack + proxy
  gap, typed failure taxonomy (`failure_signatures`), backend performance memory
  + `ContextualStrategyBandit`, candidate-pool memory (recall), cross-campaign
  failure-zone memory, decision trace / evidence / outcome / reward / replay.

Everything shipped is read-only / fail-open / shadow or approval-gated and does
not change live candidate selection by default.

---

## B. Backlog — adaptive decision & scientific reasoning

| # | Item | Origin | Status | Notes |
|---|------|--------|--------|-------|
| B1 | **HypothesisState** — active/supported/contradicted hypotheses, discriminating experiments | v3 §3 | not started | VoI `expected_hypothesis_resolution` stays 0 until this exists |
| B2 | **Instrument / runtime belief state + PUDA telemetry** — calibration confidence, drift, telemetry anomalies | v3 §6 | not started | Would let a bad reading be attributed to the instrument, not the sample |
| B3 | **OperationalAbstractionLearner** (Phase 6) — promote repeated successful action sequences to reusable ops (proposal-only) | v3 §8 | not started | Explicitly deferred until several real shadow logs are reviewed |
| B4 | **Campaign-level memory beyond candidate/failure** — objective patterns, strategy-success-by-phase, hypothesis-resolution patterns, useful context queries, per-instrument reliability | v3 §9 | not started | Higher tier than failure-zone memory |
| B5 | **StrategyClass scientific-action dimension** on the selector (PARAMETER_OPTIMIZATION / HYPOTHESIS_DISCRIMINATION / CALIBRATION / …) | v3 §10 | partial | `OptimizationMode`/`CampaignIntent` exist but not this explicit class |
| B6 | **Objective staging / fidelity escalation + ObjectiveManager** — proxy → mechanism → functional → deployment ladder, staged scoring, objective versioning wired into selection | data_layer §4; enh L7 | partial | `ObjectiveStack`/`ObjectiveState`/proxy_gap exist; staging/escalation and `objective_transitions` consumption not wired |
| B7 | **Parameter-space / synthesis-route revision as first-class** — `SpaceRevision`, `ParameterSpacePolicy`, route switching | enh L14; v3-adjacent | partial | `revise_space` intent + `space_revision` records exist but not consumed |

---

## C. Backlog — data / representation / evaluation infrastructure

| # | Item | Origin | Status | Notes |
|---|------|--------|--------|-------|
| C1 | **OptimizationDataContract** — unify the 8 scattered contract types (ResultPacket / Candidate / ObjectiveSpec / OutcomeConstraint / Observation / FailureRegionModel / DecisionResult / ProvenanceLogger) into one spec | data_layer §1 | not started | High-risk consolidation; left last |
| C2 | **Independent decision-evidence field + "why A not B" score comparison** | data_layer §5 rem. | partial | Evidence currently rides inside `strategy_trace`; A-vs-B needs `build_candidate_pool` on the live path |
| C3 | **Measurement layer** — measurement contract, calibration/blank/control/replicate/batch records, per-KPI uncertainty/LOD/LOQ/censoring, raw-signal → processed-KPI traceable pipeline | enh L4 | partial | QC store exists; formal measurement contract does not |
| C4 | **Representation layer / unified experiment ontology** — typed material/formulation/device/protocol/environment/measurement schema, composition simplex + process graph + forbidden regions, multimodal evidence bundle, cross-campaign ontology | enh L5 | not started | Determines "tuner" vs "experiment knowledge system" |
| C5 | **Layered constraint & policy layer** — physical / operational / safety / epistemic / governance constraints, versionable + explainable + dynamically editable | enh L6 | partial | Safety gates exist; layered versionable constraint model does not |
| C6 | **Richer decision memory as next-round context** — strategy-change reasons, human-override reasons, rejected hypotheses, literature validated/refuted, post-hoc constraints, "human saw it but the sensor didn't" | enh L8 | partial | Trace/replay exist; not fed back as decision context |
| C7 | **Evaluation layer** — replay benchmark, ablation harness, regret / sample-efficiency / safety-violation / invalid-proposal-rate, proxy-to-functional transfer score, reproducibility score, decision-quality score | enh L9 | not started | Independent eval to avoid "looks smart" |

---

## D. Backlog — learning progression (guardrailed)

Path (from v2.md): rule selector → +decision trace → contextual bandit →
**offline meta-policy** → **trained meta-RL**. First three shipped; remaining:

| # | Item | Origin | Status | Notes |
|---|------|--------|--------|-------|
| D1 | **Offline meta-policy proof** — imitation / offline RL / policy evaluation / counterfactual replay showing learned policy ≥ heuristic | v2 P5 | partial | `policy_evaluation` / `learned_policy` / RL selectors exist; the *proof* and promotion do not |
| D2 | **Trained meta-RL policy network** — guardrailed meta-controller (propose → rule/safety validate → execute → trace), gated by offline-eval evidence | v2 P6 | not started | Only after D1 + stable reward + replay env + hard guardrails |

---

## E. Validation & benchmarking roadmap (not started)

From `validation.md`. The defensible claim to work toward:

> HELIOS improves closed-loop SDL decision quality under context-aware,
> safety-bounded execution, while preserving traceability and graceful
> degradation.

Evidence chain: `offline replay → shadow validation → canary live influence →
real campaign A/B → ablation → paper benchmark`.

- E1 Real multi-campaign data (multiple materials/tasks, objective levels)
- E2 Canary results (top-1 change, reward delta, safety warnings, auto-disable, failure attribution)
- E3 Shadow agreement / reward correlation (intent/mode/backend agreement, confidence calibration, predicted-vs-actual)
- E4 Failure-rate comparison (separating hardware/measurement vs backend/constraint vs scientific-negative)
- E5 Ablation (without objective hierarchy / failure taxonomy / backend memory / Nexus recommendation / context; rule vs safe-influence vs bandit vs learned)
- E6 Paper-level benchmark (fixed task set, metrics, baselines, statistical tests, reproducible config)
- E7 Cost/efficiency (rounds-to-threshold, experiments-to-improvement, failed-run cost, wall-clock, reagent/instrument cost, human interventions)
- E8 Safety/governance (unsafe-action-blocked rate, escalation quality, recovery success, audit completeness, decision reproducibility, degradation reliability)

The shadow-trace comparison analyzer (`shadow_trace_comparison`) already covers
part of E3 offline; the rest needs real campaign runs.

---

## F. Known real-run gaps in the shipped substrate

- **proxy_gap threading**: the per-round `ObjectiveState` built in the hook has
  `proxy_gap=None`, so the substrate never enters VALIDATION on real rounds
  (shows as a `class_mismatch` divergence vs the legacy track).
- **per-round safety signal**: the hook feeds static `policy_snapshot`; a
  genuine per-round safety/QC signal source is still open.
- **heat metadata**: `heat` lacks an `instrument` in `agent/skills/utility.md`
  and is kept `experiment` via a temporary pending-set so its
  `experiment_without_capability` calibration flag stays a true positive.
