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
- **Adaptive campaign decision layer** — orchestrator-agnostic campaign action
  selection across optimization strategy, validation, failure-aware recovery,
  context acquisition, human/LLM query, dynamic objective/constraint handling,
  and future scale/fidelity decisions. The shipped core includes the
  (`CampaignIntent` + `OptimizationMode`) taxonomy, phase posterior,
  evidence-based scoring, safety gates, Nexus optimization-intelligence
  evidence, backend recommendations, replay/validation accounting, and a
  default-off live authority gate (`CAMPAIGN_DECISION_AUTHORITY_ENABLED`) that
  can defer candidate generation for validation/recovery/context/objective/
  constraint actions while persisting the requested state update. See README ->
  Architecture.
- **Nexus/local candidate arbitration** — provider facade, Nexus backend
  adapters, multi-source candidate-pool builder, hard-gated decision policy,
  scored arbitration portfolio, provenance logging, and the
  `ENABLE_CANDIDATE_ARBITRATION` loop seam. Nexus remains advisory/backend
  input; HELIOS retains campaign decision authority.
- **Experimental-node active learning** — Nexus advisory route-evidence client,
  HELIOS-owned route scoring and capability/safety/budget/approval gates,
  default-off live authority, per-node parameter/protocol execution mapping,
  route-labelled observations, campaign-context checkpoints, replayable
  Scientific Decision Ledger metadata, and a live cross-repository contract
  test against Nexus `/api/experimental-routes/analyze`.
- **Context / memory / logging** — campaign context, objective stack + proxy
  gap, typed failure taxonomy (`failure_signatures`), backend performance memory
  + `ContextualStrategyBandit`, candidate-pool memory (recall), cross-campaign
  failure-zone memory, decision trace / evidence / outcome / reward / replay.
- **Scientific Decision Ledger** — live Pending -> Outcome -> Reward Decision
  Cards; deterministic/redacted Markdown projections for objective,
  observations, strategy, evidence, failure, recovery, and summary; policy and
  Nexus version snapshots; exact-text scientific-memory retrieval; typed RLVR
  JSONL export; and optional per-campaign local Git history that never pushes.
  See [scientific_decision_ledger.md](scientific_decision_ledger.md).
- **Loop / goal harness primitives (pure service layer)** —
  `loop_engineering` records observe-decide-act-evaluate iterations, reward, and
  replay summaries; `goal_harness` adds persistent goal state, normalized
  observations, tool descriptors, proposed tool actions, reflection notes,
  human blockers, and bad-path kill records. These layers are side-effect-free:
  they do not call PUDA, write DB state, execute tools, or promote policies.

By default, shipped campaign-decision features are read-only / fail-open /
shadow or approval-gated and do not change live candidate selection. The
explicit live authority flag promotes selected campaign decisions into bounded
pre-candidate routing, but still does not execute hardware or auto-apply
objective/space changes.

---

## B. Backlog — adaptive decision & scientific reasoning

| # | Item | Origin | Status | Notes |
|---|------|--------|--------|-------|
| B1 | **HypothesisState** — active/supported/contradicted hypotheses, discriminating experiments | v3 §3 | not started | VoI `expected_hypothesis_resolution` stays 0 until this exists |
| B2 | **Instrument / runtime belief state + PUDA telemetry** — calibration confidence, drift, telemetry anomalies | v3 §6 | not started | Would let a bad reading be attributed to the instrument, not the sample |
| B3 | **OperationalAbstractionLearner** (Phase 6) — promote repeated successful action sequences to reusable ops (proposal-only) | v3 §8 | not started | Explicitly deferred until several real shadow logs are reviewed |
| B4 | **Campaign-level memory beyond candidate/failure** — objective patterns, strategy-success-by-phase, hypothesis-resolution patterns, useful context queries, per-instrument reliability | v3 §9 | not started | Higher tier than failure-zone memory |
| B5 | **StrategyClass scientific-action dimension** on the selector (PARAMETER_OPTIMIZATION / HYPOTHESIS_DISCRIMINATION / CALIBRATION / …) | v3 §10 | partial | `OptimizationMode`/`CampaignIntent` exist but not this explicit class |
| B6 | **Objective staging / fidelity escalation + ObjectiveManager** — proxy → mechanism → functional → deployment ladder, staged scoring, objective versioning wired into selection | data_layer §4; enh L7 | partial | `ObjectiveStack`/`ObjectiveState`/proxy_gap exist; authority gate can persist objective-transition requests, but staged ObjectiveManager execution is not wired |
| B7 | **Parameter-space / synthesis-route revision as first-class** — `SpaceRevision`, `ParameterSpacePolicy`, route switching | enh L14; v3-adjacent | partial | `revise_space` intent + `space_revision` records exist and authority gate can persist constraint/space requests; no route-switch/space-policy executor yet |

---

## C. Backlog — data / representation / evaluation infrastructure

| # | Item | Origin | Status | Notes |
|---|------|--------|--------|-------|
| C1 | **OptimizationDataContract** — unify the 8 scattered contract types (ResultPacket / Candidate / ObjectiveSpec / OutcomeConstraint / Observation / FailureRegionModel / DecisionResult / ProvenanceLogger) into one spec | data_layer §1 | not started | High-risk consolidation; left last |
| C2 | **Independent decision-evidence field + "why A not B" score comparison** | data_layer §5 rem. | partial | Candidate-pool arbitration now records a scored portfolio and provenance; remaining work is a first-class evidence field/report surface outside the arbitration record |
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
| D2 | **Trained meta-RL policy network** — guardrailed campaign decision policy (propose → rule/safety validate → execute → trace), gated by offline-eval evidence | v2 P6 | not started | Only after D1 + stable reward + replay env + hard guardrails |

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

---

## G. Backlog — autonomous optimization decision agent

Target shape:

`Goal -> Observe -> Decide -> Act -> Evaluate -> Reflect -> Continue/Stop/Human gate`

The first pure layers are in place, but the live autonomous agent remains
guardrailed and incomplete by design.

| # | Item | Status | Notes |
|---|------|--------|-------|
| G1 | **Persistent Goal-Oriented Harness** — durable mission/goal owner with status, budget, blockers, next-action proposals, and stop conditions | partial | `app/services/goal_harness.py` provides pure state transitions; persistence, scheduler ownership, and API surfaces are not wired |
| G2 | **Unified perception layer** — normalize PUDA responses, API feedback, logs, artifacts, images, QC signals, and human notes into one observation stream | partial | `ObservationEnvelope` exists; no live adapters yet for PUDA telemetry, vision, SSE logs, or analyzer outputs |
| G3 | **Agent-facing tool registry** — typed tools with capability, schema, risk, timeout, permissions, rollback, and output observation contract | partial | `ToolDescriptor` exists in the harness; it is not connected to primitives registry, PUDA backend, MCP tools, or hardware adapters |
| G4 | **Action executor bridge** — route proposed `ToolAction` through approval, safety, PUDA/run creation, execution, and returned observations | not started | Must stay separate from the pure harness; live hardware requires human/governance gates |
| G5 | **Campaign-level multi-step correction** — kill bad optimization paths, revise strategy, request calibration, switch backend, or narrow/expand parameter space | partial | Authority gate can defer a round and persist validation/recovery/context/objective/constraint requests; no campaign-loop consumer yet excludes killed paths from future candidate/proposal generation |
| G6 | **Agent notebook / reflection memory** — structured notes for failed paths, disproven hypotheses, unreliable tools, human overrides, and future constraints | partial | `ReflectionNote` exists; notes are not persisted into semantic/procedural memory or forced into next-round context |
| G7 | **Live self-optimization promotion** — learned policy moves from replay/shadow/canary into bounded live influence after evidence thresholds | not started | Existing learned-policy path remains conservative; no automatic live promotion without explicit approval workflow |
| G8 | **Long-running daemon / scheduler integration** — wake on external events, resume after restart, wait for PUDA or human, and continue the goal | not started | Existing durable run/campaign pieces exist, but no single autonomous goal daemon owns the full lifecycle |
