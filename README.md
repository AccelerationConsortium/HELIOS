<p align="center">
  <img src="docs/logo/helios_dark.svg" alt="HELIOS" width="620"/>
</p>

# HELIOS — Hierarchical Experimental Learning and Intelligent Optimization System

Helios (Hierarchical Experimental Learning and Intelligent Optimization System) is an orchestrator-agnostic adaptive campaign decision layer for closed-loop experimentation.

Helios decides which campaign-level action should happen next, including optimization strategy selection, validation, failure-aware recovery, context acquisition, human/LLM query, dynamic objective/constraint handling, and future scale/fidelity-aware decisions.

HELIOS treats optimizers, simulators, automation services, and external systems as downstream tools, backends, or evidence sources. This README focuses on the campaign decision layer: the typed inputs it consumes, the actions it can recommend, the evidence it records, and the replay/validation path that keeps those decisions auditable.

---

## What HELIOS Decides

For each campaign round, HELIOS turns scientific context into a bounded campaign-level decision:

- **Optimize** — choose the optimization strategy, mode, backend, and candidate-generation path.
- **Validate** — decide when proxy progress needs mechanism, repeatability, or higher-fidelity validation.
- **Recover** — route failure-aware recovery when results, constraints, measurements, or backends look unreliable.
- **Acquire context** — ask for literature, prior-campaign evidence, diagnostics, or missing experimental context.
- **Ask a human or LLM** — request human observation or LLM-supported context only at the language/knowledge boundary.
- **Revise objectives or constraints** — handle dynamic objective hierarchy, proxy gaps, constraints, and safety envelopes.
- **Escalate scale or fidelity** — provide the decision surface for future scale/fidelity-aware campaign moves.

The output is an auditable campaign decision envelope with evidence, rationale, selected policy/action, optional candidate portfolio, expected value, and replayable outcome accounting.

---

## Decision Flow

```
TaskContract / campaign context
        |
        v
RoundContext + objective state + failure history + backend memory
        |
        v
Adaptive campaign policy
        |
        +--> CampaignIntent / OptimizationMode / backend recommendation
        +--> validation, recovery, context, human/LLM, objective/constraint action
        |
        v
Decision trace + evidence + reward/outcome + replay record
```

The live path is conservative by design: rule-based, auditable, and bounded by explicit safety gates. Learning-based policies do not replace it by default. They move through replay evaluation, shadow records, canary runs, promotion gates, and approval workflows before they can influence live decisions.

---

## Core Capabilities

- **Orchestrator-agnostic campaign decision layer** — keeps campaign-level decision authority separate from any downstream backend.
- **Context-aware policy** — uses objective hierarchy, proxy-gap state, failure attribution, backend memory, Nexus diagnostics, BO MCP availability, candidate/failure-zone memory, and bandit/learned-policy signals.
- **Dynamic action vocabulary** — represents optimization, validation, calibration, failure diagnosis, context seeking, human observation, safety-constraint tightening, stopping, and future scale/fidelity choices.
- **Candidate and backend arbitration** — combines local baselines, Nexus/BO MCP signals, candidate pools, safety gates, and provenance into a traceable portfolio.
- **Failure-aware recovery** — separates scientific negative evidence from measurement, backend, constraint, and downstream tool failures.
- **Trace, reward, and replay** — records `StrategyTrace`, `StrategyEvidence`, `StrategyOutcome`, `StrategyReward`, typed `FailureEvent`, and replay summaries.
- **Scientific Decision Ledger** — projects every live campaign decision into deterministic, redacted Markdown Decision Cards with evidence, alternatives, outcome, reward, failures, recovery, policy/Nexus versions, exact-text memory search, and typed RLVR export.
- **LLM boundary discipline** — LLMs can help translate intent, gather context, or generate review notes; they do not steer the live optimization loop.

---

## Scientific Decision Ledger

HELIOS keeps two deliberately separate truths:

- Typed DTOs and SQLite rows are the transactional runtime truth.
- Markdown and optional campaign-local Git are the readable, reviewable, auditable scientific truth.

A decision is written before execution with `Outcome: Pending`, then finalized in place after analysis with its observed outcome, deterministic verifier scores, reward, failures, and recovery episode. A campaign is projected under `data/scientific_ledger/campaigns/<campaign-id>/`:

```text
campaign.md                  # objective, metadata, decision index
index.md                     # navigable artifact index
summary.md                   # aggregate decisions and rewards
trajectory.md                # Mermaid decision trajectory
policy.md                    # current decision policy snapshot
policy_versions/<version>.md # immutable first snapshot per policy version
nexus.md                     # Nexus contract/version and diagnostics
training_dataset.md          # human-reviewable RLVR projection
rounds/001/
  objective.md
  observations.md
  decision_001.md
  strategy.md
  evidence.md
  failure.md
  recovery.md
  summary.md
```

Decision Cards contain the question, scientific context, evidence, ranked candidate actions, selected action/backend, rationale, confidence/expected gain, outcome, reward/verifiers, failure/recovery counts, and reproducibility provenance. Values are deterministically rendered and recursively redacted before they reach Markdown.

The read-only API exposes:

- `GET /api/v1/memory/scientific/search?q=pipette%20offset`
- `GET /api/v1/memory/scientific/{campaign_id}/artifact?path=rounds/001/decision_001.md`
- `GET /api/v1/memory/scientific/{campaign_id}/rlvr`

RLVR JSONL is generated from the typed `decision_trajectories` store, not by scraping Markdown. Optional Git history is one repository per campaign, stages exact Markdown paths only, and never pushes or modifies the HELIOS source repository. See [Scientific Decision Ledger](docs/scientific_decision_ledger.md) for lifecycle, schemas, safety properties, and operations.

---

## Architecture

| Surface | Responsibility | Representative modules |
|---------|----------------|------------------------|
| **Contract and context** | Typed campaign goal, objectives, constraints, budget, safety, and round context | `app/contracts/`, `app/services/round_context.py`, `app/services/objective_state.py` |
| **Campaign policy** | Decide next campaign-level action and strategy mode | `app/services/strategy_selector.py`, `app/services/strategy_actions.py`, `app/services/decision_layer.py` |
| **Evidence and memory** | Attach diagnostics, prior-campaign evidence, failure history, and backend memory | `app/services/decision_trace.py`, `app/services/backend_memory.py`, `app/optimization/candidate_memory.py`, `app/optimization/failure_zone_memory.py` |
| **Candidate/backend arbitration** | Build, gate, score, and explain candidate/backend choices | `app/optimization/service.py`, `app/optimization/pool_service.py`, `app/optimization/decision_policy.py`, `app/optimization/provenance.py` |
| **Adaptive substrate** | Shadow-only scientific activity mode, dynamic action space, and value-of-information assessment | `app/services/adaptive_campaign_substrate.py`, `app/services/campaign_mode.py`, `app/services/dynamic_action_space.py`, `app/services/value_of_information.py` |
| **Outcome and replay** | Evaluate decision quality, reward components, and replay summaries | `app/services/decision_outcome.py`, `app/services/verifiable_reward.py`, `app/services/decision_replay.py`, `app/services/policy_evaluation.py` |

### Adaptive Campaign Decision Layer

HELIOS uses scientific context, objective hierarchy, typed failure attribution, backend performance memory, candidate/failure-zone memory, Nexus diagnostics, BO MCP availability, and bandit/learned-policy signals to decide which campaign-level action should happen next. Today that includes `CampaignIntent`, `OptimizationMode`, and candidate-generation backend selection; the same layer owns validation, failure-aware recovery, context acquisition, human/LLM query, dynamic objective/constraint handling, and future scale/fidelity-aware decisions.

The default runtime still records contextual campaign decisions in shadow mode.
When `CAMPAIGN_DECISION_AUTHORITY_ENABLED=true`, the orchestrator promotes the
decision envelope into a bounded pre-candidate gate: `STOP_CAMPAIGN` terminates
before more candidates, while validation, recovery, context, objective, and
constraint actions defer the current round, persist the requested campaign
state update, and leave candidate generation untouched for later rounds. The
gate never executes hardware or auto-applies objective/space changes.

### Experimental-node active learning

For campaigns whose alternatives are materially different experimental nodes
(for example, different synthesis routes), `experimental_route_graph` declares
the nodes, transitions, execution mapping, capability requirements, cost, and
safety metadata. Nexus `/api/experimental-routes/analyze` supplies versioned,
`advisory_only` evidence. HELIOS then scores every reachable option and enforces
local capability, safety, budget, operator-approval, and executable-protocol
gates. `NEXUS_EXPERIMENTAL_ROUTES_ENABLED` enables characterization in shadow;
the separate `EXPERIMENTAL_ROUTE_AUTHORITY_ENABLED` gate is required to change
the live node. Each decision and route-labelled outcome is checkpointed in the
campaign context and included in the Scientific Decision Ledger trajectory.

### Optimization Code Map

The optimization stack is split by authority boundary:

- `app/services/optimization_intelligence.py` enriches strategy selection with optional Nexus diagnostics, similar-campaign evidence, and backend recommendations. It emits structured evidence; it does not choose a live candidate.
- `app/optimization/nexus_provider.py` and `app/optimization/nexus_backend.py` adapt Nexus profiling and `nexus_*` algorithm plugins behind HELIOS provider/backend interfaces. Nexus remains an advisor/backend, not campaign authority.
- `app/optimization/service.py`, `app/optimization/pool_service.py`, and `app/optimization/candidate_pool.py` build the multi-source candidate portfolio.
- `app/optimization/decision_policy.py` is the hard gate and arbitration authority for concrete candidates. It enforces bounds, deduplication, safety hook results, and ranks survivors with the strategy decision's utility model.
- `app/optimization/loop_integration.py` is the campaign-loop seam. Deep candidate-pool arbitration is controlled by `ENABLE_CANDIDATE_ARBITRATION` and defaults off.
- `app/optimization/provenance.py` records selected portfolios, rejected candidates, scored pools, and strategy decisions so "why this candidate, not that one?" can be audited.

### Adaptive Campaign Substrate (shadow)

The shadow-only adaptive substrate proposes a scientific-activity `CampaignMode`, assesses the action space, and scores candidate value-of-information as an advisory artifact. It changes no routing by default and is gated by `ADAPTIVE_SUBSTRATE_SHADOW_ENABLED`.

See [docs/adaptive_campaign_substrate.md](docs/adaptive_campaign_substrate.md).

### Loop Engineering Layer

The loop-engineering layer records each observe-decide-act-evaluate unit as a replayable episode: loop spec, signals, decision, outcome, reward, and replay summary. This makes workflow data usable for offline evaluation, shadow/canary promotion, failure attribution, and future policy improvement without changing the live path.

The first pure service layer is `app/services/loop_engineering.py`. It is dependency-light and side-effect-free by design: it does not call downstream services, mutate campaign state, write to the database, or promote learned policies.

---

## Developer Setup

```bash
git clone https://github.com/SissiFeng/HELIOS.git
cd HELIOS
pip install -e ".[dev]"
```

Run the focused validation for the current positioning/reporting boundary:

```bash
pytest tests/test_system_validation_report.py
```

Run the broader decision-layer tests as needed:

```bash
pytest \
  tests/test_decision_layer.py \
  tests/test_decision_trace.py \
  tests/test_decision_outcome.py \
  tests/test_decision_replay.py \
  tests/test_verifiable_reward.py \
  tests/test_policy_evaluation.py \
  tests/test_adaptive_campaign_substrate.py \
  tests/test_shadow_trace_comparison.py \
  tests/test_backend_memory.py \
  tests/test_candidate_pool.py
```

---

## Decision-Layer Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | LLM provider for language/knowledge-boundary tasks only |
| `LLM_MODEL` | provider default | Model ID passed to the configured provider |
| `CONTEXTUAL_DECISION_SHADOW_ENABLED` | `false` | Record the legacy contextual decision shadow trace per round |
| `CAMPAIGN_DECISION_AUTHORITY_ENABLED` | `false` | Promote contextual campaign decisions into a bounded live pre-candidate gate |
| `ADAPTIVE_SUBSTRATE_SHADOW_ENABLED` | `false` | Record the adaptive campaign substrate shadow snapshot per round |
| `ENABLE_CANDIDATE_ARBITRATION` | `false` | Enable deep candidate-pool arbitration instead of legacy generation fallback |
| `NEXUS_EXPERIMENTAL_ROUTES_ENABLED` | `false` | Request advisory experimental-route characterization from Nexus each round |
| `EXPERIMENTAL_ROUTE_AUTHORITY_ENABLED` | `false` | Allow HELIOS to apply a route selected by its local safety/budget/approval policy |
| `NEXUS_URL` | `http://localhost:8000/api` | Base URL for optional Nexus REST advisory endpoints |
| `NEXUS_API_KEY` | empty | Optional `X-API-Key` sent to Nexus REST endpoints |
| `NEXUS_TIMEOUT_SECONDS` | `10` | Nexus REST request timeout |
| `SCIENTIFIC_LEDGER_ENABLED` | `true` | Persist live Decision Cards and typed outcome/reward accounting; fail-open with respect to campaign routing |
| `SCIENTIFIC_LEDGER_ROOT` | `data/scientific_ledger` | Root for campaign Markdown artifacts |
| `SCIENTIFIC_LEDGER_GIT_ENABLED` | `false` | Commit changed Markdown artifacts to each campaign's local Git repository |
| `SCIENTIFIC_LEDGER_GIT_AUTO_INIT` | `true` | Initialize a missing campaign-local repository when Git recording is enabled |
| `SCIENTIFIC_LEDGER_GIT_AUTHOR_NAME` | `HELIOS Scientific Ledger` | Local ledger commit author name |
| `SCIENTIFIC_LEDGER_GIT_AUTHOR_EMAIL` | `helios-ledger@localhost` | Local ledger commit author email |

---

## Validation Evidence

HELIOS is framed as an orchestrator-agnostic adaptive campaign decision layer. The product boundary is campaign-level decision authority rather than ownership of downstream automation or presentation surfaces. The live campaign policy remains rule-based and auditable by default; Nexus and BO MCP are optimization advisor/backend/tool paths, not campaign decision authorities. Learned policy and self-evolution paths are offline, shadow, canary, and approval-gated; their metadata does not change default BO MCP/Nexus/backend behavior.

The architecture validation report is version-controlled at [docs/HELIOS_ARCHITECTURE_VALIDATION.md](docs/HELIOS_ARCHITECTURE_VALIDATION.md). It is a static evidence pack for the current validation boundary.

Run the validation suite:

```bash
bash scripts/run_validation_suite.sh
```

Equivalent targeted test command:

```bash
pytest \
  tests/test_candidate_memory.py \
  tests/test_failure_zone_memory.py \
  tests/test_offline_closed_loop_sdl.py \
  tests/test_offline_scenario_benchmarks.py \
  tests/test_policy_evolution.py \
  tests/test_policy_evolution_workflow_e2e.py \
  tests/test_learned_policy.py \
  tests/test_system_validation_report.py \
  tests/test_backend_selection.py
```

---

## Repository Map

```
HELIOS/
├── app/
│   ├── contracts/           # Typed campaign contracts and query/task models
│   ├── optimization/        # Candidate pools, backend facades, arbitration, provenance
│   ├── services/            # Campaign policy, evidence, reward, replay, objective/failure logic
│   ├── api/v1/endpoints/    # Service API surfaces
│   └── core/                # Config, DB, startup lifecycle
├── docs/
│   ├── HELIOS_ARCHITECTURE_VALIDATION.md
│   ├── adaptive_campaign_substrate.md
│   ├── scientific_decision_ledger.md
│   └── development_progress.md
├── tests/                   # Pytest coverage for policy, replay, validation, and evidence layers
├── benchmarks/              # Offline method and policy evaluation harnesses
├── models/                  # Learned-policy checkpoints and replay artifacts
├── pyproject.toml           # Dependencies and tool config
└── README.md
```

---

## External Decision Inputs

| Integration | Role in HELIOS |
|-------------|----------------|
| **Nexus** | Optimization diagnostics, profiling, and backend/candidate evidence |
| **BO MCP / Ax / local BO** | Optimization backend signals and candidate proposals |
| **Anthropic / OpenAI** | Language/knowledge-boundary tasks such as intent parsing, context requests, and review notes |
| **Campaign memory** | Similar-campaign priors, backend history, failure zones, and replay evidence |

---

## Contributing

1. Keep decision authority explicit: backends advise, HELIOS decides.
2. Preserve typed traces, evidence, outcomes, rewards, and replay records for every new decision path.
3. Keep learned policies gated by replay, shadow/canary evidence, and explicit promotion controls.
4. Add tests beside changes to policy, arbitration, reward, replay, or validation logic.

---

## License

MIT — see [LICENSE](LICENSE) for details.
