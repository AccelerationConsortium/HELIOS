<p align="center">
  <img src="docs/logo/helios_dark.svg" alt="HELIOS" width="620"/>
</p>

<p align="center">
  <sub>
    <code>H</code> electrochem cell &nbsp;·&nbsp;
    <code>E</code> spectrometer &nbsp;·&nbsp;
    <code>L</code> pipette &nbsp;·&nbsp;
    <code>I</code> UR5 arm &nbsp;·&nbsp;
    <code>O</code> OT-2 deck &nbsp;·&nbsp;
    <code>S</code> pump
  </sub>
</p>

# HELIOS — Holistic Experiment Learning Intelligent Orchestration System

HELIOS is an **agent-native orchestrator** for self-driving laboratories (SDLs). It composes 27 specialist agents into 4 cooperating swarms behind a single natural-language interface — so scientists describe experiments in plain language, and HELIOS plans, validates, executes, and iterates autonomously, closing the loop between hypothesis and hardware.

Most "AI lab assistants" are LLM wrappers: a human stays in the driver's seat and the model translates their words into button clicks. HELIOS inverts that — **agents are the operators**. Contracts, leases, skills, the event bus, and recovery are all designed for agents as first-class citizens. The cleanest test: mock out the LLM (`LLM_PROVIDER=mock`) and the entire campaign loop still closes, because the intelligence lives in the orchestration and optimization layers, not in prompt glue.

---

## How a Campaign Flows Through the Agents

What actually happens between a scientist typing one sentence and HELIOS handing back an optimized recipe:

**1. One sentence in → a typed contract out.**
The user writes, e.g., *"Screen OER catalysts to minimize overpotential at 10 mA/cm²; Fe/Co/Ni ratios are tunable; budget 30 rounds."* The **conversation engine** and **RequirementParserAgent** run a multi-turn clarification dialogue — missing KPI? unclear bounds? which steps need a human signature? — and emit a **`TaskContract`**: a versioned, schema-validated object holding the objective, the exploration space, stop conditions, a safety envelope, and the human-gate policy. From this point on, every agent works off the contract, not the chat history. Contracts carry `schema_version` and migrate forward automatically, so a campaign archived months ago still loads after upgrades.

**2. Devices are exposed as skills.**
Every instrument registers a skill (`agent/skills/*.md`): its primitives, their typed parameters, a safety class (e.g. `HAZARDOUS`), and precondition/effect contracts ("channel must be idle"). Agents can only invoke declared primitives with type-checked arguments — **the machine's capabilities are fixed, so the LLM cannot hallucinate an operation that doesn't exist**. New instruments are onboarded by the **OnboardingAgent**, which discovers primitives, generates integration code and the skill definition, and writes them to disk after human review — hours, not weeks.

**3. The orchestrator runs the round loop.**
The **OrchestratorAgent** takes over and drives one pipeline per round, each stage a dedicated agent:

```
PlannerAgent          expands the contract into a round plan
DesignAgent           proposes candidates — an inner RL loop picks the search
                      strategy per round (LHS early, GP+EI/MES mid, refinement late)
SafetyAgent           checks every candidate against the contract's safety envelope
CompilerAgent         compiles recipes into an executable hardware DAG
Execution layer       dispatches to OT-2, PLC pumps, potentiostat (or simulation)
SensingAgent /        QC + analysis of returning data
AnalyzerAgent
StopAgent             converged? budget exhausted? continue or stop
```

All agents share one shape: a common `BaseAgent` base class, typed Pydantic I/O, and a mandatory **decision tree** record per decision (options considered, choice, rationale). Cross-agent calls go through the **ControlPlane** — lease first, call second, audit always — which is why 27 agents don't collapse into chaos.

**4. Hard problems convene a swarm.**
When the pipeline hits something that needs judgment — an anomalous data pattern, a hypothesis worth stress-testing — the **SwarmFactory** spawns an ephemeral consult: **Scientist** (hypotheses), **Engineer** (feasibility/cost), **Analyst** (data interpretation), **Validator** (tries to break the other three). They coordinate over a shared **blackboard** and disband when done. Racing swarms and adversarial hypothesis loops let cheap compute fight it out before expensive robot time is spent.

**5. Two things run through everything: safety and the event bus.**
Safety appears at three levels — the contract's safety envelope, per-primitive safety classes and preconditions, and the per-round SafetyAgent; human approval is a first-class **pause state**, resumable from checkpoint. Meanwhile every agent action streams onto the **event bus**, persists to the DB, and broadcasts live over **SSE** with exactly-once delivery — watch agents think in the browser, replay the full decision chain afterwards.

**6. Failure is a planned-for state.**
Contracts isolate failures to their layer; the **RecoveryAgent** fixes forward (re-plans the round, not the campaign). If the whole process dies, campaigns resume from their last SQLite checkpoint via `/resume`.

**7. The data stays alive.**
Results, uncertainties, and decision chains land in campaign memory — queryable in natural language ("which recipe won last week? plot its CV"). **RGPE transfer learning** warm-starts the next related campaign with interpretable ranking weights: the more HELIOS runs, the smarter the next campaign starts.

---

## Features

- **Natural language intake** — a multi-turn clarification dialogue turns a free-text experiment description into a versioned, schema-validated `TaskContract`
- **Agent-native orchestration** — 27 specialist agents grouped into 4 swarms (Scientist / Engineer / Analyst / Validator), composed as a stage graph that branches, retries, and remembers; all cross-agent calls go through a ControlPlane with agent leases and a full audit trail
- **Devices as skills** — instruments expose typed primitives with safety classes and precondition/effect contracts; agents cannot invoke operations that don't exist
- **Real-time reasoning stream** — every agent step emits SSE events with exactly-once delivery and DB-backed replay; the browser shows a live decision tree of what each agent considered and why
- **Hardware agnostic** — runs in `simulated` mode for development; switches to live Opentrons OT-2, PLC relays, and electrochemistry sensors by changing one env var
- **Context-aware dynamic strategy** — a scientific campaign meta-controller selects campaign intent, optimization mode, and candidate-generation backend from scientific context, objective hierarchy, failure attribution, backend memory, Nexus diagnostics, BO MCP availability, and shadow learning signals
- **Safety-first** — contract safety envelopes, per-primitive safety classes, preflight checks before every round, and human-in-the-loop gates as resumable pause states
- **Durable execution** — SQLite-backed campaign checkpoints survive restarts; crashed campaigns resume from the last completed round

---

## Architecture

### Four Layers

| Layer | Role | Key Components |
|-------|------|----------------|
| **L3 Orchestration** | Task contracts, admission control, agent lease pool | `OrchestratorAgent`, `ControlPlane`, `RequirementParserAgent` |
| **L2 Planning** | Experimental design & adaptive strategy | `PlannerAgent`, `DesignAgent`, `SafetyAgent`, inner RL strategy router |
| **L1 Execution** | Protocol compilation & hardware abstraction | `CompilerAgent`, `CodeWriterAgent`, `DeckLayoutAgent`, hardware dispatcher |
| **L0 Evidence** | Campaign memory, uncertainty, causal updates | `AnalyzerAgent`, `SensingAgent`, `MonitorAgent`, `RecoveryAgent` |

### Dynamic Strategy Meta-Controller

HELIOS dynamic strategy selection is a context-aware scientific campaign
meta-controller. It uses scientific context, objective hierarchy, typed failure
attribution, backend performance memory, candidate/failure-zone memory, Nexus
diagnostics, BO MCP availability, and bandit/learned-policy signals to choose
three things each round: `CampaignIntent`, `OptimizationMode`, and the
candidate-generation backend.

The default live path is conservative: rule-based, auditable, and bounded by
explicit safety gates. Learning-based policies do not replace this path by
default. They are introduced progressively through replay evaluation, shadow
records, canary runs, promotion gates, and explicit approval workflows. Each
round is backed by `StrategyTrace`, `StrategyEvidence`, `StrategyOutcome`,
`StrategyReward`, typed `FailureEvent` attribution, and replay/validation
records so strategy changes remain inspectable after the fact.

### Optimization Code Map

The merged optimization stack is split by authority boundary:

- `app/services/optimization_intelligence.py` enriches strategy selection with
  optional Nexus diagnostics, similar-campaign evidence, and backend
  recommendations. It emits structured evidence; it does not choose a live
  candidate.
- `app/optimization/nexus_provider.py` and `app/optimization/nexus_backend.py`
  adapt Nexus profiling and `nexus_*` algorithm plugins behind HELIOS provider
  and backend interfaces. Nexus remains an advisor/backend, not campaign
  authority.
- `app/optimization/service.py`, `app/optimization/pool_service.py`, and
  `app/optimization/candidate_pool.py` build the multi-source candidate
  portfolio: Nexus top-k, local baseline, archetype-scored candidates, BO MCP
  hints, replicates, and recovery probes where available.
- `app/optimization/decision_policy.py` is the hard gate and arbitration
  authority for concrete candidates. It enforces bounds, deduplication, safety
  hook results, and then ranks survivors with the strategy decision's utility
  model.
- `app/optimization/loop_integration.py` is the campaign-loop seam. Deep
  candidate-pool arbitration is controlled by `ENABLE_CANDIDATE_ARBITRATION`
  and defaults off; when disabled or failed, the loop keeps the legacy
  generation path.
- `app/optimization/provenance.py` records the selected portfolio, rejected
  candidates, scored pool, and strategy decision so "why this candidate, not
  that one?" can be audited after the round.

### Adaptive Campaign Substrate (shadow)

On top of the meta-controller sits a **shadow-only** adaptive decision
substrate that, per round, proposes a scientific-activity `CampaignMode`
(optimization, validation, calibration, failure diagnosis, context seeking,
human observation, safety-constraint tightening, stop), assesses the action
space, and scores candidate value-of-information — as an **advisory** artifact
recorded next to the campaign, never as a control signal. It is composed of
`objective_state`, `failure_attribution`, `campaign_mode`,
`dynamic_action_space`, `value_of_information`, and
`adaptive_campaign_substrate`, and is reconciled against the legacy contextual
decision track by `shadow_trace_comparison`. It changes no routing and is gated
by `ADAPTIVE_SUBSTRATE_SHADOW_ENABLED` (default off). See
[docs/adaptive_campaign_substrate.md](docs/adaptive_campaign_substrate.md).

### Agent Model

HELIOS is an **autonomous multi-agent experimentation system**, not an "LLM
agent". Agents are typed input→output services orchestrated in a deterministic
L3→L2→L1→L0 pipeline; the per-round optimization/decision loop is **LLM-free by
design** (classical BO/GP + rule-based scoring + optional non-LLM learned
policies). The LLM is used only at the language/knowledge boundary (NL→plan,
NL→code, priors, post-run review) and never steers a live round. See
[docs/agent_architecture.md](docs/agent_architecture.md).

### Agent Roster

| Agent | Purpose |
|-------|---------|
| `Orchestrator` | Root coordinator; drives the campaign loop |
| `PlannerAgent` | Generates experimental designs (DoE, LHS, prior-guided) |
| `SafetyAgent` | Preflight safety checks; blocks non-compliant rounds |
| `SimulationAgent` | Physics simulation before hardware execution |
| `AnalyzerAgent` | Post-round analytics; convergence detection; KPI tracking |
| `CompilerAgent` | High-level plan → OT-2 protocol code |
| `CodeWriterAgent` | AST-to-Python code generation |
| `NLPCodeAgent` | Natural language → protocol code |
| `MonitorAgent` | Real-time sensor monitoring; anomaly detection |
| `SensingAgent` | QC data collection and validation |
| `RecoveryAgent` | Execution error recovery (fix-forward or abort) |
| `CleaningAgent` | Equipment cleaning protocol generation |
| `OnboardingAgent` | New device initialization and configuration |
| `QueryAgent` | Historical data retrieval from structured DSL |
| `InverseDesignAgent` | Goal-driven parameter synthesis (Nexus integration) |
| `StrategySelector` | Chooses optimization algorithm per campaign phase |
| `SwarmAgent` | Multi-agent sub-task coordination |

### Four Specialist Swarms

| Swarm | Members | Focus |
|-------|---------|-------|
| **ScientistSwarm** | Planner + Design | Hypothesis generation, experimental design |
| **EngineerSwarm** | Compiler + CodeWriter | Protocol compilation, code generation |
| **AnalystSwarm** | Analyzer + Monitor | Data analysis, metrics computation |
| **ValidatorSwarm** | Safety + Sensing | Safety checks, QC validation |

---

## Quick Start

### Simulated Mode (no hardware required)

```bash
# 1. Clone
git clone https://github.com/SissiFeng/HELIOS.git
cd HELIOS

# 2. Configure
cp .env.example .env
# defaults are fine for simulation

# 3. Run
docker compose up

# UI:       http://localhost:8000/lab
# API docs: http://localhost:8000/docs
```

### Live Hardware Mode

```bash
# Edit .env
ADAPTER_MODE=live
ROBOT_IP=<your-ot2-ip>        # Opentrons OT-2 HTTP API
RELAY_PORT=/dev/ttyUSB0        # or 'auto' for auto-detect
SQUIDSTAT_PORT=auto

# Start both services (main + hardware recovery bridge)
docker compose --profile hardware up
```

### Manual Python Setup

```bash
# Base (simulated only)
pip install -e .

# With hardware drivers
pip install -e ".[hardware]"

# With ML strategies (DQN/PPO)
pip install -e ".[ml]"

# Full install
pip install -e ".[all]"

# Run
ADAPTER_MODE=simulated LLM_PROVIDER=mock \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Configuration

All configuration is via environment variables (`.env` file or shell).

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTER_MODE` | `simulated` | `simulated` — no hardware; `live` — real devices |
| `ADAPTER_DRY_RUN` | `true` | When `true`, hardware commands are logged but not sent |
| `LLM_PROVIDER` | `mock` | `mock` (testing), `anthropic`, or `openai` |
| `LLM_API_KEY` | — | API key for chosen LLM provider |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Model ID passed to provider |
| `ROBOT_IP` | — | OT-2 / OT-2 Flex HTTP API address |
| `RELAY_PORT` | `auto` | Serial port for relay controller |
| `SQUIDSTAT_PORT` | `auto` | Serial port for Squidstat potentiostat |
| `HELIOS_PORT` | `8000` | Main service port |
| `RECOVERY_PORT` | `8001` | Hardware recovery bridge port |
| `DB_PATH` | `/app/data/orchestrator.db` | SQLite database path |
| `CONTEXTUAL_DECISION_SHADOW_ENABLED` | `false` | Record the legacy contextual decision shadow trace per round (observational; no routing effect) |
| `ADAPTIVE_SUBSTRATE_SHADOW_ENABLED` | `false` | Record the adaptive campaign substrate shadow snapshot per round (observational; no routing effect) |

---

## API Overview

Base URL: `http://localhost:8000`

### Campaign Lifecycle

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/orchestrate/start` | Start a campaign from a `TaskContract` |
| `POST` | `/api/v1/orchestrate/from-session/{session_id}` | Start a campaign from an init conversation session |
| `GET` | `/api/v1/orchestrate/{campaign_id}/status` | Query campaign state and progress |
| `POST` | `/api/v1/orchestrate/{campaign_id}/stop` | Cancel a running campaign |
| `POST` | `/api/v1/orchestrate/{campaign_id}/resume` | Resume a paused/crashed campaign from its checkpoint |
| `GET` | `/api/v1/orchestrate/{campaign_id}/events/stream` | SSE event stream (supports `Last-Event-ID` replay) |

### Natural Language Interface

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/nl/parse` | Parse free-text description → `TaskContract` |

### Initialization & Onboarding

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/init/start` | Start interactive setup session |
| `POST` | `/api/v1/init/{session_id}/respond` | Respond to initialization prompts |
| `POST` | `/api/v1/onboarding/discover` | Auto-discover primitives for a new instrument |
| `POST` | `/api/v1/onboarding/generate` | Generate integration code for a new instrument |
| `POST` | `/api/v1/onboarding/confirm` | Approve safety/config confirmations |
| `POST` | `/api/v1/onboarding/write` | Write approved integration files to disk |

### Data & Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/campaigns` | List all campaigns |
| `GET` | `/api/v1/runs` | List experiment runs |
| `GET` | `/api/v1/metrics` | Campaign KPI metrics |
| `POST` | `/api/v1/query` | Query historical data with DSL |
| `GET` | `/api/v1/capabilities` | Available primitives and templates |

### Human-in-the-Loop

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/confirmations/{request_id}/respond` | Approve or reject a pending action |
| `POST` | `/api/v1/evolution/proposals/{proposal_id}/approve` | Approve a candidate proposal |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe (always 200 OK) |
| `GET` | `/api/v1/health/ready` | Readiness (DB + event bus) |
| `GET` | `/api/v1/health/detail` | Full diagnostic status |

Full interactive docs at `http://localhost:8000/docs`.

---

## Frontend Lab UI

The primary UI is a single-page app at `http://localhost:8000/lab`.

**Three-column layout:**

```
┌────────────────────┬──────────────────────────┬──────────────────────┐
│  Instrument Bar    │   Agent Pipeline          │  Context Panel       │
│  (active devices)  │   Round 1                 │  (selected step)     │
│                    │   ├─ Safety Check  ✓      │                      │
│  [Squidstat]       │   ├─ Simulation   ✓      │  Decision Tree:      │
│  [OT-2]            │   ├─ Execution    ✓      │  ├─ Strategy: LHS    │
│                    │   ├─ Analysis     ✓      │  ├─ Rounds: 20       │
│  NL Input          │   Round 2                 │  └─ Convergence: …   │
│  [text area]       │   ├─ Safety Check  …      │                      │
│  [Run Campaign]    │   └─ ...                  │  Thinking Log        │
└────────────────────┴──────────────────────────┴──────────────────────┘
```

**Key UI behaviors:**
- Paste any free-text experiment description and click **Run Campaign**
- The pipeline panel populates in real time via SSE as each agent starts/finishes
- Click any step to see the **decision tree** in the Context Panel — what each agent considered, which option it chose, and why
- Instrument chips in the Instrument Bar reflect live hardware status

---

## Testing

```bash
# All tests
pytest tests/

# Verbose with coverage
pytest -v --cov=app tests/

# Specific module
pytest tests/test_multi_agent_v3.py
```

| Test File | Coverage Area |
|-----------|--------------|
| `test_agent_runtime_integration.py` | Multi-agent orchestration, pause/approval workflows |
| `test_multi_agent_v3.py` | ControlPlane, agent leasing, concurrency |
| `test_durable_execution.py` | Campaign lifecycle: duplicate-start guard, recovery, bounded retention |
| `test_sse_events.py` | Exactly-once SSE delivery, replay/live handover, queue cleanup |
| `test_e2e_study.py` | Full campaign end-to-end (simulated) |
| `test_gp_surrogate.py` | GP surrogate model and acquisition functions |
| `test_simulation.py` | Protocol simulation engine |
| `test_mission_control.py` | Mission/workflow API |
| `test_requirement_parser_agent.py` | Natural-language requirement parsing |

Type checking and lint:

```bash
mypy app/
ruff check app/
ruff format app/
```

### Validation Evidence Pack

HELIOS is framed as an agent-native, hierarchical, graph/state-machine routed,
role-based multi-agent SDL controller. The live campaign controller remains
rule-based and auditable by default; Nexus and BO MCP are optimization
advisor/backend/tool paths, not top-level campaign decision authorities. Learned
policy and self-evolution paths are offline, shadow, canary, and approval-gated;
their metadata does not change default BO MCP/Nexus/backend behavior.

The architecture validation report is version-controlled at
`docs/HELIOS_ARCHITECTURE_VALIDATION.md`. It is a static evidence pack for the
current validation boundary, not a runtime-generated artifact.

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

Full `pytest` is expected to pass. Full-repository `ruff check .` currently has
legacy lint debt in older modules and vendored/benchmark-style test areas. The
clean ruff boundary for the new validation/reporting files is:

```bash
ruff check \
  app/optimization/candidate_memory.py \
  app/optimization/failure_zone_memory.py \
  app/services/campaign_state.py \
  app/services/system_validation_report.py \
  tests/test_candidate_memory.py \
  tests/test_failure_zone_memory.py \
  tests/test_system_validation_report.py \
  tests/test_offline_closed_loop_sdl.py \
  tests/test_offline_scenario_benchmarks.py \
  tests/test_policy_evolution_workflow_e2e.py
```

---

## Project Structure

```
HELIOS/
├── app/
│   ├── agents/              # 27 specialist agents + swarm/control-plane runtime
│   ├── api/v1/endpoints/    # FastAPI route handlers
│   ├── optimization/        # Nexus/local provider facade, candidate pools, arbitration
│   ├── services/            # 85+ domain services
│   │   ├── bayesian_opt.py  # Bayesian Optimization (Ax)
│   │   ├── campaign_loop.py # Campaign execution loop
│   │   ├── campaign_events.py # SSE event persistence & replay
│   │   ├── convergence*.py  # Termination criteria
│   │   ├── rl_*.py          # DQN / PPO strategy backends
│   │   └── nexus_advisor.py # Causal inference integration
│   ├── hardware/            # Hardware adapters (OT-2, PLC, relay, sensors)
│   ├── adapters/            # Lab-mode adapters (simulated, battery lab)
│   ├── contracts/           # Pydantic data models (TaskContract, etc.)
│   ├── core/                # DB init, config, startup lifecycle
│   ├── static/              # Frontend (lab.html / lab.js / lab.css)
│   ├── main.py              # FastAPI app entry point + lifespan
│   └── worker.py            # Background async worker
├── recovery-agent/          # Standalone hardware bridge (port 8001)
├── tests/                   # Pytest test suite
├── benchmarks/              # Performance tests and fault injection
├── examples/                # Demo scripts
├── models/                  # Pre-trained RL model checkpoints (.pkl)
├── data/                    # Runtime SQLite DB and object store (gitignored)
├── Dockerfile               # Multi-variant build (simulated / hardware / ml / all)
├── docker-compose.yml       # Two-service deployment
├── pyproject.toml           # Dependencies and tool config
└── .env.example             # Environment variable template
```

---

## Deployment

### Docker Compose (recommended)

```yaml
# docker-compose.yml provides:
# - helios       : main service on :8000, with SQLite volume
# - recovery-agent : hardware bridge on :8001 (profile: hardware)
```

```bash
# Development
docker compose up

# Production (with hardware)
docker compose --profile hardware up -d

# View logs
docker compose logs -f helios
```

### Docker Build Variants

```bash
# Simulated only (smallest image, default)
docker build -t helios .

# With hardware serial drivers
docker build --build-arg EXTRAS=hardware -t helios:hw .

# With ML strategy models
docker build --build-arg EXTRAS=ml -t helios:ml .

# Full stack
docker build --build-arg EXTRAS=all -t helios:full .
```

### Health Checks

```bash
curl http://localhost:8000/health                  # Liveness
curl http://localhost:8000/api/v1/health/ready     # Readiness
curl http://localhost:8000/api/v1/health/detail    # Full diagnostic
```

---

## Event-Driven Architecture

All internal communication flows through an async event bus. Key event types:

| Event | Trigger | Consumers |
|-------|---------|-----------|
| `CandidateExecuted` | Hardware run completes | Metrics, Analyzer, Memory |
| `MetricsUpdated` | KPI recomputed | Dashboard, Convergence |
| `ApprovalRequested` | Safety gate triggered | UI confirmation dialog |
| `KPIReached` | Objective met | Campaign termination |

Campaign events are persisted to the `campaign_events` table so SSE streams replay them on reconnect — the UI receives the full history even if it connects after the campaign has completed.

---

## External Integrations

| Integration | Purpose |
|-------------|---------|
| **Anthropic / OpenAI** | LLM backend for agent reasoning |
| **Opentrons OT-2 / Flex** | Liquid-handling robotics |
| **Ax (Meta)** | Bayesian Optimization service |
| **Nexus Advisor** | Causal inference for experimental design |
| **Potentiostats** | Electrochemical measurements (serial adapters) |
| **PLC / relay controllers** | Pump, stirrer, and process control |

---

## Contributing

1. Fork and create a feature branch
2. Code style:
   - Type hints always; typed Pydantic models for all agent I/O
   - `ruff check` + `ruff format` before committing
   - Conventional commits (`feat/fix/refactor/chore/test/docs`)
3. Write tests alongside implementation, not after
4. Open a PR with type check, lint, and the full test suite passing

---

## License

MIT — see [LICENSE](LICENSE) for details.
