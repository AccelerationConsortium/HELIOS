# OTbot — Autonomous Laboratory Orchestrator

OTbot is a multi-agent system for self-driving laboratories (SDLs). Scientists describe experiments in plain language; OTbot plans, validates, executes, and iterates autonomously — closing the loop between hypothesis and hardware.

```
Scientist (natural language) → NL Parser → Campaign Planner → Safety Gate
→ Protocol Compiler → Hardware Execution → Data Collection → Bayesian / RL Optimization
→ ... (next round)
```

---

## Features

- **Natural language intake** — paste a free-text experiment description; the NL parser extracts objective KPIs, parameter spaces, instrument requirements, and round counts
- **Multi-agent pipeline** — 24 specialist agents (planner, safety, simulation, analyzer, compiler, monitor, recovery, …) orchestrated as a campaign
- **Real-time reasoning stream** — every agent step emits SSE events; the browser shows a live decision tree of what each agent considered and why
- **Hardware agnostic** — runs in `simulated` mode for development; switches to live Opentrons OT-2, PLC relays, and electrochemistry sensors by changing one env var
- **Adaptive optimization** — Bayesian Optimization (Ax), DQN, PPO, genetic algorithms, and multi-objective Pareto search, selected automatically per campaign phase
- **Safety-first** — preflight checks before every round; human-in-the-loop confirmation gates for high-risk operations; full audit trail
- **Persistent state** — SQLite-backed campaign history survives restarts; SSE replays all missed events on reconnect

---

## Architecture

### Four Layers

| Layer | Role | Key Components |
|-------|------|----------------|
| **L3 Intake** | Campaign coordination | `Orchestrator`, `NLParser` |
| **L2 Planning** | Experimental design & strategy | `PlannerAgent`, `DesignAgent`, `SafetyAgent` |
| **L1 Compilation** | Protocol → executable code | `CompilerAgent`, `CodeWriterAgent`, `DeckLayoutAgent` |
| **L0 Execution** | Hardware control & monitoring | `MonitorAgent`, `SensingAgent`, `RecoveryAgent` |

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
git clone https://github.com/your-org/OTbot.git
cd OTbot

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
| `OTBOT_PORT` | `8000` | Main service port |
| `RECOVERY_PORT` | `8001` | Hardware recovery bridge port |
| `DB_PATH` | `/app/data/orchestrator.db` | SQLite database path |

---

## API Overview

Base URL: `http://localhost:8000`

### Campaign Lifecycle

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/orchestrate/start` | Start a campaign from a `TaskContract` |
| `GET` | `/api/v1/orchestrate/{campaign_id}/status` | Query campaign state and progress |
| `GET` | `/api/v1/orchestrate/{campaign_id}/events/stream` | SSE event stream for a campaign |

### Natural Language Interface

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/nl/parse` | Parse free-text description → `TaskContract` |

### Initialization & Onboarding

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/init/start` | Start interactive setup session |
| `POST` | `/api/v1/init/{session_id}/respond` | Respond to initialization prompts |
| `POST` | `/api/v1/onboarding/start` | Onboard a new hardware device |

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
| `GET` | `/health/ready` | Readiness (DB + event bus) |
| `GET` | `/health/detail` | Full diagnostic status |

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
pytest tests/test_strategy_router.py

# End-to-end simulated campaign
ADAPTER_MODE=simulated LLM_PROVIDER=mock python tests/e2e_simulated.py
```

| Test File | Coverage Area |
|-----------|--------------|
| `test_spectral_store.py` | Spectroscopic data persistence |
| `test_strategy_router.py` | Optimization strategy routing |
| `test_query_dsl.py` | Query language parsing |
| `test_nexus_integration.py` | Causal inference (Nexus) |
| `test_simulation.py` | Physics simulation validation |
| `e2e_simulated.py` | Full campaign end-to-end (simulated) |

Type checking and lint:

```bash
mypy app/
ruff check app/
ruff format app/
```

---

## Project Structure

```
OTbot/
├── app/
│   ├── agents/              # 24 specialist agents
│   ├── api/v1/endpoints/    # FastAPI route handlers
│   ├── services/            # 73+ domain services
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
# - otbot       : main service on :8000, with SQLite volume
# - recovery-agent : hardware bridge on :8001 (profile: hardware)
```

```bash
# Development
docker compose up

# Production (with hardware)
docker compose --profile hardware up -d

# View logs
docker compose logs -f otbot
```

### Docker Build Variants

```bash
# Simulated only (smallest image, default)
docker build -t otbot .

# With hardware serial drivers
docker build --build-arg EXTRAS=hardware -t otbot:hw .

# With ML strategy models
docker build --build-arg EXTRAS=ml -t otbot:ml .

# Full stack
docker build --build-arg EXTRAS=all -t otbot:full .
```

### Health Checks

```bash
curl http://localhost:8000/health          # Liveness
curl http://localhost:8000/health/ready    # Readiness
curl http://localhost:8000/health/detail   # Full diagnostic
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
| **Squidstat potentiostat** | Electrochemical measurements |
| **PLC controllers** | Relay and process control |

---

## Contributing

1. Fork and create a feature branch
2. Follow the code style rules in [CLAUDE.md](.claude/CLAUDE.md):
   - Type hints always; `mypy --strict` for new code
   - `ruff check` + `ruff format` before committing
   - Conventional commits (`feat/fix/refactor/chore/test/docs`)
3. Write tests alongside implementation, not after
4. Open a PR — CI runs type check, lint, and the full test suite

---

## License

See [LICENSE](LICENSE) for details.
