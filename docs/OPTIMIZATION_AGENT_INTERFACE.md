# Optimization Agent — Interface Specification

> **Purpose**: Define the boundary, input/output contracts, and integration seams
> for a new `OptimizationAgent` to be developed by a separate contributor.
>
> **Last updated**: 2025-01-XX | **System version**: 10 agents, 1387 tests passing

---

## 1. Architecture Context

### Current Agent Inventory (10 agents)

```
┌─────────────────────────────────────────────────────────────────┐
│  OrchestratorAgent (top)                                        │
│    │                                                            │
│    ├─ PlannerAgent (L2)      — 多轮排程 (round count, phases)   │
│    ├─ DesignAgent (L2)       — 单轮采样 (candidate params)      │
│    ├─ CompilerAgent (L1)     — Protocol → DAG 编译              │
│    ├─ CodeWriterAgent (L1)   — NL → OT-2 Python code           │
│    ├─ SafetyAgent (xcut)     — 安全否决                         │
│    ├─ SensingAgent (L0)      — 步骤级 QC + 异常检测             │
│    ├─ StopAgent (L0)         — 收敛/停止判断                    │
│    ├─ QueryAgent (xcut)      — NL → SQL 查询编译器 (独立)       │
│    └─ OnboardingAgent (L3)   — 仪器接入                        │
│                                                                 │
│  OptimizationAgent (L2, NEW) — 替换 strategy_selector 服务      │
└─────────────────────────────────────────────────────────────────┘
```

### What OptimizationAgent Is NOT

| Component | Role | Relationship to OptimizationAgent |
|-----------|------|-----------------------------------|
| **PlannerAgent** | "10轮实验, 前3轮explore" — **结构规划** | 不替换。Planner决定有几轮，OptAgent决定每轮怎么优化 |
| **DesignAgent** | "给我8个候选参数" — **采样执行** | 不替换。DesignAgent调用`generate_batch()`，OptAgent告诉它用什么策略 |
| **strategy_selector** | 14诊断信号 → 选策略后端 — **决策服务** | **被替换/增强**。OptAgent的核心职责 |
| **QueryAgent** | NL → SQL 查数据 — **独立工具** | 不包含。可选用作数据源 |
| **StopAgent** | 收敛检测 → continue/stop — **终止判断** | 不替换。可选增强 |

---

## 2. Layer & Lifecycle

**Layer**: `L2` (Planning Layer)

**Lifecycle**: Stateless — 每轮调用一次，不持有跨轮状态。
历史数据通过 `CampaignSnapshot` 传入。

**Base class**: `BaseAgent[OptimizationInput, OptimizationOutput]`

```python
from app.agents.base import BaseAgent

class OptimizationAgent(BaseAgent[OptimizationInput, OptimizationOutput]):
    name = "optimization_agent"
    description = "Intelligent optimization strategy selection and tuning"
    layer = "L2"
```

### BaseAgent Protocol (must implement)

```python
class BaseAgent(ABC, Generic[InputT, OutputT]):
    name: str
    description: str
    layer: str

    def validate_input(self, input_data: InputT) -> list[str]:
        """Return empty list if valid, error strings if not."""
        ...

    async def process(self, input_data: InputT) -> OutputT:
        """Core logic. Called by run() after validation."""
        ...

    # DO NOT override run() — it handles validation + timing + error wrapping
    async def run(self, input_data: InputT) -> AgentResult[OutputT]:
        ...
```

---

## 3. Integration Seam: Where It Plugs In

### Exact Location in Orchestrator

**File**: `app/agents/orchestrator.py`, lines 380-422

**Current code** (to be replaced):

```python
# In OrchestratorAgent.process(), inside the round loop:
if round_strategy == "adaptive" and kpi_history:
    from app.services.strategy_selector import (
        CampaignSnapshot,
        select_strategy,
    )
    snapshot = CampaignSnapshot(
        round_number=round_num,
        max_rounds=input_data.max_rounds,
        n_observations=total_runs,
        # ... 20+ fields
    )
    decision = select_strategy(snapshot)       # ← THIS GETS REPLACED
    round_strategy = decision.backend_name
```

**After integration**:

```python
if round_strategy == "adaptive" and kpi_history:
    from app.agents.optimization_agent import OptimizationAgent, OptimizationInput

    snapshot = CampaignSnapshot(...)           # same as before
    opt_agent = OptimizationAgent()
    opt_input = OptimizationInput(snapshot=snapshot)
    opt_result = await opt_agent.run(opt_input)

    if opt_result.success:
        decision = opt_result.output.strategy_decision
        round_strategy = decision.backend_name
    else:
        # Fallback to existing service
        decision = select_strategy(snapshot)
        round_strategy = decision.backend_name
```

### Execution Flow per Round

```
Orchestrator (each round)
    │
    ├─ 1. Build CampaignSnapshot (kpi_history, all_params, all_kpis, ...)
    │
    ├─ 2. ──→ OptimizationAgent.run(OptimizationInput)     ← NEW
    │         ├─ Compute enhanced diagnostics
    │         ├─ Select surrogate model
    │         ├─ Tune acquisition function
    │         ├─ Handle constraints
    │         └─ Return StrategyDecision + extras
    │     ←── OptimizationOutput
    │
    ├─ 3. If stabilize_spec → create replicate candidates directly
    │      Else → DesignAgent.run(DesignInput(strategy=decision.backend_name))
    │
    ├─ 4. CompilerAgent → SafetyAgent → Execute → SensingAgent
    │
    ├─ 5. Collect KPI results → feed into next round's CampaignSnapshot
    │
    └─ 6. StopAgent → continue/stop decision
```

---

## 4. Input Contract

### OptimizationInput (Pydantic model)

```python
from pydantic import BaseModel
from app.services.strategy_selector import CampaignSnapshot

class OptimizationInput(BaseModel):
    """Everything the OptimizationAgent needs to make a decision."""

    snapshot: CampaignSnapshot    # REQUIRED — the main data payload (see §4.1)

    # Optional overrides
    force_backend: str | None = None          # bypass decision, use this backend
    force_phase: str | None = None            # "explore"|"exploit"|"refine"|"stabilize"
    extra_constraints: list[dict] | None = None  # custom parameter constraints
```

### 4.1 CampaignSnapshot (existing dataclass, DO NOT MODIFY)

```python
@dataclass(frozen=True)
class CampaignSnapshot:
    # ---- Campaign structure ----
    round_number: int                          # current round (1-based)
    max_rounds: int                            # budget
    n_observations: int                        # total evaluations so far
    n_dimensions: int                          # parameter space size
    has_categorical: bool                      # any categorical dims?
    has_log_scale: bool                        # any log-scale dims?
    direction: str = "maximize"                # "minimize" | "maximize"
    user_strategy_hint: str = ""               # user override if any
    available_backends: dict[str, bool] = {}   # which backends installed

    # ---- KPI history ----
    kpi_history: tuple[float, ...] = ()        # ALL KPIs observed
    best_kpi_so_far: float | None = None       # running best

    # ---- Last round data ----
    last_batch_kpis: tuple[float, ...] = ()    # KPIs from previous round
    last_batch_params: tuple[dict, ...] = ()   # params from previous round

    # ---- Full observation history ----
    all_params: tuple[dict, ...] = ()          # ALL params ever sampled
    all_kpis: tuple[float, ...] = ()           # ALL KPIs ever observed

    # ---- QC data ----
    qc_fail_rate: float = 0.0                  # fraction failed QC
```

**Key guarantee**: `len(all_params) == len(all_kpis)` — paired observations.

---

## 5. Output Contract

### OptimizationOutput (Pydantic model)

```python
from pydantic import BaseModel
from app.services.strategy_selector import StrategyDecision

class OptimizationOutput(BaseModel):
    """The optimization decision for this round."""

    strategy_decision: StrategyDecision   # REQUIRED — must be compatible (see §5.1)

    # Optional enhanced outputs
    surrogate_info: dict | None = None    # which surrogate model, hyperparams, confidence
    acquisition_tuning: dict | None = None  # xi, kappa, or custom params
    landscape_analysis: dict | None = None  # multimodality, intrinsic dim, etc.
    warnings: list[str] = []              # advisory messages for the orchestrator
```

### 5.1 StrategyDecision (existing dataclass, DO NOT MODIFY)

```python
@dataclass
class StrategyDecision:
    backend_name: str                      # REQUIRED — "built_in"|"optuna_tpe"|"optuna_cmaes"|"scipy_de"|"pymoo_nsga2"
    phase: str                             # "exploration"|"exploitation"|"refinement"|"stabilize"
    reason: str                            # human-readable explanation
    confidence: float                      # 0.0–1.0
    fallback_backend: str = "built_in"     # if primary fails

    # Optional enrichment (all None-able)
    diagnostics: DiagnosticSignals | None = None
    phase_posterior: PhasePosterior | None = None
    actions_considered: tuple[ActionCandidate, ...] = ()
    explanation: str = ""                  # 3-line summary for SSE events
    weights_used: WeightsUsed | None = None
    drift_score: float | None = None
    evidence: tuple[EvidenceItem, ...] = ()
    stabilize_spec: StabilizeSpec | None = None   # If phase=stabilize, MUST provide this
```

### Available Backend Names

| backend_name | Method | Always Available? |
|---|---|---|
| `"built_in"` | KNN surrogate + EI/UCB | ✅ Yes (stdlib) |
| `"optuna_tpe"` | Tree-structured Parzen Estimator | If `optuna` installed |
| `"optuna_cmaes"` | CMA-ES via Optuna | If `optuna` installed |
| `"scipy_de"` | Differential Evolution | If `scipy` installed |
| `"pymoo_nsga2"` | NSGA-II evolutionary | If `pymoo` + `numpy` installed |

**Check availability**: `snapshot.available_backends` dict (name → bool).

---

## 6. What Orchestrator Does With Your Output

```python
# orchestrator.py after receiving OptimizationOutput:

decision = opt_output.strategy_decision
round_strategy = decision.backend_name     # passed to DesignAgent

if decision.stabilize_spec is not None:
    # BYPASS DesignAgent entirely — create replicate candidates directly
    for point in decision.stabilize_spec.points_to_replicate:
        for _ in range(decision.stabilize_spec.n_replicates):
            candidates.append(point)
else:
    # Normal path — DesignAgent generates candidates using your chosen strategy
    design_input = DesignInput(
        dimensions=input_data.dimensions,
        protocol_template=input_data.protocol_template,
        strategy=round_strategy,           # FROM YOUR DECISION
        batch_size=planned_round.batch_size,
        seed=round_num,
        campaign_id=campaign_id,
        kpi_name=input_data.objective_kpi,
    )
    design_result = await design_agent.run(design_input)
    candidates = design_result.output.candidates

# SSE event emitted with decision.explanation
log_event(campaign_id, "strategy_decision", {
    "round": round_num,
    "strategy": round_strategy,
    "phase": decision.phase,
    "explanation": decision.explanation,
    "diagnostics": asdict(decision.diagnostics) if decision.diagnostics else None,
})
```

---

## 7. Existing Services You CAN Use

### Import directly — no need to reimplement

```python
# Diagnostics computation (14 signals)
from app.services.strategy_selector import (
    CampaignSnapshot,       # input dataclass
    DiagnosticSignals,      # 14 signals
    PhasePosterior,         # soft phase probabilities
    StrategyDecision,       # output dataclass
    StabilizeSpec,          # replication plan
    PhaseConfig,            # threshold config
    compute_diagnostics,    # CampaignSnapshot → DiagnosticSignals
    select_strategy,        # full pipeline (can use as fallback)
)

# Convergence detection (3-method voting)
from app.services.convergence import detect_convergence  # → ConvergenceResult

# Bayesian surrogate (KNN)
from app.services.bayesian_opt import (
    SurrogateModel,              # kNN surrogate
    expected_improvement,        # EI acquisition function
    upper_confidence_bound,      # UCB acquisition function
    sample_bo,                   # full BO sampling pipeline
    load_observations_from_db,   # load historical obs from DB
    Observation,                 # (params_unit, objective) pair
)

# Backend registry
from app.services.optimization_backends import (
    list_backends,          # → dict[str, bool]
    get_backend,            # → BackendProtocol instance
)

# DB access (if you need historical data beyond CampaignSnapshot)
from app.core.db import connection  # read-only queries
```

### QueryAgent (optional data source)

QueryAgent is a **standalone** NL → SQL agent. It is **NOT** in the optimization loop.
If your OptimizationAgent needs cross-campaign historical analysis:

```python
from app.agents.query_agent import QueryAgent
from app.contracts.query_contract import QueryRequest

# Example: query historical performance across campaigns
agent = QueryAgent()
result = await agent.run(QueryRequest(
    prompt="average best KPI by strategy for the last 10 campaigns",
    constraints=QueryConstraints(max_rows=100),
))
if result.success:
    historical_data = result.output.rows
```

**Note**: This is optional. Most optimization decisions can be made from
`CampaignSnapshot` alone (which contains full observation history for
the current campaign).

---

## 8. File Locations

### Files to CREATE

```
app/agents/optimization_agent.py       # The agent class
app/contracts/optimization_contract.py  # OptimizationInput/Output models
tests/test_optimization_agent.py       # Tests
```

### Files to MODIFY (integration)

```
app/agents/orchestrator.py             # Replace select_strategy() call (§3)
app/agents/__init__.py                 # Add exports
app/contracts/__init__.py              # Add exports
```

### Files to READ (understand, don't modify)

```
app/agents/base.py                     # BaseAgent protocol
app/services/strategy_selector.py      # Current implementation (1700 lines)
app/services/candidate_gen.py          # generate_batch() and strategies
app/services/bayesian_opt.py           # KNN surrogate + EI/UCB
app/services/convergence.py            # 3-method convergence detection
app/services/optimization_backends.py  # Backend registry
app/agents/design_agent.py             # Candidate generation agent
app/agents/stop_agent.py               # Stop/continue decision
```

---

## 9. Testing Contract

### Minimum test requirements

1. **Unit**: OptimizationAgent with mock CampaignSnapshot → valid StrategyDecision
2. **Fallback**: If OptAgent fails → orchestrator falls back to `select_strategy()`
3. **Backend validation**: `decision.backend_name` must be in `snapshot.available_backends`
4. **Stabilize**: If `phase == "stabilize"` → `stabilize_spec` must not be None
5. **Determinism**: Same snapshot → same decision (for cache compatibility)
6. **Regression**: All existing 1387 tests still pass

### Run tests

```bash
# Your tests only
python3 -m pytest tests/test_optimization_agent.py -x -v

# Full regression
python3 -m pytest tests/ -x
```

---

## 10. Constraints & Rules

1. **Stateless**: No mutable instance state. All data comes via `OptimizationInput`.
2. **Pure Python preferred**: stdlib only in critical path. Optional deps (scipy, optuna) OK for enhanced features.
3. **Async**: `process()` must be `async def`.
4. **No DB writes**: Read-only. The orchestrator handles all state persistence.
5. **Fallback-safe**: If your agent raises an exception, the orchestrator falls back to `select_strategy()`.
6. **Don't modify existing dataclasses**: `CampaignSnapshot`, `StrategyDecision`, `DiagnosticSignals` are frozen/shared. Extend via your own output fields.
7. **Don't modify DesignAgent**: Your output (strategy name) is passed to DesignAgent as-is. You influence *which* strategy, not *how* it samples.

---

## 11. Summary: Boundary Diagram

```
                    ┌──────────────────────────────────────┐
                    │         YOUR BOUNDARY                 │
                    │                                      │
  CampaignSnapshot ─┤  OptimizationAgent                   │
  (14 signals,      │    │                                 │
   full KPI history,│    ├─ Enhanced diagnostics            │
   all params/kpis) │    ├─ Surrogate model selection       │
                    │    ├─ Acquisition function tuning     │
                    │    ├─ Constraint handling              │
                    │    ├─ Failure recovery logic           │
                    │    └─ Stabilize spec (if needed)      │
                    │                                      │
                    ├─→ OptimizationOutput                  │
                    │    ├─ strategy_decision: StrategyDecision (REQUIRED)
                    │    ├─ surrogate_info (optional)       │
                    │    ├─ acquisition_tuning (optional)   │
                    │    └─ landscape_analysis (optional)   │
                    └──────────────────────────────────────┘
                                    │
                    Orchestrator uses │ decision.backend_name
                                    │ decision.stabilize_spec
                                    ↓
                    DesignAgent.run(strategy=backend_name)
                                    │
                                    ↓
                              candidates → compile → execute → KPI
                                                               │
                                    next round ←───────────────┘
```
