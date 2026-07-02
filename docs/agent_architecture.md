# HELIOS Agent Architecture — and where the LLM sits

## TL;DR

HELIOS is an **autonomous experimentation agent**, not an "LLM agent". Its
per-round decision loop is **LLM-free by design**: strategy and candidate
selection are classical Bayesian optimization + rule-based scoring + optional
learned (non-LLM) policies. The LLM is used **only at the language/knowledge
boundary** (turning human intent into plans/code, injecting priors, scoring
runs after the fact) and never steers a live optimization round.

If someone asks "so where is the agent?" — the agency is in the closed loop
that **perceives → decides → acts on real instruments → learns**, under bounded
authority, not in a chat model calling tools.

---

## What "agent" means in HELIOS

An agent here is **not** "LLM + tools + a ReAct loop". It is a typed,
deterministic service:

- `BaseAgent` (`app/agents/base.py`) — a typed input→output unit with
  `validate_input`, an async `process`, a declared `layer`
  (L3/L2/L1/L0/cross-cutting), and static `capabilities`. Nothing about it
  requires an LLM.
- The orchestrator (`app/agents/orchestrator.py`) runs a **fixed, deterministic
  pipeline** per round: L3 intake → L2 planning/design → L1 compile/safety/sim
  → L0 execute/sense/stop. It is not an LLM control loop.
- The ControlPlane (`app/agents/control_plane.py`) provides capability registry,
  reentrant agent leases, audit/provenance, and pause / human-in-the-loop gates.

There is **no tool-calling / function-calling / ReAct agentic loop** anywhere in
the codebase. Of ~26 agents, only ~4 call an LLM (see below), and each does so
once per invocation with no feedback loop.

---

## Where the LLM is / isn't

Default provider is `mock` (`LLM_PROVIDER=mock`) — zero real LLM calls unless
explicitly configured to `anthropic`/`openai`.

| LLM caller | File | Purpose | On the per-round loop? |
|------------|------|---------|------------------------|
| Planner | `app/services/planner.py` | NL intent → structured plan (human-approval gated) | **No** (setup) |
| CodeWriter / NLPCode | `app/agents/code_writer_agent.py`, `nlp_code_agent.py` | NL → protocol code (generative mode only) | **No** (compile-time) |
| Onboarding | `app/agents/onboarding_agent.py` | Discover instrument primitives (KB fallback) | **No** (lab setup) |
| Inverse design | `app/agents/inverse_design_agent.py` | Recommendation priors (KB fallback) | **No** (setup) |
| Query | `app/agents/query_agent.py` | NL → SQL (cache-first) | **No** (analysis) |
| Reviewer | `app/services/reviewer.py` | Post-run scoring / failure notes (advisory, non-blocking) | **No** (post-hoc) |

**The per-round decision path is classical / rule / ML, never LLM:**

- Strategy selection — `strategy_selector.py` / `strategy_scoring.py`:
  phase-posterior softmax + weighted expected-utility scoring (rules).
- Candidate generation — `bayesian_opt.py`, `gp_surrogate.py`,
  `candidate_gen.py`, `optimization_backends.py`: GP/BO math (EI/UCB/Thompson),
  LHS/Sobol/random, Nexus backends.
- Adaptation — `backend_memory.py` (`ContextualStrategyBandit`, UCB) and
  optional PyTorch policies `dqn_strategy_selector.py` / `ppo_strategy_selector.py`
  (neural nets, **not** LLMs).
- Adaptive substrate — `campaign_mode.py`, `dynamic_action_space.py`,
  `value_of_information.py`, `objective_state.py`, `failure_attribution.py`:
  deterministic and shadow-only (see
  [adaptive_campaign_substrate.md](adaptive_campaign_substrate.md)).

---

## Where the agency lives

HELIOS is agentic in the classical (autonomous-systems) sense — a goal-directed
closed loop:

- **Perceive** — diagnostics (uncertainty, noise, convergence, drift), QC
  results, typed failure signals.
- **Decide** — campaign intent / optimization mode / backend, under **bounded
  authority** (HELIOS keeps campaign-level decision authority; Nexus and BO are
  backends + advisors, not the top decision-maker).
- **Act** — compile and execute protocols on real or simulated hardware; recover
  from failures.
- **Learn** — candidate / failure-zone memory, contextual bandit, optional RL,
  replay/evaluation.

The multi-agent structure is a **division of labor** across specialist typed
agents (Planner, Design, Compiler, Safety, Simulation, Execution, Recovery,
Sensing, Analyzer, Stop, …) plus safety/recovery vetoes, all orchestrated with
provenance and human-in-the-loop gates. It is a multi-agent *system* — just not
an LLM swarm.

---

## Why the LLM is kept out of the loop (deliberate)

Scientific decisions must be **reproducible, auditable, safe, cheap, low-latency,
and fail-closed**. An LLM in the per-round loop would be non-deterministic,
hallucination-prone, slow, expensive, and hard to audit. HELIOS therefore uses a
clean separation (matching the repo's `CLAUDE.md` principles):

> A model proposes · a policy validates · a runtime executes · a log explains ·
> a test proves.

The LLM **proposes at the edges** (translate human intent, generate code, inject
domain priors); a **deterministic, auditable policy decides in the core**.

---

## What is *not* yet done (LLM-in-loop scientific reasoning)

Putting LLM-style scientific reasoning inside the loop — hypothesis generation,
mechanism-driven route pivots (e.g. "round 1 electrodeposition → round 2 switch
to gel synthesis") — is **future work**, deliberately deferred. See
[development_progress.md](development_progress.md) (B1 HypothesisState, B7
parameter-space / synthesis-route revision). The current loop is deterministic
on purpose.

---

## One-paragraph answer to "where is the agent?"

> HELIOS is an autonomous experimentation agent: it closed-loop plans, executes
> on real instruments, senses, recovers, and adapts from feedback, built as a
> multi-agent system of typed specialist agents. It is deliberately **not** an
> "LLM agent" — the LLM works only at the human-language and domain-knowledge
> boundary (intent → plan, NL → code, priors), while the scientific decision
> loop stays deterministic, auditable, and safe. Agency comes from autonomy +
> action + goal-directed adaptation, not from an LLM running in a while-loop.
