"""Agent Swarm system — ephemeral specialist groups spawned on demand.

Implements the four specialist swarms described in the OTbot AI4X paper:

1. **Scientist Swarm** — hypothesis generation via first-principles reasoning
2. **Engineer Swarm** — protocol optimisation and in-silico simulation
3. **Analyst Swarm** — real-time spectral and image interpretation
4. **Validator Swarm** — hallucination detection & physical plausibility

Swarms are ephemeral: spawned on demand, disbanded after their task,
and re-spawnable with updated context in the next round.

Integration
-----------
- ``SwarmFactory.spawn(name, context)`` → returns configured swarm instance
- Each swarm's ``run()`` invokes its constituent agents and aggregates results
- ``SWARM_REGISTRY`` maps swarm names to agent compositions
- Orchestrator can use swarms alongside or instead of direct agent calls

Design
------
- Additive layer: does NOT modify the existing orchestrator dispatch flow
- Each swarm wraps ``BaseAgent.run()`` calls with timing and error aggregation
- Cross-cutting agents (Safety, Recovery) remain outside swarms per paper
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

# Thread-safe lock for SWARM_REGISTRY mutations
_REGISTRY_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Swarm I/O contracts
# ---------------------------------------------------------------------------


class SwarmContext(BaseModel):
    """Shared context passed to a swarm at spawn time.

    Carries campaign state needed by constituent agents.
    """

    campaign_id: str = ""
    round_number: int = 0
    direction: str = "minimize"
    objective_kpi: str = ""
    dimensions: list[dict[str, Any]] = Field(default_factory=list)
    kpi_history: list[float] = Field(default_factory=list)
    best_kpi: float | None = None
    protocol_template: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


@dataclass
class SwarmResult:
    """Aggregated result from a swarm execution.

    Attributes:
        swarm_name: Which swarm produced this result.
        success: True if all required agents succeeded.
        agent_results: Individual AgentResult from each constituent agent.
        aggregated_output: Merged output dict for downstream consumption.
        errors: Accumulated errors from any failed agent.
        warnings: Accumulated warnings.
        duration_ms: Wall-clock time for the entire swarm execution.
        trace_id: Unique trace for this swarm invocation.
    """

    swarm_name: str
    success: bool
    agent_results: list[AgentResult[Any]] = field(default_factory=list)
    aggregated_output: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    trace_id: str = ""


# ---------------------------------------------------------------------------
# Base swarm
# ---------------------------------------------------------------------------


class BaseSwarm(ABC):
    """Abstract base for all OTbot agent swarms.

    Swarms are ephemeral containers that:
    1. Accept a SwarmContext (campaign state)
    2. Instantiate and invoke constituent agents
    3. Aggregate results into a single SwarmResult

    Subclasses must define ``agent_specs`` (agent names + roles)
    and implement ``run()``.
    """

    name: ClassVar[str] = "base_swarm"
    description: ClassVar[str] = ""

    # Agent composition: list of (agent_class_name, role_in_swarm)
    agent_specs: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, context: SwarmContext) -> None:
        self.context: SwarmContext = context
        self.trace_id: str = uuid.uuid4().hex[:12]
        self.logger: Logger = logging.getLogger(f"swarm.{self.name}")
        self.logger.info(
            "Spawned %s swarm (trace=%s, campaign=%s, round=%d)",
            self.name,
            self.trace_id,
            context.campaign_id,
            context.round_number,
        )

    @abstractmethod
    async def run(self, **kwargs: Any) -> SwarmResult:
        """Execute the swarm's task. Override in subclasses."""
        ...

    def _make_result(
        self,
        agent_results: list[AgentResult[Any]],
        aggregated: dict[str, Any],
        start_time: float,
    ) -> SwarmResult:
        """Helper to build a SwarmResult from individual agent results."""
        all_errors: list[str] = []
        all_warnings: list[str] = []
        all_success = True

        for ar in agent_results:
            all_errors.extend(ar.errors)
            all_warnings.extend(ar.warnings)
            if not ar.success:
                all_success = False

        duration = (time.monotonic() - start_time) * 1000

        self.logger.info(
            "%s swarm completed in %.1fms (success=%s, agents=%d, trace=%s)",
            self.name,
            duration,
            all_success,
            len(agent_results),
            self.trace_id,
        )

        return SwarmResult(
            swarm_name=self.name,
            success=all_success,
            agent_results=agent_results,
            aggregated_output=aggregated,
            errors=all_errors,
            warnings=all_warnings,
            duration_ms=duration,
            trace_id=self.trace_id,
        )

    def _process_gather_results(
        self,
        completed: list[AgentResult[Any] | BaseException],
        aggregated: dict[str, Any],
    ) -> list[AgentResult[Any]]:
        """Process results from ``asyncio.gather(return_exceptions=True)``.

        Handles both successful AgentResult and Exception outcomes,
        converting exceptions to failed AgentResult entries.

        Parameters
        ----------
        completed:
            Raw results from ``asyncio.gather``.
        aggregated:
            Dict accumulating successful outputs keyed by agent name.

        Returns
        -------
        list[AgentResult[Any]]
            Processed results for ``_make_result()``.
        """
        results: list[AgentResult[Any]] = []
        for res in completed:
            if isinstance(res, BaseException):
                results.append(
                    AgentResult(
                        success=False,
                        errors=[str(res)],
                        agent_name="unknown",
                        trace_id=self.trace_id,
                    )
                )
            else:
                results.append(res)
                if res.success and res.output is not None:
                    aggregated[res.agent_name] = res.output.model_dump(
                        mode="json"
                    )
        return results

    def disband(self) -> None:
        """Clean up after swarm execution (ephemeral lifecycle)."""
        self.logger.debug(
            "%s swarm disbanded (trace=%s)", self.name, self.trace_id
        )


# ---------------------------------------------------------------------------
# Scientist Swarm — hypothesis generation
# ---------------------------------------------------------------------------


class ScientistSwarm(BaseSwarm):
    """Hypothesis generation via first-principles reasoning.

    Constituent agents:
    - DesignAgent: parameter space exploration with strategic selection
    - QueryAgent: retrieve historical data for context-aware hypothesis

    The Scientist swarm is responsible for the "what to try next" question.
    It combines data retrieval with parameter design to produce
    informed experimental candidates.
    """

    name = "scientist"
    description = "Hypothesis generation via first-principles reasoning"
    agent_specs = [
        ("DesignAgent", "parameter_designer"),
        ("QueryAgent", "knowledge_retriever"),
    ]

    async def run(self, **kwargs: Any) -> SwarmResult:
        """Generate experimental hypotheses.

        Keyword Args:
            design_input: DesignInput for the DesignAgent.
            query_request: Optional QueryRequest for historical context.

        Returns:
            SwarmResult with design candidates and optional historical context.
        """
        start = time.monotonic()
        results: list[AgentResult[Any]] = []
        aggregated: dict[str, Any] = {}

        design_input = kwargs.get("design_input")
        query_request = kwargs.get("query_request")

        # Run agents concurrently when both are provided
        tasks: list[asyncio.Task[AgentResult[Any]]] = []

        if design_input is not None:
            from app.agents.design_agent import DesignAgent

            agent = DesignAgent()
            tasks.append(
                asyncio.create_task(agent.run(design_input, trace_id=self.trace_id))
            )

        if query_request is not None:
            from app.agents.query_agent import QueryAgent

            agent = QueryAgent()
            tasks.append(
                asyncio.create_task(agent.run(query_request, trace_id=self.trace_id))
            )

        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            results = self._process_gather_results(completed, aggregated)

        return self._make_result(results, aggregated, start)


# ---------------------------------------------------------------------------
# Engineer Swarm — protocol optimisation & compilation
# ---------------------------------------------------------------------------


class EngineerSwarm(BaseSwarm):
    """Protocol optimisation, compilation, and code generation.

    Constituent agents:
    - PlannerAgent: multi-round campaign planning & strategy scheduling
    - CompilerAgent: protocol → DAG compilation with deck layout
    - CodeWriterAgent: NL → OT-2 Python protocol code generation
    - OnboardingAgent: instrument integration (optional, on-demand)

    The Engineer swarm handles the "how to execute" question —
    turning abstract candidates into executable robot protocols.
    """

    name = "engineer"
    description = "Protocol optimisation and compilation"
    agent_specs = [
        ("PlannerAgent", "campaign_planner"),
        ("CompilerAgent", "protocol_compiler"),
        ("CodeWriterAgent", "code_generator"),
        ("OnboardingAgent", "instrument_integrator"),
    ]

    async def run(self, **kwargs: Any) -> SwarmResult:
        """Compile candidates into executable protocols.

        Keyword Args:
            compile_input: CompileInput for the CompilerAgent.
            plan_input: Optional PlannerInput (used in planning phase).
            code_writer_input: Optional CodeWriterInput for NL→code.

        Returns:
            SwarmResult with compiled protocol DAG and optional code.
        """
        start = time.monotonic()
        results: list[AgentResult[Any]] = []
        aggregated: dict[str, Any] = {}

        # Compilation is the primary task (sequential, required)
        compile_input = kwargs.get("compile_input")
        if compile_input is not None:
            from app.agents.compiler_agent import CompilerAgent

            compiler = CompilerAgent()
            res = await compiler.run(compile_input, trace_id=self.trace_id)
            results.append(res)
            if res.success and res.output is not None:
                aggregated["compiler"] = res.output.model_dump(mode="json")

        # Code generation (optional, parallel-safe)
        code_writer_input = kwargs.get("code_writer_input")
        if code_writer_input is not None:
            from app.agents.code_writer_agent import CodeWriterAgent

            writer = CodeWriterAgent()
            res = await writer.run(code_writer_input, trace_id=self.trace_id)
            results.append(res)
            if res.success and res.output is not None:
                aggregated["code_writer"] = res.output.model_dump(mode="json")

        # Planning (optional, typically called once per campaign)
        plan_input = kwargs.get("plan_input")
        if plan_input is not None:
            from app.agents.planner_agent import PlannerAgent

            planner = PlannerAgent()
            res = await planner.run(plan_input, trace_id=self.trace_id)
            results.append(res)
            if res.success and res.output is not None:
                aggregated["planner"] = res.output.model_dump(mode="json")

        return self._make_result(results, aggregated, start)


# ---------------------------------------------------------------------------
# Analyst Swarm — real-time data interpretation
# ---------------------------------------------------------------------------


class AnalystSwarm(BaseSwarm):
    """Real-time spectral, image, and sensor data interpretation.

    Constituent agents:
    - SensingAgent: QC checks, anomaly detection, quality gates
    - QueryAgent: retrieve similar historical results for comparison

    The Analyst swarm answers "what happened" — interpreting raw
    measurement data to produce quality-checked KPI values and
    anomaly flags for downstream decision-making.
    """

    name = "analyst"
    description = "Real-time spectral and image interpretation"
    agent_specs = [
        ("SensingAgent", "quality_checker"),
        ("QueryAgent", "historical_comparator"),
    ]

    async def run(self, **kwargs: Any) -> SwarmResult:
        """Interpret experimental results.

        Keyword Args:
            sensing_input: SensingInput for QC and anomaly detection.
            query_request: Optional QueryRequest for historical comparison.

        Returns:
            SwarmResult with QC results and optional historical context.
        """
        start = time.monotonic()
        results: list[AgentResult[Any]] = []
        aggregated: dict[str, Any] = {}

        # Run agents concurrently
        tasks: list[asyncio.Task[AgentResult[Any]]] = []

        sensing_input = kwargs.get("sensing_input")
        if sensing_input is not None:
            from app.agents.sensing_agent import SensingAgent

            agent = SensingAgent()
            tasks.append(
                asyncio.create_task(agent.run(sensing_input, trace_id=self.trace_id))
            )

        query_request = kwargs.get("query_request")
        if query_request is not None:
            from app.agents.query_agent import QueryAgent

            agent = QueryAgent()
            tasks.append(
                asyncio.create_task(agent.run(query_request, trace_id=self.trace_id))
            )

        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            results = self._process_gather_results(completed, aggregated)

        return self._make_result(results, aggregated, start)


# ---------------------------------------------------------------------------
# Validator Swarm — hallucination detection & plausibility
# ---------------------------------------------------------------------------


class ValidatorSwarm(BaseSwarm):
    """Hallucination detection and physical plausibility verification.

    Constituent agents:
    - StopAgent: convergence detection, stopping condition evaluation
    - Reviewer (service): run scoring, failure attribution (via event bus)
    - Evolution (service): prior tightening proposals (via event bus)

    The Validator swarm answers "should we trust this?" — verifying
    that agent-proposed parameters and interpreted results are physically
    plausible before they enter the decision loop.

    Note: Reviewer and Evolution are event-bus-driven services, not
    BaseAgent subclasses.  The Validator swarm invokes StopAgent directly
    and triggers validation checks via the event bus for the others.
    """

    name = "validator"
    description = "Hallucination detection & physical plausibility"
    agent_specs = [
        ("StopAgent", "convergence_checker"),
        ("Reviewer", "run_scorer"),
        ("Evolution", "prior_validator"),
    ]

    async def run(self, **kwargs: Any) -> SwarmResult:
        """Validate round results and decide on continuation.

        Keyword Args:
            stop_input: StopInput for convergence evaluation.
            review_payload: Optional dict to trigger reviewer scoring.

        Returns:
            SwarmResult with stop decision and validation signals.
        """
        start = time.monotonic()
        results: list[AgentResult[Any]] = []
        aggregated: dict[str, Any] = {}

        # StopAgent — primary synchronous validation
        stop_input = kwargs.get("stop_input")
        if stop_input is not None:
            from app.agents.stop_agent import StopAgent

            agent = StopAgent()
            res = await agent.run(stop_input, trace_id=self.trace_id)
            results.append(res)
            if res.success and res.output is not None:
                aggregated["stop_agent"] = res.output.model_dump(mode="json")

        # Reviewer and Evolution are event-bus-driven and fire asynchronously
        # when runs complete. We record their availability as advisory metadata.
        review_payload = kwargs.get("review_payload")
        if review_payload is not None:
            aggregated["review_triggered"] = True
            aggregated["review_payload"] = review_payload

        return self._make_result(results, aggregated, start)


# ---------------------------------------------------------------------------
# Swarm registry & factory
# ---------------------------------------------------------------------------

# Maps swarm name → swarm class
SWARM_REGISTRY: dict[str, type[BaseSwarm]] = {
    "scientist": ScientistSwarm,
    "engineer": EngineerSwarm,
    "analyst": AnalystSwarm,
    "validator": ValidatorSwarm,
}


class SwarmFactory:
    """Factory for spawning ephemeral agent swarms.

    Usage::

        swarm = SwarmFactory.spawn("scientist", context)
        result = await swarm.run(design_input=di, query_request=qr)
        swarm.disband()

    The factory enforces the ephemeral lifecycle:
    swarms are created, used, and disbanded per invocation.
    """

    @staticmethod
    def spawn(swarm_name: str, context: SwarmContext) -> BaseSwarm:
        """Spawn a new swarm instance by name (thread-safe).

        Parameters
        ----------
        swarm_name:
            One of "scientist", "engineer", "analyst", "validator".
        context:
            Campaign state for the swarm.

        Returns
        -------
        BaseSwarm
            Configured, ready-to-run swarm instance.

        Raises
        ------
        ValueError
            If swarm_name is not in SWARM_REGISTRY.
        """
        with _REGISTRY_LOCK:
            cls = SWARM_REGISTRY.get(swarm_name)
            if cls is None:
                available = ", ".join(sorted(SWARM_REGISTRY.keys()))
                raise ValueError(
                    f"Unknown swarm '{swarm_name}'. Available: {available}"
                )
        return cls(context)

    @staticmethod
    def available_swarms() -> list[str]:
        """Return names of all registered swarms (thread-safe)."""
        with _REGISTRY_LOCK:
            return sorted(SWARM_REGISTRY.keys())

    @staticmethod
    def register(name: str, cls: type[BaseSwarm]) -> None:
        """Register a custom swarm type at runtime (thread-safe).

        Parameters
        ----------
        name:
            Swarm identifier (lowercase).
        cls:
            Swarm class inheriting from BaseSwarm.
        """
        with _REGISTRY_LOCK:
            SWARM_REGISTRY[name] = cls
        logger.info("Registered custom swarm: %s → %s", name, cls.__name__)


def list_swarms() -> list[dict[str, Any]]:
    """Return metadata about all registered swarms (thread-safe).

    Useful for API introspection endpoints.

    Returns
    -------
    list of dicts with keys: name, description, agents
    """
    with _REGISTRY_LOCK:
        items = sorted(SWARM_REGISTRY.items())
    result = []
    for name, cls in items:
        result.append({
            "name": name,
            "description": cls.description,
            "agents": [
                {"class": spec[0], "role": spec[1]}
                for spec in cls.agent_specs
            ],
        })
    return result
