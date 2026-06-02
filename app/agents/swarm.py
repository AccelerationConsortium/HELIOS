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
from collections import Counter
from collections.abc import Callable
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

    def __init__(self, context: SwarmContext, control_plane: Any | None = None) -> None:
        self.context: SwarmContext = context
        self.control_plane: Any | None = control_plane or context.extra.get("control_plane")
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

    async def _call_agent(
        self,
        target: str,
        input_data: BaseModel,
        fallback_factory: Callable[[], BaseAgent],
    ) -> AgentResult[Any]:
        """Invoke an agent through ControlPlane when available."""
        if self.control_plane is not None:
            call = getattr(self.control_plane, "call", None)
            if callable(call):
                return await call(
                    target,
                    input_data,
                    caller=f"swarm.{self.name}",
                    trace_id=self.trace_id,
                )
        agent = fallback_factory()
        return await agent.run(input_data, trace_id=self.trace_id)

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

            tasks.append(
                asyncio.create_task(
                    self._call_agent(
                        "design_agent",
                        design_input,
                        DesignAgent,
                    )
                )
            )

        if query_request is not None:
            from app.agents.query_agent import QueryAgent

            tasks.append(
                asyncio.create_task(
                    self._call_agent(
                        "query_agent",
                        query_request,
                        QueryAgent,
                    )
                )
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

            res = await self._call_agent(
                "compiler_agent",
                compile_input,
                CompilerAgent,
            )
            results.append(res)
            if res.success and res.output is not None:
                aggregated["compiler"] = res.output.model_dump(mode="json")

        # Code generation (optional, parallel-safe)
        code_writer_input = kwargs.get("code_writer_input")
        if code_writer_input is not None:
            from app.agents.code_writer_agent import CodeWriterAgent

            res = await self._call_agent(
                "code_writer_agent",
                code_writer_input,
                CodeWriterAgent,
            )
            results.append(res)
            if res.success and res.output is not None:
                aggregated["code_writer"] = res.output.model_dump(mode="json")

        # Planning (optional, typically called once per campaign)
        plan_input = kwargs.get("plan_input")
        if plan_input is not None:
            from app.agents.planner_agent import PlannerAgent

            res = await self._call_agent(
                "planner_agent",
                plan_input,
                PlannerAgent,
            )
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

            tasks.append(
                asyncio.create_task(
                    self._call_agent(
                        "sensing_agent",
                        sensing_input,
                        SensingAgent,
                    )
                )
            )

        query_request = kwargs.get("query_request")
        if query_request is not None:
            from app.agents.query_agent import QueryAgent

            tasks.append(
                asyncio.create_task(
                    self._call_agent(
                        "query_agent",
                        query_request,
                        QueryAgent,
                    )
                )
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

            res = await self._call_agent(
                "stop_agent",
                stop_input,
                StopAgent,
            )
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
    def spawn(
        swarm_name: str,
        context: SwarmContext,
        control_plane: Any | None = None,
    ) -> BaseSwarm:
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
        return cls(context, control_plane=control_plane)

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


# ===========================================================================
# Advanced multi-agent coordination patterns
# ===========================================================================
#
# The patterns below extend the swarm model with three additional
# coordination strategies that operate over a ``ControlPlane`` — a runtime
# registry of live agent instances exposing capability metadata and
# parallel/serial invocation primitives. Unlike the swarm classes above
# (which statically wire constituent agents), these patterns assemble or
# select agents dynamically at invocation time.
#
# Expected ControlPlane protocol (duck-typed, ``Any`` for loose coupling):
#   - ``control_plane.agents``: dict[str, agent] of registered agents
#   - ``agent.has_capability(tag: str) -> bool``: capability predicate
#   - ``await control_plane.call(name, input_data, *, caller, trace_id)``
#         -> AgentResult
#   - ``await control_plane.call_parallel(names, input_data, *, caller,
#         trace_id) -> list[AgentResult]``
#
# These are additive: existing swarms, the registry, and the factory above
# remain unchanged.
# ---------------------------------------------------------------------------


# Judge function signature for RacingSwarm: collapse competing results
# into a single winning AgentResult.
JudgeFn = Callable[[list[AgentResult]], AgentResult]


def _extract_confidence(result: AgentResult, default: float = 0.5) -> float:
    """Best-effort extraction of a confidence score from an AgentResult.

    Looks for ``output.confidence`` and falls back to ``default`` when the
    output is missing, has no confidence attribute, or the value is not a
    finite number.
    """
    output = getattr(result, "output", None)
    if output is None:
        return default
    conf = getattr(output, "confidence", None)
    if conf is None:
        return default
    try:
        return float(conf)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# A. DynamicSwarm — capability-based assembly at spawn time
# ---------------------------------------------------------------------------


class DynamicSwarm:
    """Capability-based dynamic swarm assembled at spawn time.

    Rather than statically wiring a fixed set of agents, a DynamicSwarm
    queries a ControlPlane for all live agents advertising any of the
    requested capability tags, then invokes the matched set in parallel.

    This supports open-ended task routing: a caller specifies *what
    capabilities the task needs* (e.g. ``["spectral", "anomaly"]``) and the
    swarm self-assembles from whatever agents currently provide them.

    Usage::

        swarm = DynamicSwarm.for_task(
            ["spectral", "anomaly"], context, control_plane
        )
        result = await swarm.run(input_data=sensing_input)
    """

    name: ClassVar[str] = "dynamic"
    description: ClassVar[str] = "Capability-based dynamic swarm"

    def __init__(
        self,
        context: SwarmContext,
        control_plane: Any,
        required_tags: list[str],
    ) -> None:
        self.context: SwarmContext = context
        self.control_plane: Any = control_plane
        self.required_tags: list[str] = list(required_tags)
        self.trace_id: str = uuid.uuid4().hex[:12]
        self.logger: Logger = logging.getLogger(f"swarm.{self.name}")
        self.selected_agents: list[str] = self._select_agents()
        self.logger.info(
            "Spawned dynamic swarm (trace=%s, campaign=%s, round=%d, "
            "tags=%s, selected=%s)",
            self.trace_id,
            context.campaign_id,
            context.round_number,
            self.required_tags,
            self.selected_agents,
        )

    def _select_agents(self) -> list[str]:
        """Resolve agent names whose capabilities match any required tag."""
        selected: list[str] = []
        agents = getattr(self.control_plane, "agents", None) or {}
        for agent_name, agent in agents.items():
            has_cap = getattr(agent, "has_capability", None)
            if not callable(has_cap):
                continue
            for tag in self.required_tags:
                try:
                    if has_cap(tag):
                        selected.append(agent_name)
                        break
                except Exception:  # noqa: BLE001 - defensive: bad predicate
                    self.logger.warning(
                        "has_capability(%r) raised for agent %s; skipping tag",
                        tag,
                        agent_name,
                    )
        return selected

    @classmethod
    def for_task(
        cls,
        required_tags: list[str],
        context: SwarmContext,
        control_plane: Any,
    ) -> DynamicSwarm:
        """Build a swarm from control_plane agents matching any required_tag.

        Parameters
        ----------
        required_tags:
            Capability tags the task requires. An agent is selected if it
            advertises *any* one of these via ``has_capability``.
        context:
            Campaign state for the swarm.
        control_plane:
            Runtime agent registry exposing ``agents`` and
            ``call_parallel``.

        Returns
        -------
        DynamicSwarm
            A swarm pre-resolved to the matching agent set.
        """
        return cls(context, control_plane, required_tags)

    async def run(self, **kwargs: Any) -> SwarmResult:
        """Invoke all selected agents via ``call_parallel`` and aggregate.

        Keyword Args:
            input_data: Input payload forwarded to every selected agent.

        Returns:
            SwarmResult aggregating the matched agents' outputs. If no agents
            matched, returns a graceful warning result (success=True, no
            errors) so callers can degrade rather than fail hard.
        """
        start = time.monotonic()
        aggregated: dict[str, Any] = {
            "required_tags": list(self.required_tags),
            "selected_agents": list(self.selected_agents),
        }

        # Graceful fallback: nothing matched the requested capabilities.
        if not self.selected_agents:
            msg = (
                "No agents matched required capability tags "
                f"{self.required_tags}; dynamic swarm assembled empty."
            )
            self.logger.warning(msg)
            duration = (time.monotonic() - start) * 1000
            return SwarmResult(
                swarm_name=self.name,
                success=True,
                agent_results=[],
                aggregated_output=aggregated,
                errors=[],
                warnings=[msg],
                duration_ms=duration,
                trace_id=self.trace_id,
            )

        input_data = kwargs.get("input_data")
        results = await self.control_plane.call_parallel(
            [(agent_name, input_data) for agent_name in self.selected_agents],
            caller=self.name,
            trace_id=self.trace_id,
        )

        return self._make_result(results, aggregated, start)

    def _make_result(
        self,
        agent_results: list[AgentResult],
        aggregated: dict[str, Any],
        start_time: float,
    ) -> SwarmResult:
        """Build a SwarmResult from control-plane invocation results."""
        all_errors: list[str] = []
        all_warnings: list[str] = []
        all_success = True

        for ar in agent_results:
            all_errors.extend(getattr(ar, "errors", []) or [])
            all_warnings.extend(getattr(ar, "warnings", []) or [])
            if not getattr(ar, "success", False):
                all_success = False
            output = getattr(ar, "output", None)
            if getattr(ar, "success", False) and output is not None:
                aggregated[getattr(ar, "agent_name", "unknown")] = (
                    output.model_dump(mode="json")
                    if hasattr(output, "model_dump")
                    else output
                )

        duration = (time.monotonic() - start_time) * 1000
        self.logger.info(
            "dynamic swarm completed in %.1fms (success=%s, agents=%d, "
            "trace=%s)",
            duration,
            all_success,
            len(agent_results),
            self.trace_id,
        )
        return SwarmResult(
            swarm_name=self.name,
            success=all_success,
            agent_results=list(agent_results),
            aggregated_output=aggregated,
            errors=all_errors,
            warnings=all_warnings,
            duration_ms=duration,
            trace_id=self.trace_id,
        )

    def disband(self) -> None:
        """Clean up after swarm execution (ephemeral lifecycle)."""
        self.logger.debug(
            "dynamic swarm disbanded (trace=%s)", self.trace_id
        )


# ---------------------------------------------------------------------------
# B. RacingSwarm — speculative parallel execution with judge selection
# ---------------------------------------------------------------------------


class RacingSwarm:
    """Speculative parallel execution: N agents race, best result wins.

    All competing agents receive the same input and run concurrently. A
    *judge* function collapses the competing results into a single winner.
    This trades compute for latency/quality: useful when several agents
    can attempt the same task with differing approaches and the best
    output should be selected post-hoc.

    Usage::

        racer = RacingSwarm(context, control_plane)
        winner = await racer.race(
            ["design_a", "design_b"], design_input,
            judge=RacingSwarm.highest_confidence_judge,
        )
    """

    name: ClassVar[str] = "racing"
    description: ClassVar[str] = "Speculative parallel execution"

    def __init__(self, context: SwarmContext, control_plane: Any) -> None:
        self.context: SwarmContext = context
        self.control_plane: Any = control_plane
        self.trace_id: str = uuid.uuid4().hex[:12]
        self.logger: Logger = logging.getLogger(f"swarm.{self.name}")
        self.logger.info(
            "Spawned racing swarm (trace=%s, campaign=%s, round=%d)",
            self.trace_id,
            context.campaign_id,
            context.round_number,
        )

    async def race(
        self,
        agent_names: list[str],
        input_data: BaseModel,
        *,
        judge: JudgeFn | None = None,
        caller: str = "racing_swarm",
    ) -> AgentResult:
        """Run all agents in parallel, apply judge, return the winner.

        Parameters
        ----------
        agent_names:
            Competing agents; all receive ``input_data``.
        input_data:
            Shared input payload.
        judge:
            Selection function over the (successful) results. Defaults to
            :meth:`highest_confidence_judge`.
        caller:
            Caller identity for control-plane provenance.

        Returns
        -------
        AgentResult
            The winning result. If every competitor failed, the first
            failure is returned so the caller still sees an error trail.
        """
        judge = judge or RacingSwarm.highest_confidence_judge
        self.logger.info(
            "Racing %d agents %s (trace=%s, judge=%s)",
            len(agent_names),
            agent_names,
            self.trace_id,
            getattr(judge, "__name__", repr(judge)),
        )

        results: list[AgentResult] = await self.control_plane.call_parallel(
            [(agent_name, input_data) for agent_name in agent_names],
            caller=caller,
            trace_id=self.trace_id,
        )

        successful = [r for r in results if getattr(r, "success", False)]

        if not successful:
            self.logger.warning(
                "All %d racing agents failed (trace=%s); returning first "
                "failure",
                len(results),
                self.trace_id,
            )
            if results:
                return results[0]
            # No competitors at all — synthesize an explicit failure.
            return AgentResult(
                success=False,
                errors=["RacingSwarm.race called with no agents"],
                agent_name=self.name,
                trace_id=self.trace_id,
            )

        winner = judge(successful)
        self.logger.info(
            "Racing winner: %s (trace=%s, candidates=%d)",
            getattr(winner, "agent_name", "unknown"),
            self.trace_id,
            len(successful),
        )
        return winner

    @staticmethod
    def highest_confidence_judge(results: list[AgentResult]) -> AgentResult:
        """Select the result with the highest ``output.confidence``.

        Missing or unparseable confidence values default to ``0.5``.
        """
        return max(results, key=_extract_confidence)

    @staticmethod
    def majority_vote_judge(results: list[AgentResult]) -> AgentResult:
        """For boolean/categorical outputs: pick the most common output.

        Each result's ``output`` is reduced to a hashable vote key (a
        ``decision``/``value``/``label`` attribute if present, else the
        serialized output, else its repr). The result backing the most
        common key is returned; ties resolve to the first such result, and
        within that group the highest-confidence member is preferred.
        """
        if not results:
            raise ValueError("majority_vote_judge requires at least one result")

        def vote_key(r: AgentResult) -> Any:
            output = getattr(r, "output", None)
            if output is None:
                return None
            for attr in ("decision", "value", "label", "category", "verdict"):
                if hasattr(output, attr):
                    return getattr(output, attr)
            if hasattr(output, "model_dump"):
                try:
                    return repr(sorted(output.model_dump(mode="json").items()))
                except Exception:  # noqa: BLE001 - non-dict / unsortable
                    return repr(output.model_dump(mode="json"))
            return repr(output)

        keyed = [(vote_key(r), r) for r in results]
        counts = Counter(k for k, _ in keyed)
        winning_key, _ = counts.most_common(1)[0]
        group = [r for k, r in keyed if k == winning_key]
        return max(group, key=_extract_confidence)


def build_racing_swarm(
    context: SwarmContext, control_plane: Any
) -> RacingSwarm:
    """Construct a :class:`RacingSwarm` bound to a control plane."""
    return RacingSwarm(context, control_plane)


# ---------------------------------------------------------------------------
# C. AdversarialHypothesisLoop — Scientist↔Validator structured debate
# ---------------------------------------------------------------------------


@dataclass
class DebateRound:
    """A single round of the adversarial hypothesis debate.

    Attributes:
        round_idx: Zero-based index of this debate round.
        hypothesis: The hypothesis proposed by the scientist this round.
        refutations: Objections raised by the validator (empty = accepted).
        confidence: Validator's confidence in the hypothesis [0.0, 1.0].
    """

    round_idx: int
    hypothesis: str
    refutations: list[str]
    confidence: float


class AdversarialHypothesisLoop:
    """Scientist-Validator adversarial debate loop.

    Implements a structured debate protocol that hardens a hypothesis
    against hallucination and physical implausibility before it enters the
    decision loop:

    1. A *scientist* agent proposes a hypothesis from the initial input.
    2. A *validator* agent runs in adversarial mode and returns refutations
       plus a confidence score.
    3. If there are no refutations OR confidence meets the consensus
       threshold, the debate resolves.
    4. Otherwise the scientist revises, given the refutations as context,
       and the loop repeats up to ``MAX_ROUNDS``.
    5. If unresolved after ``MAX_ROUNDS``, the loop escalates to a human and
       returns the last hypothesis with reduced confidence.

    The full debate history is returned for audit and provenance.
    """

    name: ClassVar[str] = "adversarial_hypothesis"
    description: ClassVar[str] = "Scientist-Validator adversarial debate"

    MAX_ROUNDS: ClassVar[int] = 3
    CONSENSUS_THRESHOLD: ClassVar[float] = 0.75

    def __init__(self, context: SwarmContext, control_plane: Any) -> None:
        self.context: SwarmContext = context
        self.control_plane: Any = control_plane
        self.trace_id: str = uuid.uuid4().hex[:12]
        self.logger: Logger = logging.getLogger(f"swarm.{self.name}")
        self.logger.info(
            "Spawned adversarial hypothesis loop (trace=%s, campaign=%s, "
            "round=%d, max_rounds=%d, threshold=%.2f)",
            self.trace_id,
            context.campaign_id,
            context.round_number,
            self.MAX_ROUNDS,
            self.CONSENSUS_THRESHOLD,
        )

    @staticmethod
    def _inject_refutations(
        input_data: BaseModel, refutations: list[str], round_idx: int
    ) -> Any:
        """Attach refutation context to the scientist's next input.

        Prefers an in-place ``model_copy(update=...)`` if the model declares
        a ``refutations`` field; otherwise wraps the input alongside the
        refutation context so the scientist can still consume it.
        """
        fields = getattr(type(input_data), "model_fields", {}) or {}
        if "refutations" in fields:
            return input_data.model_copy(update={"refutations": refutations})
        # Fallback wrapper: preserve original input + debate context.
        return {
            "input": input_data,
            "refutations": refutations,
            "debate_round": round_idx,
        }

    @staticmethod
    def _extract_hypothesis(result: AgentResult, fallback: str) -> str:
        """Pull a hypothesis string out of a scientist AgentResult."""
        output = getattr(result, "output", None)
        if output is None:
            return fallback
        for attr in ("hypothesis", "summary", "rationale", "description"):
            val = getattr(output, attr, None)
            if isinstance(val, str) and val:
                return val
        if hasattr(output, "model_dump"):
            try:
                return repr(output.model_dump(mode="json"))
            except Exception:  # noqa: BLE001
                return fallback
        return fallback

    @staticmethod
    def _extract_refutations(result: AgentResult) -> list[str]:
        """Pull a list of refutation strings out of a validator result."""
        output = getattr(result, "output", None)
        if output is None:
            return []
        for attr in ("refutations", "objections", "issues", "violations"):
            val = getattr(output, attr, None)
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
        return []

    async def run(
        self,
        *,
        scientist_agent: str = "design_agent",
        validator_agent: str = "validation_agent",
        initial_input: BaseModel,
        trace_id: str | None = None,
    ) -> tuple[str, float, list[DebateRound]]:
        """Run the adversarial debate to convergence or escalation.

        Parameters
        ----------
        scientist_agent:
            Control-plane name of the proposing agent.
        validator_agent:
            Control-plane name of the adversarial validating agent.
        initial_input:
            Seed input for the scientist's first hypothesis.
        trace_id:
            Optional externally-supplied trace id; overrides the internal one
            so the debate can be correlated with a parent operation.

        Returns
        -------
        tuple[str, float, list[DebateRound]]
            ``(winning_hypothesis, final_confidence, debate_history)``.
        """
        if trace_id:
            self.trace_id = trace_id

        history: list[DebateRound] = []
        current_input: Any = initial_input
        hypothesis = ""
        confidence = 0.0

        for round_idx in range(self.MAX_ROUNDS):
            self.logger.info(
                "Debate round %d/%d begins (trace=%s, scientist=%s, "
                "validator=%s)",
                round_idx + 1,
                self.MAX_ROUNDS,
                self.trace_id,
                scientist_agent,
                validator_agent,
            )

            # --- 1. Scientist proposes / revises -------------------------
            sci_result = await self.control_plane.call(
                scientist_agent,
                current_input,
                caller=self.name,
                trace_id=self.trace_id,
            )
            hypothesis = self._extract_hypothesis(sci_result, fallback=hypothesis)
            self.logger.info(
                "Round %d: scientist proposed hypothesis (trace=%s, "
                "success=%s): %.120s",
                round_idx + 1,
                self.trace_id,
                getattr(sci_result, "success", False),
                hypothesis,
            )

            # --- 2. Validator refutes (adversarial mode) -----------------
            val_input = self._inject_refutations(
                initial_input, [], round_idx
            )
            # Carry the proposed hypothesis to the validator when possible.
            if isinstance(val_input, dict):
                val_input["hypothesis"] = hypothesis
                val_input["mode"] = "adversarial"
            elif "hypothesis" in (
                getattr(type(val_input), "model_fields", {}) or {}
            ):
                val_input = val_input.model_copy(
                    update={"hypothesis": hypothesis}
                )

            val_result = await self.control_plane.call(
                validator_agent,
                val_input,
                caller=self.name,
                trace_id=self.trace_id,
            )
            refutations = self._extract_refutations(val_result)
            confidence = _extract_confidence(val_result, default=0.5)

            history.append(
                DebateRound(
                    round_idx=round_idx,
                    hypothesis=hypothesis,
                    refutations=refutations,
                    confidence=confidence,
                )
            )
            self.logger.info(
                "Round %d: validator returned %d refutation(s), "
                "confidence=%.3f (trace=%s)",
                round_idx + 1,
                len(refutations),
                confidence,
                self.trace_id,
            )

            # --- 3. Consensus check --------------------------------------
            if not refutations or confidence >= self.CONSENSUS_THRESHOLD:
                self.logger.info(
                    "Debate resolved at round %d (trace=%s, confidence=%.3f, "
                    "refutations=%d)",
                    round_idx + 1,
                    self.trace_id,
                    confidence,
                    len(refutations),
                )
                return hypothesis, confidence, history

            # --- 4. Scientist revises with refutation context ------------
            current_input = self._inject_refutations(
                initial_input, refutations, round_idx + 1
            )

        # --- 5. Unresolved after MAX_ROUNDS → escalate to human ----------
        self.logger.warning(
            "Debate unresolved after %d rounds (trace=%s); escalating to "
            "human",
            self.MAX_ROUNDS,
            self.trace_id,
        )
        esc_hypothesis, esc_confidence = await self._escalate_to_human(history)
        return esc_hypothesis, esc_confidence, history

    async def _escalate_to_human(
        self, history: list[DebateRound]
    ) -> tuple[str, float]:
        """Handle MAX_ROUNDS exhaustion via human escalation.

        Returns the last proposed hypothesis with a deliberately conservative
        confidence (0.4) signalling that automated consensus failed and human
        review is required.
        """
        last_hypothesis = history[-1].hypothesis if history else ""
        self.logger.warning(
            "Human escalation triggered (trace=%s, rounds=%d, "
            "last_confidence=%.3f)",
            self.trace_id,
            len(history),
            history[-1].confidence if history else 0.0,
        )
        return last_hypothesis, 0.4

    def disband(self) -> None:
        """Clean up after the debate loop (ephemeral lifecycle)."""
        self.logger.debug(
            "adversarial hypothesis loop disbanded (trace=%s)", self.trace_id
        )


def build_adversarial_loop(
    context: SwarmContext, control_plane: Any
) -> AdversarialHypothesisLoop:
    """Construct an :class:`AdversarialHypothesisLoop` bound to a control plane."""
    return AdversarialHypothesisLoop(context, control_plane)
