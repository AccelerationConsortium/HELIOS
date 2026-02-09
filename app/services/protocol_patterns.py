"""Protocol Pattern Library for structured experiment protocol definitions.

Provides reusable, composable protocol patterns for OER catalyst optimization
and other electrochemistry experiments.  Each pattern declares steps, params,
optimizability flags and safety locks -- bridging the gap between the optimizer
(candidate_gen.ParameterSpace) and the compiler (protocol JSON).

Zero LLM in the critical path.  Pure Python stdlib.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services.candidate_gen import ParameterSpace, SearchDimension

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternParam:
    """A single tuneable or fixed parameter within a protocol step."""

    name: str
    param_type: str  # "number" | "integer" | "categorical"
    min_value: float | None = None
    max_value: float | None = None
    default: Any = None
    unit: str = ""
    optimizable: bool = True
    safety_locked: bool = False
    log_scale: bool = False
    choices: tuple[Any, ...] | None = None
    description: str = ""


@dataclass(frozen=True)
class PatternStep:
    """A single step in a protocol pattern."""

    name: str
    primitive: str  # OTbot primitive e.g. "heat", "robot.aspirate"
    params: tuple[PatternParam, ...]
    order: int
    description: str = ""


@dataclass(frozen=True)
class ProtocolPattern:
    """A complete, reusable protocol pattern with typed parameters.

    Bridges between high-level experiment definitions and the low-level
    protocol JSON consumed by the compiler + executor.
    """

    id: str
    name: str
    domain: str  # e.g. "oer", "corrosion", "battery"
    description: str
    steps: tuple[PatternStep, ...]
    version: str = "1.0"
    tags: tuple[str, ...] = ()

    # -- query helpers -----------------------------------------------------

    def get_optimizable_params(self) -> list[PatternParam]:
        """Return all params the optimizer is allowed to tune."""
        result: list[PatternParam] = []
        for step in self.steps:
            for p in step.params:
                if p.optimizable and not p.safety_locked:
                    result.append(p)
        return result

    def get_safety_locked_params(self) -> list[PatternParam]:
        """Return all params that must not be changed by any optimizer."""
        result: list[PatternParam] = []
        for step in self.steps:
            for p in step.params:
                if p.safety_locked:
                    result.append(p)
        return result

    # -- conversion --------------------------------------------------------

    def to_parameter_space(self) -> ParameterSpace:
        """Convert optimizable params into a candidate_gen.ParameterSpace.

        Only params that are ``optimizable=True`` *and* ``safety_locked=False``
        are included as search dimensions.  The protocol_template is built
        from the pattern defaults so the sampler has a valid base protocol.
        """
        dims: list[SearchDimension] = []
        for step in self.steps:
            for p in step.params:
                if not p.optimizable or p.safety_locked:
                    continue
                dims.append(
                    SearchDimension(
                        param_name=p.name,
                        param_type=p.param_type,
                        min_value=p.min_value,
                        max_value=p.max_value,
                        log_scale=p.log_scale,
                        choices=p.choices,
                        step_key=step.name,
                        primitive=step.primitive,
                    )
                )

        template = self.to_protocol_json({})
        return ParameterSpace(dimensions=tuple(dims), protocol_template=template)

    def to_protocol_json(self, params: dict[str, Any]) -> dict[str, Any]:
        """Generate a compiler-compatible protocol JSON.

        *params* overrides defaults for any matching param name.  Safety-locked
        params are always forced to their declared default.
        """
        json_steps: list[dict[str, Any]] = []
        for step in sorted(self.steps, key=lambda s: s.order):
            step_params: dict[str, Any] = {}
            for p in step.params:
                if p.safety_locked:
                    step_params[p.name] = p.default
                elif p.name in params:
                    step_params[p.name] = params[p.name]
                else:
                    step_params[p.name] = p.default

            prev_key = json_steps[-1]["step_key"] if json_steps else None
            json_steps.append(
                {
                    "step_key": step.name,
                    "primitive": step.primitive,
                    "params": step_params,
                    "depends_on": [prev_key] if prev_key else [],
                    "resources": [],
                }
            )

        return {
            "metadata": {
                "pattern_id": self.id,
                "pattern_name": self.name,
                "domain": self.domain,
                "version": self.version,
                "tags": list(self.tags),
            },
            "steps": json_steps,
        }

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate a param dict against this pattern.

        Returns a list of human-readable error strings (empty = valid).
        """
        errors: list[str] = []
        known: dict[str, PatternParam] = {}
        for step in self.steps:
            for p in step.params:
                known[p.name] = p

        for name, value in params.items():
            if name not in known:
                errors.append(f"unknown param '{name}'")
                continue
            p = known[name]

            if p.safety_locked:
                if value != p.default:
                    errors.append(
                        f"param '{name}' is safety-locked at {p.default!r}, "
                        f"got {value!r}"
                    )
                continue

            if p.param_type == "categorical":
                if p.choices and value not in p.choices:
                    errors.append(
                        f"param '{name}': value {value!r} not in "
                        f"choices {p.choices!r}"
                    )
            elif p.param_type in ("number", "integer"):
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    errors.append(
                        f"param '{name}': expected numeric, got {type(value).__name__}"
                    )
                    continue
                if p.min_value is not None and num < p.min_value:
                    errors.append(
                        f"param '{name}': {num} below min {p.min_value}"
                    )
                if p.max_value is not None and num > p.max_value:
                    errors.append(
                        f"param '{name}': {num} above max {p.max_value}"
                    )
                if p.param_type == "integer" and value != int(value):
                    errors.append(
                        f"param '{name}': expected integer, got {value!r}"
                    )

        return errors


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

_PATTERN_REGISTRY: dict[str, ProtocolPattern] = {}


def register_pattern(pattern: ProtocolPattern) -> None:
    """Register a pattern, making it available by id."""
    if pattern.id in _PATTERN_REGISTRY:
        logger.warning("overwriting pattern '%s'", pattern.id)
    _PATTERN_REGISTRY[pattern.id] = pattern
    logger.info("registered pattern '%s' (domain=%s)", pattern.id, pattern.domain)


def get_pattern(pattern_id: str) -> ProtocolPattern | None:
    """Look up a pattern by id.  Returns ``None`` if not found."""
    return _PATTERN_REGISTRY.get(pattern_id)


def list_patterns(domain: str | None = None) -> list[ProtocolPattern]:
    """List registered patterns, optionally filtered by domain."""
    patterns = list(_PATTERN_REGISTRY.values())
    if domain is not None:
        patterns = [p for p in patterns if p.domain == domain]
    return patterns


# ---------------------------------------------------------------------------
# Built-in OER screening pattern
# ---------------------------------------------------------------------------

_OER_SYNTHESIS_PARAMS = (
    PatternParam(
        name="precursor_ratio",
        param_type="number",
        min_value=0.1,
        max_value=10.0,
        default=1.0,
        unit="ratio",
        optimizable=True,
        description="Molar ratio of metal precursors (e.g. Ni:Fe)",
    ),
    PatternParam(
        name="solvent_volume_ul",
        param_type="number",
        min_value=50.0,
        max_value=500.0,
        default=200.0,
        unit="ul",
        optimizable=True,
        description="Total solvent volume for ink preparation",
    ),
    PatternParam(
        name="mixing_temp_c",
        param_type="number",
        min_value=20.0,
        max_value=80.0,
        default=25.0,
        unit="celsius",
        optimizable=True,
        description="Temperature during precursor mixing",
    ),
    PatternParam(
        name="mixing_time_s",
        param_type="number",
        min_value=60.0,
        max_value=3600.0,
        default=600.0,
        unit="seconds",
        optimizable=True,
        description="Duration of precursor mixing / sonication",
    ),
)

_OER_DEPOSITION_PARAMS = (
    PatternParam(
        name="deposition_volume_ul",
        param_type="number",
        min_value=5.0,
        max_value=100.0,
        default=20.0,
        unit="ul",
        optimizable=True,
        description="Volume of catalyst ink deposited on substrate",
    ),
    PatternParam(
        name="spin_speed_rpm",
        param_type="integer",
        min_value=500.0,
        max_value=5000.0,
        default=2000,
        unit="rpm",
        optimizable=True,
        description="Spin-coater rotation speed for uniform film",
    ),
)

_OER_ANNEALING_PARAMS = (
    PatternParam(
        name="annealing_temp_c",
        param_type="number",
        min_value=100.0,
        max_value=600.0,
        default=350.0,
        unit="celsius",
        optimizable=True,
        description="Annealing temperature for oxide formation",
    ),
    PatternParam(
        name="annealing_duration_s",
        param_type="number",
        min_value=300.0,
        max_value=7200.0,
        default=1800.0,
        unit="seconds",
        optimizable=True,
        description="Annealing hold time at target temperature",
    ),
    PatternParam(
        name="max_temp_c",
        param_type="number",
        min_value=700.0,
        max_value=700.0,
        default=700.0,
        unit="celsius",
        optimizable=False,
        safety_locked=True,
        description="Absolute maximum furnace temperature -- safety limit",
    ),
)

_OER_ELECTROCHEM_PARAMS = (
    PatternParam(
        name="scan_rate_mv_s",
        param_type="number",
        min_value=1.0,
        max_value=100.0,
        default=10.0,
        unit="mV/s",
        optimizable=True,
        description="Potential sweep rate for LSV / CV measurement",
    ),
    PatternParam(
        name="potential_range_v",
        param_type="number",
        min_value=0.0,
        max_value=2.0,
        default=2.0,
        unit="V",
        optimizable=False,
        safety_locked=True,
        description="Maximum anodic potential -- safety limit for electrolyte window",
    ),
    PatternParam(
        name="cycles",
        param_type="integer",
        min_value=1.0,
        max_value=100.0,
        default=10,
        unit="count",
        optimizable=True,
        description="Number of CV cycles for activation / measurement",
    ),
    PatternParam(
        name="electrolyte_ph",
        param_type="number",
        min_value=0.0,
        max_value=14.0,
        default=13.0,
        unit="pH",
        optimizable=True,
        description="Electrolyte pH (13 = typical 1 M KOH for OER)",
    ),
)


OER_SCREENING = ProtocolPattern(
    id="oer_screening",
    name="OER Catalyst Screening",
    domain="oer",
    description=(
        "Four-step OER catalyst screening workflow: ink synthesis, "
        "thin-film deposition, thermal annealing, and electrochemical "
        "characterisation via linear sweep voltammetry."
    ),
    steps=(
        PatternStep(
            name="synthesis",
            primitive="robot.aspirate",
            params=_OER_SYNTHESIS_PARAMS,
            order=1,
            description="Prepare catalyst ink from metal-salt precursors",
        ),
        PatternStep(
            name="deposition",
            primitive="robot.dispense",
            params=_OER_DEPOSITION_PARAMS,
            order=2,
            description="Deposit catalyst ink onto substrate via spin-coating",
        ),
        PatternStep(
            name="annealing",
            primitive="heat",
            params=_OER_ANNEALING_PARAMS,
            order=3,
            description="Anneal deposited film to form metal-oxide phase",
        ),
        PatternStep(
            name="electrochem_test",
            primitive="squidstat.run_experiment",
            params=_OER_ELECTROCHEM_PARAMS,
            order=4,
            description="Run electrochemical OER characterisation (LSV/CV)",
        ),
    ),
    version="1.0",
    tags=("oer", "screening", "electrocatalysis", "high-throughput"),
)

# Auto-register built-in pattern at import time
register_pattern(OER_SCREENING)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_protocol_from_pattern(
    pattern_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build a compiler-ready protocol JSON from a registered pattern.

    Parameters
    ----------
    pattern_id:
        Registered pattern id (e.g. ``"oer_screening"``).
    params:
        Dict of ``{param_name: value}`` overrides.  Safety-locked params
        are silently forced to their declared defaults.

    Returns
    -------
    dict
        Protocol JSON with ``metadata`` and ``steps`` keys, compatible
        with :func:`app.services.compiler.compile_protocol`.

    Raises
    ------
    ValueError
        If the pattern is not registered or params fail validation.
    """
    pattern = get_pattern(pattern_id)
    if pattern is None:
        raise ValueError(f"unknown pattern '{pattern_id}'")

    errors = pattern.validate_params(params)
    if errors:
        raise ValueError(
            f"param validation failed for pattern '{pattern_id}': "
            + "; ".join(errors)
        )

    protocol = pattern.to_protocol_json(params)
    logger.info(
        "built protocol from pattern '%s' (%d steps, %d param overrides)",
        pattern_id,
        len(protocol["steps"]),
        len(params),
    )
    return protocol
