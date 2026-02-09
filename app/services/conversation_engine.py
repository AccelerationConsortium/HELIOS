"""Conversation engine for campaign initialisation — slot-filling state machine.

Drives a 5-round structured conversation that collects all information
needed to launch an autonomous campaign.  Each round has typed slots,
validation rules, and dynamic options that depend on previous rounds.

All state is persisted in the ``conversation_sessions`` SQLite table
for crash recovery and audit.

Round layout
============
1. Goal & Success Criteria — objective_type, objective_kpi, direction, target_value, acceptable_range_pct
2. Equipment & Constraints — available_instruments, max_temp_c, max_volume_ul, hazardous_reagents, require_human_approval
3. Protocol Template       — pattern_id, optional_steps (dynamic from pattern)
4. Parameter Space         — optimizable_params (from pattern), forbidden_combinations, strategy, batch_size
5. Stopping & Approval     — max_rounds, plateau_threshold, budget_limit_runs, auto_approve_magnitude, human_gate_triggers
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.api.v1.schemas_init import (
    DimensionSpec,
    GoalSpec,
    HumanGatePolicySpec,
    InjectionPack,
    InjectionPackMetadata,
    KPIConfigSpec,
    ParamSpaceSpec,
    ProtocolPatternSpec,
    RoundPresentation,
    RoundResult,
    SafetyRulesSpec,
    SessionStatus,
    SlotPresentation,
)
from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known KPIs by objective type
# ---------------------------------------------------------------------------

_KPI_BY_OBJECTIVE: dict[str, list[str]] = {
    "oer_screening": [
        "overpotential_mv",
        "current_density_ma_cm2",
        "coulombic_efficiency",
        "stability_decay_pct",
        "charge_passed_c",
        "impedance_ohm",
    ],
    "synthesis_optimization": [
        "volume_accuracy_pct",
        "temp_accuracy_c",
        "step_duration_s",
    ],
    "stability_testing": [
        "stability_decay_pct",
        "impedance_ohm",
        "charge_passed_c",
    ],
    "custom": [
        "overpotential_mv",
        "current_density_ma_cm2",
        "coulombic_efficiency",
        "stability_decay_pct",
        "charge_passed_c",
        "impedance_ohm",
        "volume_accuracy_pct",
        "temp_accuracy_c",
        "step_duration_s",
    ],
}

# Instruments known to the system
_AVAILABLE_INSTRUMENTS = ["ot2", "plc", "relay", "squidstat", "furnace", "spin_coater"]

# KPI → required instrument mapping
_KPI_INSTRUMENT_MAP: dict[str, str] = {
    "overpotential_mv": "squidstat",
    "current_density_ma_cm2": "squidstat",
    "coulombic_efficiency": "squidstat",
    "stability_decay_pct": "squidstat",
    "charge_passed_c": "squidstat",
    "impedance_ohm": "squidstat",
    "volume_accuracy_pct": "ot2",
    "temp_accuracy_c": "furnace",
}

# Hazardous reagent options
_HAZARDOUS_REAGENTS = ["KOH", "H2SO4", "HNO3", "organic_solvents", "none"]

# Human gate trigger options
_HUMAN_GATE_TRIGGERS = [
    "new_protocol_pattern",
    "safety_boundary_change",
    "prior_tightening_large",
    "anomaly_detected",
]


# ---------------------------------------------------------------------------
# Session data
# ---------------------------------------------------------------------------


@dataclass
class ConversationSession:
    """In-memory representation of a conversation session."""

    session_id: str
    status: str  # "active" | "completed" | "abandoned"
    current_round: int
    slots: dict[str, Any]
    validation_errors: dict[str, list[str]]
    completed_rounds: list[int]
    created_by: str
    created_at: str
    updated_at: str
    injection_pack_json: str | None = None
    campaign_id: str | None = None


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


def _load_session(session_id: str) -> ConversationSession | None:
    """Load a session from DB.  Returns None if not found."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM conversation_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return ConversationSession(
            session_id=row["id"],
            status=row["status"],
            current_round=row["current_round"],
            slots=parse_json(row["slots_json"], {}),
            validation_errors=parse_json(row["validation_errors_json"], {}),
            completed_rounds=parse_json(row["completed_rounds_json"], []),
            created_by=row["created_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            injection_pack_json=row["injection_pack_json"],
            campaign_id=row["campaign_id"],
        )


def _save_session(session: ConversationSession) -> None:
    """Persist session state to DB (UPSERT)."""
    now = utcnow_iso()

    def _inner(conn: sqlite3.Connection) -> None:
        conn.execute(
            """INSERT INTO conversation_sessions
               (id, status, current_round, slots_json, validation_errors_json,
                completed_rounds_json, injection_pack_json, campaign_id,
                created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 status = excluded.status,
                 current_round = excluded.current_round,
                 slots_json = excluded.slots_json,
                 validation_errors_json = excluded.validation_errors_json,
                 completed_rounds_json = excluded.completed_rounds_json,
                 injection_pack_json = excluded.injection_pack_json,
                 campaign_id = excluded.campaign_id,
                 updated_at = excluded.updated_at
            """,
            (
                session.session_id,
                session.status,
                session.current_round,
                json_dumps(session.slots),
                json_dumps(session.validation_errors),
                json_dumps(session.completed_rounds),
                session.injection_pack_json,
                session.campaign_id,
                session.created_by,
                session.created_at,
                now,
            ),
        )

    run_txn(_inner)
    session.updated_at = now


# ---------------------------------------------------------------------------
# Round builders — produce SlotPresentation lists for each round
# ---------------------------------------------------------------------------

_ROUND_NAMES = {
    1: "Goal & Success Criteria",
    2: "Equipment & Constraints",
    3: "Protocol Template",
    4: "Parameter Space",
    5: "Stopping & Approval",
}

_ROUND_MESSAGES = {
    1: "Let's define your optimization objective. What type of experiment are you running, and what KPI should we optimize?",
    2: "Which instruments are available in your lab, and what are the safety constraints?",
    3: "Select a protocol template for your experiment workflow. You can customize which steps to include.",
    4: "Configure the parameter search space. Which parameters should the optimizer tune, and within what ranges?",
    5: "Set the stopping criteria and approval policies. When should the campaign stop, and what changes need your review?",
}


def _build_round_1(session: ConversationSession) -> list[SlotPresentation]:
    """Goal & Success Criteria."""
    objective_type = session.slots.get("objective_type")
    kpi_options = _KPI_BY_OBJECTIVE.get(objective_type or "custom", _KPI_BY_OBJECTIVE["custom"])
    errors = session.validation_errors

    return [
        SlotPresentation(
            name="objective_type",
            widget="select",
            label="Experiment Type",
            hint="What kind of experiment are you optimizing?",
            options=["oer_screening", "synthesis_optimization", "stability_testing", "custom"],
            default="oer_screening",
            required=True,
            current_value=session.slots.get("objective_type"),
            error=_first_error(errors, "objective_type"),
        ),
        SlotPresentation(
            name="objective_kpi",
            widget="select",
            label="Objective KPI",
            hint="Which KPI should the campaign optimize?",
            options=kpi_options,
            required=True,
            current_value=session.slots.get("objective_kpi"),
            error=_first_error(errors, "objective_kpi"),
        ),
        SlotPresentation(
            name="direction",
            widget="select",
            label="Direction",
            hint="Should this KPI be minimized or maximized?",
            options=["minimize", "maximize"],
            default="minimize",
            required=True,
            current_value=session.slots.get("direction"),
            error=_first_error(errors, "direction"),
        ),
        SlotPresentation(
            name="target_value",
            widget="number",
            label="Target Value",
            hint="Optional: campaign stops when this value is reached",
            required=False,
            current_value=session.slots.get("target_value"),
            error=_first_error(errors, "target_value"),
        ),
        SlotPresentation(
            name="acceptable_range_pct",
            widget="number",
            label="Acceptable Range (%)",
            hint="How close to target is 'good enough'?",
            min_val=1.0,
            max_val=50.0,
            default=10.0,
            required=True,
            unit="%",
            current_value=session.slots.get("acceptable_range_pct"),
            error=_first_error(errors, "acceptable_range_pct"),
        ),
    ]


def _build_round_2(session: ConversationSession) -> list[SlotPresentation]:
    """Equipment & Constraints."""
    errors = session.validation_errors
    return [
        SlotPresentation(
            name="available_instruments",
            widget="multiselect",
            label="Available Instruments",
            hint="Select all instruments connected in your lab",
            options=_AVAILABLE_INSTRUMENTS,
            default=["ot2", "squidstat"],
            required=True,
            current_value=session.slots.get("available_instruments"),
            error=_first_error(errors, "available_instruments"),
        ),
        SlotPresentation(
            name="max_temp_c",
            widget="number",
            label="Max Temperature",
            hint="Absolute maximum temperature allowed",
            min_val=20.0,
            max_val=1200.0,
            default=95.0,
            required=True,
            unit="\u00b0C",
            current_value=session.slots.get("max_temp_c"),
            error=_first_error(errors, "max_temp_c"),
        ),
        SlotPresentation(
            name="max_volume_ul",
            widget="number",
            label="Max Volume",
            hint="Maximum liquid volume per transfer",
            min_val=1.0,
            max_val=10000.0,
            default=1000.0,
            required=True,
            unit="\u00b5L",
            current_value=session.slots.get("max_volume_ul"),
            error=_first_error(errors, "max_volume_ul"),
        ),
        SlotPresentation(
            name="hazardous_reagents",
            widget="multiselect",
            label="Hazardous Reagents",
            hint="Select any hazardous chemicals used",
            options=_HAZARDOUS_REAGENTS,
            default=["none"],
            required=True,
            current_value=session.slots.get("hazardous_reagents"),
            error=_first_error(errors, "hazardous_reagents"),
        ),
        SlotPresentation(
            name="require_human_approval",
            widget="toggle",
            label="Require Human Approval",
            hint="Require manual approval for every run?",
            default=False,
            required=False,
            current_value=session.slots.get("require_human_approval"),
            error=_first_error(errors, "require_human_approval"),
        ),
    ]


def _build_round_3(session: ConversationSession) -> list[SlotPresentation]:
    """Protocol Template selection."""
    from app.services.protocol_patterns import list_patterns

    errors = session.validation_errors
    patterns = list_patterns()
    pattern_options = [{"id": p.id, "name": p.name, "domain": p.domain} for p in patterns]
    pattern_ids = [p.id for p in patterns]

    # If a pattern is selected, show its steps
    selected_id = session.slots.get("pattern_id")
    mandatory = []
    optional = []
    if selected_id:
        from app.services.protocol_patterns import get_pattern

        pattern = get_pattern(selected_id)
        if pattern:
            mandatory = [s.name for s in pattern.steps]
            # For now all built-in steps are mandatory; future patterns may declare optional steps

    return [
        SlotPresentation(
            name="pattern_id",
            widget="select",
            label="Protocol Pattern",
            hint="Select the experiment workflow template",
            options=pattern_ids if pattern_ids else ["oer_screening"],
            required=True,
            current_value=session.slots.get("pattern_id"),
            error=_first_error(errors, "pattern_id"),
        ),
        SlotPresentation(
            name="optional_steps",
            widget="multiselect",
            label="Optional Steps",
            hint="Deselect steps you want to skip (if any)",
            options=optional if optional else [],
            required=False,
            current_value=session.slots.get("optional_steps"),
            error=_first_error(errors, "optional_steps"),
        ),
        SlotPresentation(
            name="mandatory_steps",
            widget="display",
            label="Required Steps",
            hint="These steps are always included",
            required=False,
            current_value=mandatory,
        ),
    ]


def _build_round_4(session: ConversationSession) -> list[SlotPresentation]:
    """Parameter Space configuration."""
    errors = session.validation_errors

    # Load params from selected pattern
    param_dims: list[dict[str, Any]] = []
    selected_id = session.slots.get("pattern_id")
    if selected_id:
        from app.services.protocol_patterns import get_pattern

        pattern = get_pattern(selected_id)
        if pattern:
            for step in pattern.steps:
                for p in step.params:
                    param_dims.append({
                        "param_name": p.name,
                        "param_type": p.param_type,
                        "min_value": p.min_value,
                        "max_value": p.max_value,
                        "log_scale": p.log_scale,
                        "choices": list(p.choices) if p.choices else None,
                        "optimizable": p.optimizable and not p.safety_locked,
                        "step_key": step.name,
                        "primitive": step.primitive,
                        "unit": p.unit,
                        "description": p.description,
                        "safety_locked": p.safety_locked,
                        "default": p.default,
                    })

    # Use user's previous edits if available
    existing_params = session.slots.get("optimizable_params")
    if existing_params is None:
        existing_params = param_dims

    return [
        SlotPresentation(
            name="optimizable_params",
            widget="param_editor",
            label="Optimization Parameters",
            hint="Configure which parameters to optimize and their ranges",
            required=True,
            current_value=existing_params,
            error=_first_error(errors, "optimizable_params"),
        ),
        SlotPresentation(
            name="forbidden_combinations",
            widget="text",
            label="Forbidden Combinations",
            hint="Optional: constraint expressions (e.g. 'annealing_temp > 500 AND ratio > 5')",
            required=False,
            current_value=session.slots.get("forbidden_combinations"),
            error=_first_error(errors, "forbidden_combinations"),
        ),
        SlotPresentation(
            name="strategy",
            widget="select",
            label="Sampling Strategy",
            hint="How to explore the parameter space",
            options=["lhs", "prior_guided", "random", "grid"],
            default="lhs",
            required=True,
            current_value=session.slots.get("strategy"),
            error=_first_error(errors, "strategy"),
        ),
        SlotPresentation(
            name="batch_size",
            widget="number",
            label="Batch Size",
            hint="Number of candidates per round",
            min_val=1,
            max_val=100,
            default=10,
            required=True,
            current_value=session.slots.get("batch_size"),
            error=_first_error(errors, "batch_size"),
        ),
    ]


def _build_round_5(session: ConversationSession) -> list[SlotPresentation]:
    """Stopping & Approval policies."""
    errors = session.validation_errors
    return [
        SlotPresentation(
            name="max_rounds",
            widget="number",
            label="Max Rounds",
            hint="Maximum number of optimization rounds",
            min_val=1,
            max_val=1000,
            default=20,
            required=True,
            current_value=session.slots.get("max_rounds"),
            error=_first_error(errors, "max_rounds"),
        ),
        SlotPresentation(
            name="plateau_threshold",
            widget="number",
            label="Plateau Threshold",
            hint="Convergence detection sensitivity (lower = more sensitive)",
            min_val=0.001,
            max_val=0.1,
            step_val=0.001,
            default=0.01,
            required=True,
            current_value=session.slots.get("plateau_threshold"),
            error=_first_error(errors, "plateau_threshold"),
        ),
        SlotPresentation(
            name="budget_limit_runs",
            widget="number",
            label="Budget Limit (runs)",
            hint="Optional: hard limit on total number of runs",
            min_val=1,
            max_val=10000,
            required=False,
            current_value=session.slots.get("budget_limit_runs"),
            error=_first_error(errors, "budget_limit_runs"),
        ),
        SlotPresentation(
            name="auto_approve_magnitude",
            widget="number",
            label="Auto-Approve Threshold",
            hint="Evolution changes below this magnitude are auto-approved (0-1)",
            min_val=0.0,
            max_val=1.0,
            step_val=0.05,
            default=0.3,
            required=True,
            current_value=session.slots.get("auto_approve_magnitude"),
            error=_first_error(errors, "auto_approve_magnitude"),
        ),
        SlotPresentation(
            name="human_gate_triggers",
            widget="multiselect",
            label="Human Gate Triggers",
            hint="Events that require manual review",
            options=_HUMAN_GATE_TRIGGERS,
            default=["safety_boundary_change"],
            required=True,
            current_value=session.slots.get("human_gate_triggers"),
            error=_first_error(errors, "human_gate_triggers"),
        ),
    ]


_ROUND_BUILDERS = {
    1: _build_round_1,
    2: _build_round_2,
    3: _build_round_3,
    4: _build_round_4,
    5: _build_round_5,
}


def _first_error(errors: dict[str, list[str]], slot_name: str) -> str | None:
    """Return first error message for a slot, or None."""
    errs = errors.get(slot_name)
    return errs[0] if errs else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_round(
    round_num: int, responses: dict[str, Any], session: ConversationSession
) -> dict[str, list[str]]:
    """Validate responses for a given round.  Returns {slot_name: [errors]}."""
    errors: dict[str, list[str]] = {}

    if round_num == 1:
        _validate_round_1(responses, errors)
    elif round_num == 2:
        _validate_round_2(responses, session, errors)
    elif round_num == 3:
        _validate_round_3(responses, errors)
    elif round_num == 4:
        _validate_round_4(responses, errors)
    elif round_num == 5:
        _validate_round_5(responses, errors)

    return errors


def _validate_round_1(responses: dict[str, Any], errors: dict[str, list[str]]) -> None:
    obj_type = responses.get("objective_type")
    if not obj_type or obj_type not in _KPI_BY_OBJECTIVE:
        errors.setdefault("objective_type", []).append(
            f"Must be one of: {', '.join(_KPI_BY_OBJECTIVE.keys())}"
        )

    kpi = responses.get("objective_kpi")
    valid_kpis = _KPI_BY_OBJECTIVE.get(obj_type or "custom", _KPI_BY_OBJECTIVE["custom"])
    if not kpi or kpi not in valid_kpis:
        errors.setdefault("objective_kpi", []).append(
            f"Must be one of: {', '.join(valid_kpis)}"
        )

    direction = responses.get("direction")
    if direction not in ("minimize", "maximize"):
        errors.setdefault("direction", []).append("Must be 'minimize' or 'maximize'")

    target = responses.get("target_value")
    if target is not None:
        try:
            float(target)
        except (TypeError, ValueError):
            errors.setdefault("target_value", []).append("Must be a number")

    rng = responses.get("acceptable_range_pct")
    if rng is not None:
        try:
            val = float(rng)
            if val < 1 or val > 50:
                errors.setdefault("acceptable_range_pct", []).append("Must be between 1 and 50")
        except (TypeError, ValueError):
            errors.setdefault("acceptable_range_pct", []).append("Must be a number")


def _validate_round_2(
    responses: dict[str, Any],
    session: ConversationSession,
    errors: dict[str, list[str]],
) -> None:
    instruments = responses.get("available_instruments")
    if not instruments or not isinstance(instruments, list) or len(instruments) == 0:
        errors.setdefault("available_instruments", []).append("Select at least one instrument")
    elif any(i not in _AVAILABLE_INSTRUMENTS for i in instruments):
        errors.setdefault("available_instruments", []).append(
            f"Unknown instruments. Valid: {', '.join(_AVAILABLE_INSTRUMENTS)}"
        )

    # Cross-round: check KPI requires instrument
    kpi = session.slots.get("objective_kpi")
    if kpi and instruments and isinstance(instruments, list):
        required_instr = _KPI_INSTRUMENT_MAP.get(kpi)
        if required_instr and required_instr not in instruments:
            errors.setdefault("available_instruments", []).append(
                f"KPI '{kpi}' requires instrument '{required_instr}'"
            )

    _validate_number(responses, errors, "max_temp_c", 20.0, 1200.0, required=True)
    _validate_number(responses, errors, "max_volume_ul", 1.0, 10000.0, required=True)


def _validate_round_3(responses: dict[str, Any], errors: dict[str, list[str]]) -> None:
    from app.services.protocol_patterns import get_pattern

    pattern_id = responses.get("pattern_id")
    if not pattern_id:
        errors.setdefault("pattern_id", []).append("Select a protocol pattern")
    elif get_pattern(pattern_id) is None:
        errors.setdefault("pattern_id", []).append(f"Unknown pattern '{pattern_id}'")


def _validate_round_4(responses: dict[str, Any], errors: dict[str, list[str]]) -> None:
    strategy = responses.get("strategy")
    if strategy not in ("lhs", "prior_guided", "random", "grid"):
        errors.setdefault("strategy", []).append("Must be one of: lhs, prior_guided, random, grid")

    _validate_number(responses, errors, "batch_size", 1, 100, required=True)


def _validate_round_5(responses: dict[str, Any], errors: dict[str, list[str]]) -> None:
    _validate_number(responses, errors, "max_rounds", 1, 1000, required=True)
    _validate_number(responses, errors, "plateau_threshold", 0.001, 0.1, required=True)
    _validate_number(responses, errors, "budget_limit_runs", 1, 10000, required=False)
    _validate_number(responses, errors, "auto_approve_magnitude", 0.0, 1.0, required=True)

    triggers = responses.get("human_gate_triggers")
    if not triggers or not isinstance(triggers, list) or len(triggers) == 0:
        errors.setdefault("human_gate_triggers", []).append("Select at least one trigger")


def _validate_number(
    responses: dict[str, Any],
    errors: dict[str, list[str]],
    name: str,
    min_val: float,
    max_val: float,
    required: bool,
) -> None:
    """Validate a numeric slot."""
    val = responses.get(name)
    if val is None or val == "":
        if required:
            errors.setdefault(name, []).append("This field is required")
        return
    try:
        num = float(val)
    except (TypeError, ValueError):
        errors.setdefault(name, []).append("Must be a number")
        return
    if num < min_val or num > max_val:
        errors.setdefault(name, []).append(f"Must be between {min_val} and {max_val}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_session(author: str = "user") -> ConversationSession:
    """Create a new conversation session and return the initial state."""
    session = ConversationSession(
        session_id=str(uuid.uuid4()),
        status="active",
        current_round=1,
        slots={},
        validation_errors={},
        completed_rounds=[],
        created_by=author,
        created_at=utcnow_iso(),
        updated_at=utcnow_iso(),
    )
    _save_session(session)
    return session


def get_current_round(session_id: str) -> RoundPresentation:
    """Return the presentation for the current round."""
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found")
    return _build_round_presentation(session)


def submit_round(session_id: str, responses: dict[str, Any]) -> RoundResult:
    """Submit responses for the current round.

    Validates, stores, and advances to the next round on success.
    Returns the same round with error annotations on failure.
    """
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found")
    if session.status != "active":
        raise ValueError(f"Session '{session_id}' is {session.status}, not active")

    round_num = session.current_round

    # Apply defaults for missing optional fields
    _apply_defaults(round_num, responses)

    # Validate
    errors = _validate_round(round_num, responses, session)
    if errors:
        session.validation_errors = errors
        _save_session(session)
        return RoundResult(
            success=False,
            next_round=_build_round_presentation(session),
            errors=errors,
            injection_pack_preview=_build_preview(session),
        )

    # Store responses in session slots
    session.validation_errors = {}
    for key, value in responses.items():
        session.slots[key] = value

    # Mark round completed
    if round_num not in session.completed_rounds:
        session.completed_rounds.append(round_num)

    # Advance to next round or mark complete
    if round_num < 5:
        session.current_round = round_num + 1
        _save_session(session)
        return RoundResult(
            success=True,
            next_round=_build_round_presentation(session),
            injection_pack_preview=_build_preview(session),
        )
    else:
        # Final round completed — session ready for confirmation
        _save_session(session)
        return RoundResult(
            success=True,
            next_round=_build_round_presentation(session),
            injection_pack_preview=_build_preview(session),
        )


def go_back(session_id: str) -> RoundPresentation:
    """Go back to the previous round."""
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found")
    if session.current_round > 1:
        session.current_round -= 1
        session.validation_errors = {}
        _save_session(session)
    return _build_round_presentation(session)


def get_session_status(session_id: str) -> SessionStatus:
    """Return current status of the session."""
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found")
    return SessionStatus(
        session_id=session.session_id,
        status=session.status,
        current_round=session.current_round,
        completed_rounds=session.completed_rounds,
        filled_slots=session.slots,
        injection_pack_preview=_build_preview(session),
    )


def confirm_and_build(session_id: str) -> InjectionPack:
    """Build the injection pack from completed session slots.

    Requires all 5 rounds to be completed.
    """
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"Session '{session_id}' not found")

    missing_rounds = [r for r in range(1, 6) if r not in session.completed_rounds]
    if missing_rounds:
        raise ValueError(f"Rounds {missing_rounds} not completed yet")

    pack = _assemble_injection_pack(session)

    # Store and mark session completed
    session.injection_pack_json = pack.model_dump_json()
    session.status = "completed"
    _save_session(session)

    return pack


def get_all_kpis() -> list[dict[str, str]]:
    """Return all known KPI definitions for the init UI."""
    result = []
    for kpis in _KPI_BY_OBJECTIVE.values():
        for kpi in kpis:
            if not any(r["name"] == kpi for r in result):
                result.append({"name": kpi, "instrument": _KPI_INSTRUMENT_MAP.get(kpi, "any")})
    return result


def get_all_patterns() -> list[dict[str, str]]:
    """Return all registered patterns for the init UI."""
    from app.services.protocol_patterns import list_patterns

    return [
        {"id": p.id, "name": p.name, "domain": p.domain, "description": p.description}
        for p in list_patterns()
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_round_presentation(session: ConversationSession) -> RoundPresentation:
    """Build a RoundPresentation for the current round."""
    round_num = session.current_round
    builder = _ROUND_BUILDERS.get(round_num)
    if builder is None:
        raise ValueError(f"Invalid round {round_num}")

    slots = builder(session)
    return RoundPresentation(
        round_number=round_num,
        round_name=_ROUND_NAMES[round_num],
        message=_ROUND_MESSAGES[round_num],
        slots=slots,
        is_final=(round_num == 5),
        completed=(round_num in session.completed_rounds),
    )


def _apply_defaults(round_num: int, responses: dict[str, Any]) -> None:
    """Fill in defaults for optional / toggle fields that weren't sent."""
    if round_num == 1:
        responses.setdefault("acceptable_range_pct", 10.0)
    elif round_num == 2:
        responses.setdefault("require_human_approval", False)
        responses.setdefault("hazardous_reagents", ["none"])
    elif round_num == 4:
        responses.setdefault("strategy", "lhs")
        responses.setdefault("batch_size", 10)
        responses.setdefault("forbidden_combinations", "")
    elif round_num == 5:
        responses.setdefault("auto_approve_magnitude", 0.3)
        responses.setdefault("human_gate_triggers", ["safety_boundary_change"])


def _build_preview(session: ConversationSession) -> dict[str, Any]:
    """Build a partial injection pack preview from current slots."""
    preview: dict[str, Any] = {}
    s = session.slots

    if 1 in session.completed_rounds:
        preview["goal"] = {
            "objective_type": s.get("objective_type"),
            "objective_kpi": s.get("objective_kpi"),
            "direction": s.get("direction"),
            "target_value": s.get("target_value"),
        }

    if 2 in session.completed_rounds:
        preview["safety"] = {
            "available_instruments": s.get("available_instruments"),
            "max_temp_c": s.get("max_temp_c"),
            "max_volume_ul": s.get("max_volume_ul"),
            "hazardous_reagents": s.get("hazardous_reagents"),
        }

    if 3 in session.completed_rounds:
        preview["protocol"] = {
            "pattern_id": s.get("pattern_id"),
        }

    if 4 in session.completed_rounds:
        preview["param_space"] = {
            "strategy": s.get("strategy"),
            "batch_size": s.get("batch_size"),
            "n_params": len([
                p for p in (s.get("optimizable_params") or [])
                if isinstance(p, dict) and p.get("optimizable")
            ]),
        }

    if 5 in session.completed_rounds:
        preview["human_gate"] = {
            "max_rounds": s.get("max_rounds"),
            "plateau_threshold": s.get("plateau_threshold"),
            "human_gate_triggers": s.get("human_gate_triggers"),
        }

    return preview


def _assemble_injection_pack(session: ConversationSession) -> InjectionPack:
    """Assemble a complete InjectionPack from session slots."""
    s = session.slots

    # Build dimensions from optimizable_params
    dims: list[DimensionSpec] = []
    raw_params = s.get("optimizable_params", [])
    for p in raw_params:
        if isinstance(p, dict) and p.get("optimizable") and not p.get("safety_locked"):
            dims.append(DimensionSpec(
                param_name=p["param_name"],
                param_type=p.get("param_type", "number"),
                min_value=p.get("min_value"),
                max_value=p.get("max_value"),
                log_scale=p.get("log_scale", False),
                choices=p.get("choices"),
                optimizable=True,
                step_key=p.get("step_key"),
                primitive=p.get("primitive"),
                unit=p.get("unit", ""),
                description=p.get("description", ""),
                safety_locked=False,
            ))

    # Derive allowed primitives from instruments
    instrument_primitives: dict[str, list[str]] = {
        "ot2": ["robot.home", "robot.load_pipettes", "robot.set_lights",
                "robot.load_labware", "robot.load_custom_labware",
                "robot.move_to_well", "robot.pick_up_tip", "robot.drop_tip",
                "robot.aspirate", "robot.dispense", "robot.blowout"],
        "plc": ["plc.dispense_ml", "plc.set_pump_on_timer", "plc.set_ultrasonic_on_timer"],
        "relay": ["relay.set_channel", "relay.turn_on", "relay.turn_off", "relay.switch_to"],
        "squidstat": ["squidstat.run_experiment", "squidstat.get_data",
                      "squidstat.save_snapshot", "squidstat.reset_plot"],
        "furnace": ["heat"],
        "spin_coater": [],
    }
    allowed = ["wait", "log"]  # always available
    for instr in (s.get("available_instruments") or []):
        allowed.extend(instrument_primitives.get(instr, []))

    # Compute forbidden_combinations
    forbidden_raw = s.get("forbidden_combinations", "")
    forbidden = [fc.strip() for fc in forbidden_raw.split(",") if fc.strip()] if isinstance(forbidden_raw, str) else []

    # Get pattern mandatory steps
    mandatory_steps: list[str] = []
    pattern_id = s.get("pattern_id", "")
    if pattern_id:
        from app.services.protocol_patterns import get_pattern
        pattern = get_pattern(pattern_id)
        if pattern:
            mandatory_steps = [step.name for step in pattern.steps]

    goal = GoalSpec(
        objective_type=s.get("objective_type", "custom"),
        objective_kpi=s["objective_kpi"],
        direction=s.get("direction", "minimize"),
        target_value=_to_float_or_none(s.get("target_value")),
        acceptable_range_pct=float(s.get("acceptable_range_pct", 10.0)),
    )

    protocol = ProtocolPatternSpec(
        pattern_id=pattern_id,
        optional_steps=s.get("optional_steps", []),
        mandatory_steps=mandatory_steps,
    )

    param_space = ParamSpaceSpec(
        dimensions=dims,
        strategy=s.get("strategy", "lhs"),
        batch_size=int(s.get("batch_size", 10)),
        forbidden_combinations=forbidden,
    )

    safety = SafetyRulesSpec(
        max_temp_c=float(s.get("max_temp_c", 95.0)),
        max_volume_ul=float(s.get("max_volume_ul", 1000.0)),
        allowed_primitives=sorted(set(allowed)),
        require_human_approval=bool(s.get("require_human_approval", False)),
        hazardous_reagents=s.get("hazardous_reagents", []),
    )

    kpi_config = KPIConfigSpec(
        primary_kpi=s["objective_kpi"],
        secondary_kpis=[],
        target_value=_to_float_or_none(s.get("target_value")),
        acceptable_range_pct=float(s.get("acceptable_range_pct", 10.0)),
    )

    human_gate = HumanGatePolicySpec(
        auto_approve_magnitude=float(s.get("auto_approve_magnitude", 0.3)),
        human_gate_triggers=s.get("human_gate_triggers", ["safety_boundary_change"]),
        plateau_threshold=float(s.get("plateau_threshold", 0.01)),
        max_rounds=int(s.get("max_rounds", 20)),
        budget_limit_runs=_to_int_or_none(s.get("budget_limit_runs")),
    )

    # Build body JSON for checksum (exclude metadata)
    body = {
        "goal": goal.model_dump(),
        "protocol": protocol.model_dump(),
        "param_space": param_space.model_dump(),
        "safety": safety.model_dump(),
        "kpi_config": kpi_config.model_dump(),
        "human_gate": human_gate.model_dump(),
    }
    body_json = json.dumps(body, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(body_json.encode()).hexdigest()

    metadata = InjectionPackMetadata(
        session_id=session.session_id,
        version="1.0",
        created_at=utcnow_iso(),
        created_by=session.created_by,
        checksum=checksum,
    )

    return InjectionPack(
        goal=goal,
        protocol=protocol,
        param_space=param_space,
        safety=safety,
        kpi_config=kpi_config,
        human_gate=human_gate,
        metadata=metadata,
    )


def _to_float_or_none(val: Any) -> float | None:
    if val is None or val == "" or val == "null":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(val: Any) -> int | None:
    if val is None or val == "" or val == "null":
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None
