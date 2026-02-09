"""Safety engine unit tests — boundary conditions, runtime checks, approval flow, contract validation."""
from __future__ import annotations

from app.services.action_contracts import (
    ActionContract,
    Effect,
    Precondition,
    SafetyClass,
    TimeoutConfig,
)
from app.services.safety import (
    evaluate_contract_preflight,
    evaluate_preflight,
    evaluate_runtime_step,
)


def _compiled(*steps: dict) -> dict:
    return {"steps": list(steps)}


def _step(key: str = "s1", primitive: str = "heat", **params) -> dict:
    return {"step_key": key, "primitive": primitive, "params": params}


# ─── preflight: temperature ─────────────────────────────────────────

def test_temp_at_exact_limit_passes() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="heat", temp_c=95.0)),
        policy_snapshot={"max_temp_c": 95.0, "allowed_primitives": ["heat"]},
    )
    assert result.allowed is True


def test_temp_just_above_limit_fails() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="heat", temp_c=95.01)),
        policy_snapshot={"max_temp_c": 95.0, "allowed_primitives": ["heat"]},
    )
    assert result.allowed is False
    assert len(result.violations) == 1


def test_temp_zero_passes() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="heat", temp_c=0)),
        policy_snapshot={"max_temp_c": 95.0, "allowed_primitives": ["heat"]},
    )
    assert result.allowed is True


# ─── preflight: volume ──────────────────────────────────────────────

def test_volume_at_exact_limit_passes() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="aspirate", volume_ul=1000.0)),
        policy_snapshot={"max_volume_ul": 1000.0, "allowed_primitives": ["aspirate"]},
    )
    assert result.allowed is True


def test_volume_above_limit_fails() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="aspirate", volume_ul=1001)),
        policy_snapshot={"max_volume_ul": 1000.0, "allowed_primitives": ["aspirate"]},
    )
    assert result.allowed is False


# ─── preflight: allowed primitives ──────────────────────────────────

def test_disallowed_primitive_fails() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="laser")),
        policy_snapshot={"allowed_primitives": ["heat", "aspirate"]},
    )
    assert result.allowed is False
    assert "primitive not allowed: laser" in result.violations[0]


def test_allowed_primitive_passes() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="heat", temp_c=50)),
        policy_snapshot={"max_temp_c": 95, "allowed_primitives": ["heat"]},
    )
    assert result.allowed is True


# ─── preflight: multiple violations ─────────────────────────────────

def test_multiple_violations_reported() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(
            _step(key="s1", primitive="heat", temp_c=200),
            _step(key="s2", primitive="aspirate", volume_ul=9999),
            _step(key="s3", primitive="laser"),
        ),
        policy_snapshot={
            "max_temp_c": 95,
            "max_volume_ul": 1000,
            "allowed_primitives": ["heat", "aspirate"],
        },
    )
    assert result.allowed is False
    assert len(result.violations) == 3


# ─── preflight: approval flag ───────────────────────────────────────

def test_requires_approval_flag() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="aspirate", volume_ul=10)),
        policy_snapshot={
            "require_human_approval": True,
            "allowed_primitives": ["aspirate"],
        },
    )
    assert result.allowed is True
    assert result.requires_approval is True


def test_no_approval_by_default() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(_step(primitive="aspirate", volume_ul=10)),
        policy_snapshot={"allowed_primitives": ["aspirate"]},
    )
    assert result.requires_approval is False


# ─── preflight: empty steps ────────────────────────────────────────

def test_empty_steps_passes() -> None:
    result = evaluate_preflight(
        compiled_graph=_compiled(),
        policy_snapshot={"allowed_primitives": ["heat"]},
    )
    assert result.allowed is True


# ─── runtime: interlock checks ──────────────────────────────────────

def test_runtime_hardware_interlock_fail() -> None:
    result = evaluate_runtime_step(
        step=_step(primitive="aspirate", volume_ul=50),
        policy_snapshot={"allowed_primitives": ["aspirate"]},
        interlock_state={"hardware_interlock_ok": False},
    )
    assert result.allowed is False
    assert "hardware interlock" in result.violations[0]


def test_runtime_cooling_fail_for_heat() -> None:
    result = evaluate_runtime_step(
        step=_step(primitive="heat", temp_c=50),
        policy_snapshot={"max_temp_c": 95, "allowed_primitives": ["heat"]},
        interlock_state={"hardware_interlock_ok": True, "cooling_ok": False},
    )
    assert result.allowed is False
    assert "cooling" in result.violations[0]


def test_runtime_cooling_irrelevant_for_aspirate() -> None:
    result = evaluate_runtime_step(
        step=_step(primitive="aspirate", volume_ul=50),
        policy_snapshot={"max_volume_ul": 1000, "allowed_primitives": ["aspirate"]},
        interlock_state={"hardware_interlock_ok": True, "cooling_ok": False},
    )
    assert result.allowed is True


def test_runtime_all_ok() -> None:
    result = evaluate_runtime_step(
        step=_step(primitive="heat", temp_c=50),
        policy_snapshot={"max_temp_c": 95, "allowed_primitives": ["heat"]},
        interlock_state={"hardware_interlock_ok": True, "cooling_ok": True},
    )
    assert result.allowed is True


# ─── contract preflight: forward simulation ───────────────────────────


def _contract(
    preconditions: list[str] | None = None,
    effects: list[str] | None = None,
    safety: SafetyClass = SafetyClass.CAREFUL,
) -> ActionContract:
    return ActionContract(
        preconditions=tuple(Precondition(p) for p in (preconditions or [])),
        effects=tuple(Effect(e) for e in (effects or [])),
        timeout=TimeoutConfig(),
        safety_class=safety,
    )


def test_contract_preflight_happy_path() -> None:
    """Valid 3-step sequence: home → load_labware → pick_up_tip."""
    contracts = {
        "robot.home": _contract(effects=["set:robot_homed:true"]),
        "robot.load_labware": _contract(effects=["set:labware_loaded:{name}:true"]),
        "robot.pick_up_tip": _contract(
            preconditions=["labware_loaded:{labware}", "tip_off:{pipette}"],
            effects=["set:tip_state:{pipette}:on"],
        ),
    }
    steps = [
        {"step_key": "s1", "primitive": "robot.home", "params": {}},
        {"step_key": "s2", "primitive": "robot.load_labware", "params": {"name": "tips"}},
        {"step_key": "s3", "primitive": "robot.pick_up_tip", "params": {"labware": "tips", "pipette": "left"}},
    ]
    result = evaluate_contract_preflight(steps=steps, contracts=contracts)
    assert result.allowed is True
    assert result.violations == []


def test_contract_preflight_missing_precondition() -> None:
    """Aspirate without pick_up_tip → precondition failure."""
    contracts = {
        "robot.aspirate": _contract(
            preconditions=["tip_on:{pipette}", "labware_loaded:{labware}"],
            safety=SafetyClass.HAZARDOUS,
        ),
    }
    steps = [
        {"step_key": "s1", "primitive": "robot.aspirate", "params": {"pipette": "left", "labware": "plate1"}},
    ]
    result = evaluate_contract_preflight(steps=steps, contracts=contracts)
    assert result.allowed is False
    assert len(result.violations) == 2
    assert "tip_on:left" in result.violations[0]
    assert "labware_loaded:plate1" in result.violations[1]


def test_contract_preflight_effects_satisfy_later_preconditions() -> None:
    """home's effect (robot_homed:true) isn't a precondition for anything in this chain,
    but load_labware's effect makes labware available for aspirate."""
    contracts = {
        "robot.load_labware": _contract(effects=["set:labware_loaded:{name}:true"]),
        "robot.pick_up_tip": _contract(
            preconditions=["labware_loaded:{labware}"],
            effects=["set:tip_state:{pipette}:on"],
        ),
        "robot.aspirate": _contract(
            preconditions=["tip_on:{pipette}", "labware_loaded:{labware}"],
        ),
    }
    steps = [
        {"step_key": "s1", "primitive": "robot.load_labware", "params": {"name": "plate1"}},
        {"step_key": "s2", "primitive": "robot.pick_up_tip", "params": {"labware": "plate1", "pipette": "left"}},
        {"step_key": "s3", "primitive": "robot.aspirate", "params": {"pipette": "left", "labware": "plate1"}},
    ]
    result = evaluate_contract_preflight(steps=steps, contracts=contracts)
    assert result.allowed is True


def test_contract_preflight_no_contracts() -> None:
    """Steps without contracts should pass (no-op)."""
    steps = [
        {"step_key": "s1", "primitive": "unknown.action", "params": {}},
    ]
    result = evaluate_contract_preflight(steps=steps, contracts={})
    assert result.allowed is True


def test_contract_preflight_empty_steps() -> None:
    """No steps = no violations."""
    result = evaluate_contract_preflight(steps=[], contracts={})
    assert result.allowed is True


def test_contract_preflight_partial_failure() -> None:
    """Only the step with unmet preconditions should generate violations."""
    contracts = {
        "robot.home": _contract(effects=["set:robot_homed:true"]),
        "robot.aspirate": _contract(
            preconditions=["tip_on:{pipette}"],
        ),
    }
    steps = [
        {"step_key": "s1", "primitive": "robot.home", "params": {}},
        {"step_key": "s2", "primitive": "robot.aspirate", "params": {"pipette": "left"}},
    ]
    result = evaluate_contract_preflight(steps=steps, contracts=contracts)
    assert result.allowed is False
    assert len(result.violations) == 1
    assert "s2" in result.violations[0]
    assert "tip_on:left" in result.violations[0]


def test_contract_preflight_cascading_effects() -> None:
    """Effects from failed precondition steps are still applied
    to avoid cascading false positives."""
    contracts = {
        "robot.pick_up_tip": _contract(
            preconditions=["labware_loaded:{labware}"],
            effects=["set:tip_state:{pipette}:on"],
        ),
        "robot.aspirate": _contract(
            preconditions=["tip_on:{pipette}"],
        ),
    }
    steps = [
        # pick_up_tip fails precondition (no labware loaded), but effect still applied
        {"step_key": "s1", "primitive": "robot.pick_up_tip", "params": {"labware": "tips", "pipette": "left"}},
        # aspirate's precondition (tip_on:left) should be satisfied by s1's effect
        {"step_key": "s2", "primitive": "robot.aspirate", "params": {"pipette": "left"}},
    ]
    result = evaluate_contract_preflight(steps=steps, contracts=contracts)
    # Only s1 should fail, not s2 (because effects are applied regardless)
    assert len(result.violations) == 1
    assert "s1" in result.violations[0]
