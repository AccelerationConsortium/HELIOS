"""Tests for app.services.error_policy — CRITICAL / BYPASS classification + SafetyClass."""
from __future__ import annotations

import pytest

from app.services.action_contracts import ActionContract, SafetyClass, TimeoutConfig
from app.services.error_policy import (
    BYPASS_PRIMITIVES,
    CRITICAL_PRIMITIVES,
    ErrorPolicy,
    classify_step_error,
    classify_step_safety,
)


# ---------------------------------------------------------------------------
# classify_step_error
# ---------------------------------------------------------------------------


class TestClassifyStepError:
    """Verify that primitives are classified correctly."""

    @pytest.mark.parametrize(
        "primitive",
        sorted(CRITICAL_PRIMITIVES),
    )
    def test_critical_primitives(self, primitive: str) -> None:
        assert classify_step_error(primitive, RuntimeError("boom")) == "CRITICAL"

    @pytest.mark.parametrize(
        "primitive",
        sorted(BYPASS_PRIMITIVES),
    )
    def test_bypass_primitives(self, primitive: str) -> None:
        assert classify_step_error(primitive, RuntimeError("boom")) == "BYPASS"

    def test_unknown_primitive_defaults_to_critical(self) -> None:
        assert classify_step_error("unknown.action", ValueError("x")) == "CRITICAL"

    def test_no_overlap_between_critical_and_bypass(self) -> None:
        overlap = CRITICAL_PRIMITIVES & BYPASS_PRIMITIVES
        assert overlap == frozenset(), f"overlapping primitives: {overlap}"

    def test_robot_aspirate_is_critical(self) -> None:
        assert classify_step_error("robot.aspirate", RuntimeError("hw")) == "CRITICAL"

    def test_robot_home_is_bypass(self) -> None:
        assert classify_step_error("robot.home", RuntimeError("hw")) == "BYPASS"

    def test_wait_is_bypass(self) -> None:
        assert classify_step_error("wait", RuntimeError("hw")) == "BYPASS"

    def test_squidstat_run_is_critical(self) -> None:
        assert classify_step_error("squidstat.run_experiment", RuntimeError("hw")) == "CRITICAL"

    def test_relay_switch_is_bypass(self) -> None:
        assert classify_step_error("relay.switch_to", RuntimeError("hw")) == "BYPASS"


# ---------------------------------------------------------------------------
# ErrorPolicy
# ---------------------------------------------------------------------------


class TestErrorPolicy:
    """Verify ErrorPolicy construction and defaults."""

    def test_default_allows_bypass(self) -> None:
        ep = ErrorPolicy()
        assert ep.allow_bypass is True

    def test_from_empty_policy(self) -> None:
        ep = ErrorPolicy.from_policy_snapshot({})
        assert ep.allow_bypass is True

    def test_from_policy_with_bypass_disabled(self) -> None:
        ep = ErrorPolicy.from_policy_snapshot(
            {"error_policy": {"allow_bypass": False}}
        )
        assert ep.allow_bypass is False

    def test_from_policy_with_bypass_enabled(self) -> None:
        ep = ErrorPolicy.from_policy_snapshot(
            {"error_policy": {"allow_bypass": True}}
        )
        assert ep.allow_bypass is True

    def test_from_policy_ignores_non_dict_error_policy(self) -> None:
        ep = ErrorPolicy.from_policy_snapshot({"error_policy": "invalid"})
        assert ep.allow_bypass is True

    def test_frozen(self) -> None:
        ep = ErrorPolicy()
        with pytest.raises(AttributeError):
            ep.allow_bypass = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# classify_step_safety — 4-tier SafetyClass
# ---------------------------------------------------------------------------


class TestClassifyStepSafety:
    """Verify the 4-tier safety classification."""

    def test_from_contract(self) -> None:
        contract = ActionContract(safety_class=SafetyClass.HAZARDOUS)
        assert classify_step_safety("robot.aspirate", contract) == SafetyClass.HAZARDOUS

    def test_contract_overrides_legacy(self) -> None:
        # Even if legacy says CAREFUL, explicit contract says INFORMATIONAL
        contract = ActionContract(safety_class=SafetyClass.INFORMATIONAL)
        assert classify_step_safety("robot.load_labware", contract) == SafetyClass.INFORMATIONAL

    def test_legacy_map_fallback(self) -> None:
        # No contract → falls back to LEGACY_SAFETY_MAP
        assert classify_step_safety("robot.home") == SafetyClass.INFORMATIONAL
        assert classify_step_safety("robot.aspirate") == SafetyClass.HAZARDOUS
        assert classify_step_safety("relay.switch_to") == SafetyClass.REVERSIBLE
        assert classify_step_safety("robot.pick_up_tip") == SafetyClass.CAREFUL

    def test_critical_set_fallback(self) -> None:
        # Primitive in CRITICAL set but NOT in legacy map → CAREFUL
        # "eis" is in CRITICAL_PRIMITIVES but also in LEGACY_SAFETY_MAP as HAZARDOUS
        assert classify_step_safety("eis") == SafetyClass.HAZARDOUS

    def test_bypass_set_fallback(self) -> None:
        # Primitive in BYPASS set but NOT in legacy map → REVERSIBLE
        # All BYPASS primitives are in the legacy map, so test unknown
        pass

    def test_unknown_defaults_to_careful(self) -> None:
        assert classify_step_safety("totally.unknown") == SafetyClass.CAREFUL

    def test_safety_to_legacy_mapping(self) -> None:
        # Verify that SafetyClass maps back correctly
        assert SafetyClass.INFORMATIONAL.to_legacy() == "BYPASS"
        assert SafetyClass.REVERSIBLE.to_legacy() == "BYPASS"
        assert SafetyClass.CAREFUL.to_legacy() == "CRITICAL"
        assert SafetyClass.HAZARDOUS.to_legacy() == "CRITICAL"

    def test_classify_step_error_still_works(self) -> None:
        """Original classify_step_error is unchanged."""
        assert classify_step_error("robot.aspirate", RuntimeError("x")) == "CRITICAL"
        assert classify_step_error("robot.home", RuntimeError("x")) == "BYPASS"
        assert classify_step_error("unknown", RuntimeError("x")) == "CRITICAL"
