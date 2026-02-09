"""Tests for the conversation engine slot-filling state machine."""
from __future__ import annotations

import os
import tempfile

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_conv_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "conv_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db  # noqa: E402
from app.services.conversation_engine import (  # noqa: E402
    ConversationSession,
    confirm_and_build,
    get_all_kpis,
    get_all_patterns,
    get_current_round,
    get_session_status,
    go_back,
    start_session,
    submit_round,
)


@pytest.fixture(autouse=True)
def _setup_db():
    get_settings.cache_clear()
    init_db()
    # Clean conversation sessions for isolation
    with connection() as conn:
        conn.execute("DELETE FROM conversation_sessions")
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers — canonical round responses
# ---------------------------------------------------------------------------

ROUND_1_VALID = {
    "objective_type": "oer_screening",
    "objective_kpi": "overpotential_mv",
    "direction": "minimize",
    "target_value": None,
    "acceptable_range_pct": 10.0,
}

ROUND_2_VALID = {
    "available_instruments": ["ot2", "squidstat"],
    "max_temp_c": 95.0,
    "max_volume_ul": 1000.0,
    "hazardous_reagents": ["KOH"],
    "require_human_approval": False,
}

ROUND_3_VALID = {
    "pattern_id": "oer_screening",
    "optional_steps": [],
}

ROUND_4_VALID = {
    "strategy": "lhs",
    "batch_size": 10,
    "forbidden_combinations": "",
}

ROUND_5_VALID = {
    "max_rounds": 20,
    "plateau_threshold": 0.01,
    "budget_limit_runs": None,
    "auto_approve_magnitude": 0.3,
    "human_gate_triggers": ["safety_boundary_change"],
}


def _advance_to_round(session_id: str, target_round: int) -> None:
    """Helper to advance session from round 1 to target_round."""
    rounds_data = [None, ROUND_1_VALID, ROUND_2_VALID, ROUND_3_VALID, ROUND_4_VALID, ROUND_5_VALID]
    for r in range(1, target_round):
        result = submit_round(session_id, rounds_data[r])
        assert result.success, f"Round {r} submit failed: {result.errors}"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_start_session_creates_active_session(self):
        session = start_session(author="tester")
        assert session.status == "active"
        assert session.current_round == 1
        assert session.session_id is not None
        assert session.created_by == "tester"

    def test_start_session_returns_round_1(self):
        session = start_session(author="tester")
        rp = get_current_round(session.session_id)
        assert rp.round_number == 1
        assert rp.round_name == "Goal & Success Criteria"
        assert not rp.is_final

    def test_get_status_returns_correct_state(self):
        session = start_session()
        status = get_session_status(session.session_id)
        assert status.session_id == session.session_id
        assert status.status == "active"
        assert status.current_round == 1
        assert status.completed_rounds == []

    def test_invalid_session_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_current_round("nonexistent-id")

    def test_submit_to_completed_session_raises(self):
        session = start_session()
        _advance_to_round(session.session_id, 6)  # complete all 5
        pack = confirm_and_build(session.session_id)
        assert pack is not None
        with pytest.raises(ValueError, match="not active"):
            submit_round(session.session_id, ROUND_1_VALID)


# ---------------------------------------------------------------------------
# Round navigation
# ---------------------------------------------------------------------------


class TestRoundNavigation:
    def test_submit_round_1_advances_to_round_2(self):
        session = start_session()
        result = submit_round(session.session_id, ROUND_1_VALID)
        assert result.success
        assert result.next_round.round_number == 2

    def test_submit_all_rounds_returns_completed_state(self):
        session = start_session()
        _advance_to_round(session.session_id, 6)  # all 5 rounds done
        status = get_session_status(session.session_id)
        assert 5 in status.completed_rounds
        assert len(status.completed_rounds) == 5

    def test_go_back_from_round_2_returns_round_1(self):
        session = start_session()
        submit_round(session.session_id, ROUND_1_VALID)
        rp = go_back(session.session_id)
        assert rp.round_number == 1

    def test_go_back_from_round_1_stays_at_round_1(self):
        session = start_session()
        rp = go_back(session.session_id)
        assert rp.round_number == 1

    def test_round_progression_1_through_5(self):
        session = start_session()
        for r, data in enumerate([ROUND_1_VALID, ROUND_2_VALID, ROUND_3_VALID, ROUND_4_VALID, ROUND_5_VALID], 1):
            result = submit_round(session.session_id, data)
            assert result.success, f"Round {r} failed: {result.errors}"
            if r < 5:
                assert result.next_round.round_number == r + 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_round_1_invalid_objective_type(self):
        session = start_session()
        bad = {**ROUND_1_VALID, "objective_type": "invalid_type"}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "objective_type" in result.errors

    def test_round_1_invalid_direction(self):
        session = start_session()
        bad = {**ROUND_1_VALID, "direction": "sideways"}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "direction" in result.errors

    def test_round_1_invalid_kpi_for_objective(self):
        session = start_session()
        bad = {**ROUND_1_VALID, "objective_kpi": "volume_accuracy_pct"}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "objective_kpi" in result.errors

    def test_round_1_acceptable_range_out_of_bounds(self):
        session = start_session()
        bad = {**ROUND_1_VALID, "acceptable_range_pct": 99.0}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "acceptable_range_pct" in result.errors

    def test_round_2_empty_instruments_rejected(self):
        session = start_session()
        submit_round(session.session_id, ROUND_1_VALID)
        bad = {**ROUND_2_VALID, "available_instruments": []}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "available_instruments" in result.errors

    def test_round_2_cross_round_kpi_instrument_check(self):
        """If KPI is overpotential_mv, squidstat must be selected."""
        session = start_session()
        submit_round(session.session_id, ROUND_1_VALID)  # KPI = overpotential_mv
        bad = {**ROUND_2_VALID, "available_instruments": ["ot2"]}  # no squidstat
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "available_instruments" in result.errors
        assert any("squidstat" in e for e in result.errors["available_instruments"])

    def test_round_2_invalid_temp_range(self):
        session = start_session()
        submit_round(session.session_id, ROUND_1_VALID)
        bad = {**ROUND_2_VALID, "max_temp_c": 99999.0}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "max_temp_c" in result.errors

    def test_round_3_unknown_pattern_rejected(self):
        session = start_session()
        _advance_to_round(session.session_id, 3)
        bad = {"pattern_id": "nonexistent_pattern"}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "pattern_id" in result.errors

    def test_round_4_invalid_strategy(self):
        session = start_session()
        _advance_to_round(session.session_id, 4)
        bad = {**ROUND_4_VALID, "strategy": "magic"}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "strategy" in result.errors

    def test_round_5_invalid_max_rounds(self):
        session = start_session()
        _advance_to_round(session.session_id, 5)
        bad = {**ROUND_5_VALID, "max_rounds": 0}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "max_rounds" in result.errors

    def test_round_5_empty_triggers_rejected(self):
        session = start_session()
        _advance_to_round(session.session_id, 5)
        bad = {**ROUND_5_VALID, "human_gate_triggers": []}
        result = submit_round(session.session_id, bad)
        assert not result.success
        assert "human_gate_triggers" in result.errors


# ---------------------------------------------------------------------------
# Round presentation
# ---------------------------------------------------------------------------


class TestRoundPresentation:
    def test_round_1_has_expected_slots(self):
        session = start_session()
        rp = get_current_round(session.session_id)
        names = [s.name for s in rp.slots]
        assert "objective_type" in names
        assert "objective_kpi" in names
        assert "direction" in names

    def test_round_3_shows_pattern_options(self):
        session = start_session()
        _advance_to_round(session.session_id, 3)
        rp = get_current_round(session.session_id)
        pattern_slot = next(s for s in rp.slots if s.name == "pattern_id")
        assert "oer_screening" in pattern_slot.options

    def test_round_4_shows_params_from_pattern(self):
        session = start_session()
        _advance_to_round(session.session_id, 4)
        rp = get_current_round(session.session_id)
        param_slot = next(s for s in rp.slots if s.name == "optimizable_params")
        assert param_slot.widget == "param_editor"
        # Should have params from oer_screening pattern
        assert param_slot.current_value is not None
        assert len(param_slot.current_value) > 0

    def test_round_5_is_marked_final(self):
        session = start_session()
        _advance_to_round(session.session_id, 5)
        rp = get_current_round(session.session_id)
        assert rp.is_final

    def test_validation_error_shows_in_slot_error(self):
        session = start_session()
        bad = {**ROUND_1_VALID, "direction": "invalid"}
        result = submit_round(session.session_id, bad)
        assert not result.success
        # The re-rendered round should have error on direction slot
        direction_slot = next(s for s in result.next_round.slots if s.name == "direction")
        assert direction_slot.error is not None


# ---------------------------------------------------------------------------
# Injection pack preview
# ---------------------------------------------------------------------------


class TestInjectionPackPreview:
    def test_preview_empty_at_start(self):
        session = start_session()
        result = submit_round(session.session_id, ROUND_1_VALID)
        preview = result.injection_pack_preview or {}
        assert "goal" in preview

    def test_preview_accumulates_across_rounds(self):
        session = start_session()
        submit_round(session.session_id, ROUND_1_VALID)
        result = submit_round(session.session_id, ROUND_2_VALID)
        preview = result.injection_pack_preview or {}
        assert "goal" in preview
        assert "safety" in preview


# ---------------------------------------------------------------------------
# Confirm and build
# ---------------------------------------------------------------------------


class TestConfirmAndBuild:
    def test_confirm_builds_complete_pack(self):
        session = start_session()
        _advance_to_round(session.session_id, 6)  # all 5 rounds
        pack = confirm_and_build(session.session_id)
        assert pack.goal.objective_kpi == "overpotential_mv"
        assert pack.goal.direction == "minimize"
        assert pack.protocol.pattern_id == "oer_screening"
        assert pack.safety.max_temp_c == 95.0
        assert pack.kpi_config.primary_kpi == "overpotential_mv"
        assert pack.human_gate.max_rounds == 20
        assert pack.metadata.checksum is not None
        assert len(pack.metadata.checksum) == 64  # SHA-256

    def test_confirm_marks_session_completed(self):
        session = start_session()
        _advance_to_round(session.session_id, 6)
        confirm_and_build(session.session_id)
        status = get_session_status(session.session_id)
        assert status.status == "completed"

    def test_confirm_incomplete_rounds_raises(self):
        session = start_session()
        submit_round(session.session_id, ROUND_1_VALID)
        with pytest.raises(ValueError, match="not completed"):
            confirm_and_build(session.session_id)

    def test_pack_has_dimensions_when_params_provided(self):
        """When the user sends optimizable_params from the param_editor,
        the pack should include them as dimensions."""
        session = start_session()
        _advance_to_round(session.session_id, 4)
        round4_with_params = {
            **ROUND_4_VALID,
            "optimizable_params": [
                {
                    "param_name": "precursor_ratio",
                    "param_type": "number",
                    "min_value": 0.1,
                    "max_value": 10.0,
                    "optimizable": True,
                    "safety_locked": False,
                    "step_key": "synthesis",
                    "primitive": "robot.aspirate",
                    "unit": "ratio",
                    "description": "Molar ratio",
                },
            ],
        }
        result = submit_round(session.session_id, round4_with_params)
        assert result.success
        submit_round(session.session_id, ROUND_5_VALID)
        pack = confirm_and_build(session.session_id)
        assert len(pack.param_space.dimensions) == 1
        assert pack.param_space.dimensions[0].param_name == "precursor_ratio"

    def test_pack_has_allowed_primitives(self):
        session = start_session()
        _advance_to_round(session.session_id, 6)
        pack = confirm_and_build(session.session_id)
        # ot2 + squidstat instruments → robot.* + squidstat.* primitives
        assert any(p.startswith("robot.") for p in pack.safety.allowed_primitives)
        assert any(p.startswith("squidstat.") for p in pack.safety.allowed_primitives)


# ---------------------------------------------------------------------------
# Reference data endpoints
# ---------------------------------------------------------------------------


class TestReferenceData:
    def test_get_all_kpis_returns_list(self):
        kpis = get_all_kpis()
        assert isinstance(kpis, list)
        assert len(kpis) > 0
        assert all("name" in k for k in kpis)

    def test_get_all_patterns_returns_list(self):
        patterns = get_all_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert any(p["id"] == "oer_screening" for p in patterns)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_round_1_applies_default_acceptable_range(self):
        session = start_session()
        minimal = {
            "objective_type": "oer_screening",
            "objective_kpi": "overpotential_mv",
            "direction": "minimize",
        }
        result = submit_round(session.session_id, minimal)
        assert result.success  # defaults applied

    def test_round_4_applies_default_strategy(self):
        session = start_session()
        _advance_to_round(session.session_id, 4)
        minimal = {}  # all defaults
        result = submit_round(session.session_id, minimal)
        assert result.success
