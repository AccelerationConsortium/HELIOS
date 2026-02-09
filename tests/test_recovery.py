"""Tests for the adaptive recovery engine (Phase C1).

Covers:
- RecoveryPolicy parsing from policy_snapshot
- Guard conditions (disabled, max attempts, no context)
- Recipe matching (error pattern, safety validation)
- Recipe execution (adapter calls, RunContext effects)
- Provenance event recording
- Hit count increment
- Graceful failure handling (memory errors, recipe step failures)
"""
from __future__ import annotations

import os
import tempfile
import uuid

# Isolate test DB before any app imports.
_tmpdir = tempfile.mkdtemp(prefix="otbot_recovery_test_")
os.environ["DATA_DIR"] = _tmpdir
os.environ["DB_PATH"] = os.path.join(_tmpdir, "recovery_test.db")
os.environ["OBJECT_STORE_DIR"] = os.path.join(_tmpdir, "obj")
os.environ["LLM_PROVIDER"] = "mock"

import pytest  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.core.db import connection, init_db, json_dumps, utcnow_iso  # noqa: E402
from app.services.action_contracts import SafetyClass  # noqa: E402
from app.services.memory import (  # noqa: E402
    RepairRecipe,
    increment_recipe_hit_count,
    seed_initial_recipes,
)
from app.services.recovery import (  # noqa: E402
    RecoveryPolicy,
    RecoveryResult,
    _NOT_ATTEMPTED,
    _get_safety_class,
    _validate_recipe_safety,
    attempt_recovery,
)
from app.services.run_context import RunContext  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_db():
    """Initialize a fresh DB for each test."""
    get_settings.cache_clear()
    init_db()
    # Clean memory tables between tests
    with connection() as conn:
        conn.execute("DELETE FROM memory_episodes")
        conn.execute("DELETE FROM memory_semantic")
        conn.execute("DELETE FROM memory_procedures")
        conn.execute("DELETE FROM provenance_events")
        conn.commit()


def _insert_seed_recipes() -> None:
    """Insert seed recipes for testing."""
    seed_initial_recipes()


def _make_adapter(side_effects: list | None = None) -> MagicMock:
    """Create a mock adapter."""
    adapter = MagicMock()
    if side_effects is not None:
        adapter.execute_primitive.side_effect = side_effects
    else:
        adapter.execute_primitive.return_value = {"ok": True}
    return adapter


def _make_recipe(
    trigger_primitive: str = "robot.aspirate",
    trigger_error_pattern: str = "tip",
    steps: list[dict] | None = None,
    source: str = "seed",
    hit_count: int = 0,
) -> RepairRecipe:
    """Create a RepairRecipe for testing."""
    if steps is None:
        steps = [
            {"primitive": "robot.drop_tip", "params": {}},
            {"primitive": "robot.pick_up_tip", "params": {}},
        ]
    return RepairRecipe(
        trigger_primitive=trigger_primitive,
        trigger_error_pattern=trigger_error_pattern,
        steps=steps,
        source=source,
        hit_count=hit_count,
    )


# ---------------------------------------------------------------------------
# RecoveryPolicy
# ---------------------------------------------------------------------------


class TestRecoveryPolicy:
    def test_defaults(self) -> None:
        policy = RecoveryPolicy()
        assert policy.enabled is True
        assert policy.max_attempts_per_step == 2

    def test_from_policy_snapshot_defaults(self) -> None:
        """Missing key → enabled=True, max=2."""
        policy = RecoveryPolicy.from_policy_snapshot({})
        assert policy.enabled is True
        assert policy.max_attempts_per_step == 2

    def test_from_policy_snapshot_explicit(self) -> None:
        """Explicit values parsed correctly."""
        policy = RecoveryPolicy.from_policy_snapshot({
            "recovery_policy": {
                "enabled": False,
                "max_attempts_per_step": 5,
            }
        })
        assert policy.enabled is False
        assert policy.max_attempts_per_step == 5

    def test_from_policy_snapshot_invalid_type(self) -> None:
        """Non-dict value → defaults."""
        policy = RecoveryPolicy.from_policy_snapshot({
            "recovery_policy": "invalid",
        })
        assert policy.enabled is True
        assert policy.max_attempts_per_step == 2


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------


class TestSafetyValidation:
    def test_validate_recipe_safety_allows_careful(self) -> None:
        """CAREFUL steps should be allowed in recipes."""
        steps = [
            {"primitive": "robot.drop_tip", "params": {}},  # CAREFUL
            {"primitive": "robot.pick_up_tip", "params": {}},  # CAREFUL
        ]
        assert _validate_recipe_safety(steps) is True

    def test_validate_recipe_safety_allows_reversible(self) -> None:
        """REVERSIBLE steps should be allowed."""
        steps = [
            {"primitive": "robot.blowout", "params": {}},  # REVERSIBLE
        ]
        assert _validate_recipe_safety(steps) is True

    def test_validate_recipe_safety_allows_informational(self) -> None:
        """INFORMATIONAL steps should be allowed."""
        steps = [
            {"primitive": "robot.home", "params": {}},  # INFORMATIONAL
        ]
        assert _validate_recipe_safety(steps) is True

    def test_validate_recipe_safety_blocks_hazardous(self) -> None:
        """HAZARDOUS steps should be rejected."""
        steps = [
            {"primitive": "robot.aspirate", "params": {}},  # HAZARDOUS
        ]
        assert _validate_recipe_safety(steps) is False

    def test_validate_recipe_safety_blocks_mixed(self) -> None:
        """Mixed recipe with one HAZARDOUS step should be rejected."""
        steps = [
            {"primitive": "robot.drop_tip", "params": {}},  # CAREFUL
            {"primitive": "robot.aspirate", "params": {}},  # HAZARDOUS
        ]
        assert _validate_recipe_safety(steps) is False

    def test_get_safety_class_known(self) -> None:
        """Known primitive returns correct class."""
        assert _get_safety_class("robot.aspirate") == SafetyClass.HAZARDOUS
        assert _get_safety_class("robot.drop_tip") == SafetyClass.CAREFUL
        assert _get_safety_class("robot.home") == SafetyClass.INFORMATIONAL

    def test_get_safety_class_unknown(self) -> None:
        """Unknown primitive defaults to CAREFUL."""
        assert _get_safety_class("totally.unknown") == SafetyClass.CAREFUL


# ---------------------------------------------------------------------------
# attempt_recovery — guard conditions
# ---------------------------------------------------------------------------


class TestRecoveryGuards:
    def test_recovery_disabled_by_policy(self) -> None:
        """When policy.enabled=False, should not attempt."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        with patch("app.services.recovery.get_repair_recipes") as mock_get:
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip not attached",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(enabled=False),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is False
        assert result.succeeded is False
        mock_get.assert_not_called()

    def test_recovery_max_attempts_exceeded(self) -> None:
        """When step already at max attempts, should not attempt."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts = {"s1": 2}  # already at max

        with patch("app.services.recovery.get_repair_recipes") as mock_get:
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip not attached",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(max_attempts_per_step=2),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is False
        mock_get.assert_not_called()

    def test_recovery_no_matching_recipe(self) -> None:
        """When no recipe matches the error, should not attempt."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        with patch("app.services.recovery.get_repair_recipes") as mock_get:
            mock_get.return_value = [_make_recipe(trigger_error_pattern="tip")]
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="completely unrelated error",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is False

    def test_recovery_memory_failure_graceful(self) -> None:
        """If get_repair_recipes raises, should gracefully not attempt."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        with patch("app.services.recovery.get_repair_recipes") as mock_get:
            mock_get.side_effect = RuntimeError("db locked")
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip not attached",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is False
        assert result.succeeded is False

    def test_recovery_no_recipes_for_primitive(self) -> None:
        """When no recipes exist for the primitive, return not attempted."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        with patch("app.services.recovery.get_repair_recipes") as mock_get:
            mock_get.return_value = []
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip not attached",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is False


# ---------------------------------------------------------------------------
# attempt_recovery — successful execution
# ---------------------------------------------------------------------------


class TestRecoveryExecution:
    @patch("app.services.recovery._record_recovery_event")
    def test_recovery_executes_recipe_steps(self, mock_event: MagicMock) -> None:
        """Recovery should call adapter for each recipe step in order."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        recipe = _make_recipe()

        with patch("app.services.recovery.get_repair_recipes", return_value=[recipe]), \
             patch("app.services.recovery._get_contract_for_step", return_value=None), \
             patch("app.services.recovery.evaluate_runtime_step") as mock_safety, \
             patch("app.services.recovery.increment_recipe_hit_count"):
            mock_safety.return_value = MagicMock(allowed=True, violations=[])
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip not attached",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is True
        assert result.succeeded is True
        assert result.recipe_used == "tip"
        assert result.steps_executed == 2
        assert result.error is None

        # Verify adapter called for both recipe steps
        assert adapter.execute_primitive.call_count == 2
        calls = adapter.execute_primitive.call_args_list
        assert calls[0].kwargs["primitive"] == "robot.drop_tip"
        assert calls[1].kwargs["primitive"] == "robot.pick_up_tip"

    @patch("app.services.recovery._record_recovery_event")
    def test_recovery_applies_effects(self, mock_event: MagicMock) -> None:
        """Recovery should apply contract effects to RunContext."""
        from app.services.action_contracts import ActionContract, Effect, TimeoutConfig

        adapter = _make_adapter()
        ctx = RunContext()
        ctx.tip_state["left"] = "on"  # Start with tip on
        counts: dict[str, int] = {}

        recipe = _make_recipe()

        # drop_tip effect: set tip_state to off
        drop_contract = ActionContract(
            effects=(Effect("set:tip_state:left:off"),),
            safety_class=SafetyClass.CAREFUL,
        )
        # pick_up_tip effect: set tip_state to on
        pickup_contract = ActionContract(
            effects=(Effect("set:tip_state:left:on"),),
            safety_class=SafetyClass.CAREFUL,
        )

        def mock_get_contract(primitive: str):
            if primitive == "robot.drop_tip":
                return drop_contract
            if primitive == "robot.pick_up_tip":
                return pickup_contract
            return None

        with patch("app.services.recovery.get_repair_recipes", return_value=[recipe]), \
             patch("app.services.recovery._get_contract_for_step", side_effect=mock_get_contract), \
             patch("app.services.recovery.evaluate_runtime_step") as mock_safety, \
             patch("app.services.recovery.increment_recipe_hit_count"):
            mock_safety.return_value = MagicMock(allowed=True, violations=[])
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip error detected",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.succeeded is True
        # After drop_tip → off, then pick_up_tip → on
        assert ctx.tip_state["left"] == "on"

    @patch("app.services.recovery._record_recovery_event")
    def test_recovery_increments_attempt_count(self, mock_event: MagicMock) -> None:
        """Recovery should increment the attempt counter."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        recipe = _make_recipe()

        with patch("app.services.recovery.get_repair_recipes", return_value=[recipe]), \
             patch("app.services.recovery._get_contract_for_step", return_value=None), \
             patch("app.services.recovery.evaluate_runtime_step") as mock_safety, \
             patch("app.services.recovery.increment_recipe_hit_count"):
            mock_safety.return_value = MagicMock(allowed=True, violations=[])
            attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip error",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert counts["s1"] == 1

    @patch("app.services.recovery._record_recovery_event")
    def test_recovery_records_provenance_events(self, mock_event: MagicMock) -> None:
        """Recovery should record provenance events for audit."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        recipe = _make_recipe()

        with patch("app.services.recovery.get_repair_recipes", return_value=[recipe]), \
             patch("app.services.recovery._get_contract_for_step", return_value=None), \
             patch("app.services.recovery.evaluate_runtime_step") as mock_safety, \
             patch("app.services.recovery.increment_recipe_hit_count"):
            mock_safety.return_value = MagicMock(allowed=True, violations=[])
            attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip not attached",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        # Should have: attempted, step_executed x2, succeeded
        actions = [call.args[1] for call in mock_event.call_args_list]
        assert "recovery.attempted" in actions
        assert actions.count("recovery.step_executed") == 2
        assert "recovery.succeeded" in actions

    @patch("app.services.recovery._record_recovery_event")
    def test_recovery_blocks_hazardous_recipe(self, mock_event: MagicMock) -> None:
        """Recipe with HAZARDOUS step should be skipped."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        # Recipe that includes an aspirate (HAZARDOUS)
        recipe = _make_recipe(steps=[
            {"primitive": "robot.aspirate", "params": {"volume": 100}},
        ])

        with patch("app.services.recovery.get_repair_recipes", return_value=[recipe]):
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip error",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is False
        adapter.execute_primitive.assert_not_called()

    @patch("app.services.recovery._record_recovery_event")
    def test_recovery_recipe_step_fails(self, mock_event: MagicMock) -> None:
        """If a recipe step fails mid-way, should return succeeded=False."""
        adapter = MagicMock()
        adapter.execute_primitive.side_effect = [
            {"ok": True},  # drop_tip succeeds
            RuntimeError("pick_up failed"),  # pick_up_tip fails
        ]
        ctx = RunContext()
        counts: dict[str, int] = {}

        recipe = _make_recipe()

        with patch("app.services.recovery.get_repair_recipes", return_value=[recipe]), \
             patch("app.services.recovery._get_contract_for_step", return_value=None), \
             patch("app.services.recovery.evaluate_runtime_step") as mock_safety:
            mock_safety.return_value = MagicMock(allowed=True, violations=[])
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip not attached",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is True
        assert result.succeeded is False
        assert result.steps_executed == 1  # Only drop_tip ran
        assert "pick_up failed" in result.error

    @patch("app.services.recovery._record_recovery_event")
    def test_recovery_safety_gate_blocks_step(self, mock_event: MagicMock) -> None:
        """If runtime safety gate blocks a recipe step, recovery fails."""
        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        recipe = _make_recipe()

        with patch("app.services.recovery.get_repair_recipes", return_value=[recipe]), \
             patch("app.services.recovery._get_contract_for_step", return_value=None), \
             patch("app.services.recovery.evaluate_runtime_step") as mock_safety:
            mock_safety.return_value = MagicMock(
                allowed=False,
                violations=["interlock failed"],
            )
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip error",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is True
        assert result.succeeded is False
        assert "safety gate blocked" in result.error


# ---------------------------------------------------------------------------
# increment_recipe_hit_count (DB integration)
# ---------------------------------------------------------------------------


class TestIncrementHitCount:
    def test_increment_hit_count(self) -> None:
        """Hit count should increment for existing recipe."""
        _insert_seed_recipes()

        # Get initial hit_count
        with connection() as conn:
            row = conn.execute(
                "SELECT hit_count FROM memory_procedures "
                "WHERE trigger_primitive = 'robot.aspirate' AND trigger_error_pattern = 'tip'",
            ).fetchone()
            assert row is not None
            initial = row["hit_count"]

        increment_recipe_hit_count("robot.aspirate", "tip")

        with connection() as conn:
            row = conn.execute(
                "SELECT hit_count FROM memory_procedures "
                "WHERE trigger_primitive = 'robot.aspirate' AND trigger_error_pattern = 'tip'",
            ).fetchone()
            assert row["hit_count"] == initial + 1

    def test_increment_hit_count_missing_recipe(self) -> None:
        """Incrementing nonexistent recipe should be a no-op."""
        # Should not raise
        increment_recipe_hit_count("nonexistent.primitive", "no_match")

    def test_increment_hit_count_idempotent(self) -> None:
        """Multiple increments should work."""
        _insert_seed_recipes()

        increment_recipe_hit_count("robot.aspirate", "tip")
        increment_recipe_hit_count("robot.aspirate", "tip")
        increment_recipe_hit_count("robot.aspirate", "tip")

        with connection() as conn:
            row = conn.execute(
                "SELECT hit_count FROM memory_procedures "
                "WHERE trigger_primitive = 'robot.aspirate' AND trigger_error_pattern = 'tip'",
            ).fetchone()
            assert row["hit_count"] == 3


# ---------------------------------------------------------------------------
# E2E: recovery with real DB recipes
# ---------------------------------------------------------------------------


class TestRecoveryE2E:
    """End-to-end tests using real DB seed recipes."""

    def test_recovery_with_seed_recipe(self) -> None:
        """Full E2E: seed recipe matches tip error → recovery succeeds."""
        _insert_seed_recipes()

        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        with patch("app.services.recovery.evaluate_runtime_step") as mock_safety, \
             patch("app.services.recovery._get_contract_for_step", return_value=None):
            mock_safety.return_value = MagicMock(allowed=True, violations=[])
            result = attempt_recovery(
                primitive="robot.aspirate",
                error_text="tip not attached to pipette",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is True
        assert result.succeeded is True
        assert result.recipe_used == "tip"
        assert result.steps_executed == 2

        # Verify hit count incremented
        with connection() as conn:
            row = conn.execute(
                "SELECT hit_count FROM memory_procedures "
                "WHERE trigger_primitive = 'robot.aspirate' AND trigger_error_pattern = 'tip'",
            ).fetchone()
            assert row["hit_count"] == 1

    def test_recovery_dispense_tip_error(self) -> None:
        """Seed recipe for robot.dispense tip error should also work."""
        _insert_seed_recipes()

        adapter = _make_adapter()
        ctx = RunContext()
        counts: dict[str, int] = {}

        with patch("app.services.recovery.evaluate_runtime_step") as mock_safety, \
             patch("app.services.recovery._get_contract_for_step", return_value=None):
            mock_safety.return_value = MagicMock(allowed=True, violations=[])
            result = attempt_recovery(
                primitive="robot.dispense",
                error_text="tip fell off during dispense",
                run_id="run-1",
                instrument_id="sim-1",
                adapter=adapter,
                run_context=ctx,
                policy_snapshot={},
                step_key="s1",
                recovery_policy=RecoveryPolicy(),
                recovery_attempt_counts=counts,
            )

        assert result.attempted is True
        assert result.succeeded is True
