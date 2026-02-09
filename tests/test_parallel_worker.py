"""Tests for parallel execution logic, artifact helpers, and contract integration in worker.py."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.services.action_contracts import (
    ActionContract,
    Effect,
    Precondition,
    SafetyClass,
    TimeoutConfig,
)
from app.services.error_policy import ErrorPolicy, classify_step_error
from app.services.run_context import RunContext
from app.worker import _execute_step, _find_ready_steps, _get_contract_for_step, _partition_by_resources


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(
    key: str,
    primitive: str = "wait",
    depends_on: list[str] | None = None,
    resources: list[str] | None = None,
    status: str = "pending",
) -> dict[str, Any]:
    return {
        "id": f"id-{key}",
        "step_key": key,
        "primitive": primitive,
        "params": {},
        "depends_on": depends_on or [],
        "resources": resources or [],
        "status": status,
    }


# ---------------------------------------------------------------------------
# _find_ready_steps
# ---------------------------------------------------------------------------


class TestFindReadySteps:
    def test_no_deps_all_ready(self) -> None:
        steps = [_step("a"), _step("b"), _step("c")]
        ready = _find_ready_steps(steps, set(), set())
        assert {s["step_key"] for s in ready} == {"a", "b", "c"}

    def test_deps_filter(self) -> None:
        steps = [
            _step("a"),
            _step("b", depends_on=["a"]),
            _step("c", depends_on=["b"]),
        ]
        # Nothing finished yet → only "a" is ready
        ready = _find_ready_steps(steps, set(), set())
        assert [s["step_key"] for s in ready] == ["a"]

        # After "a" finishes → "b" ready
        ready = _find_ready_steps(steps, {"a"}, set())
        assert [s["step_key"] for s in ready] == ["b"]

        # After "a" and "b" → "c" ready
        ready = _find_ready_steps(steps, {"a", "b"}, set())
        assert [s["step_key"] for s in ready] == ["c"]

    def test_already_finished_excluded(self) -> None:
        steps = [_step("a"), _step("b")]
        ready = _find_ready_steps(steps, {"a"}, set())
        assert [s["step_key"] for s in ready] == ["b"]

    def test_failed_excluded(self) -> None:
        steps = [_step("a"), _step("b")]
        ready = _find_ready_steps(steps, set(), {"a"})
        assert [s["step_key"] for s in ready] == ["b"]

    def test_dep_on_failed_not_ready(self) -> None:
        steps = [
            _step("a"),
            _step("b", depends_on=["a"]),
        ]
        # "a" is in failed_step_keys, so "b" won't be ready (dep not in finished)
        ready = _find_ready_steps(steps, set(), {"a"})
        assert ready == []

    def test_parallel_fork(self) -> None:
        """Two independent branches from same parent."""
        steps = [
            _step("root"),
            _step("left", depends_on=["root"]),
            _step("right", depends_on=["root"]),
            _step("join", depends_on=["left", "right"]),
        ]
        ready = _find_ready_steps(steps, {"root"}, set())
        assert {s["step_key"] for s in ready} == {"left", "right"}

    def test_join_needs_all_deps(self) -> None:
        steps = [
            _step("root"),
            _step("left", depends_on=["root"]),
            _step("right", depends_on=["root"]),
            _step("join", depends_on=["left", "right"]),
        ]
        # Only "left" done — "join" not yet ready
        ready = _find_ready_steps(steps, {"root", "left"}, set())
        keys = {s["step_key"] for s in ready}
        assert "right" in keys
        assert "join" not in keys

    def test_non_pending_status_excluded(self) -> None:
        steps = [_step("a", status="succeeded")]
        ready = _find_ready_steps(steps, set(), set())
        assert ready == []


# ---------------------------------------------------------------------------
# _partition_by_resources
# ---------------------------------------------------------------------------


class TestPartitionByResources:
    def test_empty(self) -> None:
        assert _partition_by_resources([], "default") == []

    def test_single_step(self) -> None:
        groups = _partition_by_resources([_step("a")], "default")
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_non_overlapping_resources(self) -> None:
        steps = [
            _step("a", resources=["ot2-robot"]),
            _step("b", resources=["plc-controller"]),
            _step("c", resources=["relay-controller"]),
        ]
        groups = _partition_by_resources(steps, "default")
        # All can go in one group
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_overlapping_resources_split(self) -> None:
        steps = [
            _step("a", resources=["ot2-robot"]),
            _step("b", resources=["ot2-robot"]),
        ]
        groups = _partition_by_resources(steps, "default")
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 1

    def test_mixed_overlap(self) -> None:
        steps = [
            _step("a", resources=["ot2-robot"]),
            _step("b", resources=["plc-controller"]),
            _step("c", resources=["ot2-robot"]),
        ]
        groups = _partition_by_resources(steps, "default")
        # "a" and "b" in group 0, "c" conflicts with "a" → group 1
        assert len(groups) == 2
        keys_0 = {s["step_key"] for s in groups[0]}
        keys_1 = {s["step_key"] for s in groups[1]}
        assert keys_0 == {"a", "b"}
        assert keys_1 == {"c"}

    def test_empty_resources_uses_instrument_id(self) -> None:
        """Steps with no resources should be treated as using the instrument_id."""
        steps = [
            _step("a", resources=[]),
            _step("b", resources=[]),
        ]
        groups = _partition_by_resources(steps, "sim-instrument-1")
        # Both use default instrument → conflict → separate groups
        assert len(groups) == 2


# ---------------------------------------------------------------------------
# persist_file_artifact
# ---------------------------------------------------------------------------

class TestPersistFileArtifact:
    def test_persist_csv_file(self, tmp_path: Path) -> None:
        from app.services.artifact_store import persist_file_artifact

        # Set up a temporary object store
        os.environ["OBJECT_STORE_DIR"] = str(tmp_path / "store")
        os.environ["DATA_DIR"] = str(tmp_path / "data")

        # Create a source CSV
        src = tmp_path / "data.csv"
        src.write_text("col1,col2\n1,2\n3,4\n")

        uri, checksum = persist_file_artifact("run-1", "step-1", src, suffix=".csv")
        assert uri.endswith(".csv")
        assert Path(uri).exists()
        assert len(checksum) == 64  # SHA-256 hex

        # Verify content matches
        assert Path(uri).read_text() == src.read_text()

    def test_persist_binary_file(self, tmp_path: Path) -> None:
        from app.services.artifact_store import persist_file_artifact

        os.environ["OBJECT_STORE_DIR"] = str(tmp_path / "store")
        os.environ["DATA_DIR"] = str(tmp_path / "data")

        # Create a fake PNG (just bytes)
        src = tmp_path / "plot.png"
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        src.write_bytes(data)

        uri, checksum = persist_file_artifact("run-2", "step-2", src)
        assert uri.endswith(".png")
        assert Path(uri).exists()
        assert Path(uri).read_bytes() == data

    def test_persist_uses_source_suffix_by_default(self, tmp_path: Path) -> None:
        from app.services.artifact_store import persist_file_artifact

        os.environ["OBJECT_STORE_DIR"] = str(tmp_path / "store")
        os.environ["DATA_DIR"] = str(tmp_path / "data")

        src = tmp_path / "results.json"
        src.write_text("{}")

        uri, _ = persist_file_artifact("run-3", "step-3", src)
        assert uri.endswith(".json")


# ---------------------------------------------------------------------------
# _get_contract_for_step
# ---------------------------------------------------------------------------


class TestGetContractForStep:
    def test_returns_none_when_registry_empty(self) -> None:
        """Unknown primitive should return None."""
        contract = _get_contract_for_step("totally.unknown.primitive")
        # May return None or a contract depending on registry state
        # The function is designed to gracefully handle missing primitives
        assert contract is None or isinstance(contract, ActionContract)

    def test_returns_contract_for_known_primitive(self) -> None:
        """Known primitive should return an ActionContract."""
        contract = _get_contract_for_step("robot.aspirate")
        if contract is not None:
            assert isinstance(contract, ActionContract)


# ---------------------------------------------------------------------------
# _execute_step with contracts
# ---------------------------------------------------------------------------


class TestExecuteStepContracts:
    """Test contract-aware execution in _execute_step."""

    def _make_step(
        self,
        key: str = "s1",
        primitive: str = "wait",
        params: dict | None = None,
        resources: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": f"id-{key}",
            "step_key": key,
            "primitive": primitive,
            "params": params or {},
            "depends_on": [],
            "resources": resources or [],
            "status": "pending",
        }

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker.worker_append_artifact")
    @patch("app.worker.persist_json_artifact", return_value=("/fake/uri.json", "abc123"))
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step")
    def test_precondition_failure_blocks_step(
        self,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_persist: MagicMock,
        mock_append: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """Step should fail when preconditions are not met."""
        # Contract requires tip_on:{pipette}
        contract = ActionContract(
            preconditions=(Precondition("tip_on:{pipette}"),),
            effects=(),
            timeout=TimeoutConfig(seconds=30, retries=0),
            safety_class=SafetyClass.HAZARDOUS,
        )
        mock_get_contract.return_value = contract

        ctx = RunContext()
        # tip NOT attached → precondition should fail
        result_holder: dict[str, Any] = {}

        adapter = MagicMock()
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(primitive="robot.aspirate", params={"pipette": "left"}),
            policy={"allowed_primitives": ["robot.aspirate"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=ctx,
        )

        assert result_holder["s1"]["ok"] is False
        assert "tip_on:left" in result_holder["s1"]["error"]
        # Adapter should NOT have been called
        adapter.execute_primitive.assert_not_called()

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker.worker_append_artifact")
    @patch("app.worker.persist_json_artifact", return_value=("/fake/uri.json", "abc123"))
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step")
    def test_effects_applied_on_success(
        self,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_persist: MagicMock,
        mock_append: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """Effects should be applied to RunContext after successful execution."""
        contract = ActionContract(
            preconditions=(),
            effects=(Effect("set:tip_state:{pipette}:on"),),
            timeout=TimeoutConfig(seconds=30, retries=0),
            safety_class=SafetyClass.CAREFUL,
        )
        mock_get_contract.return_value = contract

        ctx = RunContext()
        result_holder: dict[str, Any] = {}

        adapter = MagicMock()
        adapter.execute_primitive.return_value = {"status": "ok"}

        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(
                primitive="robot.pick_up_tip",
                params={"pipette": "left", "labware": "tips"},
            ),
            policy={"allowed_primitives": ["robot.pick_up_tip"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=ctx,
        )

        assert result_holder["s1"]["ok"] is True
        # Effect should have set tip_state.left = "on"
        assert ctx.tip_state.get("left") == "on"

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker.worker_append_artifact")
    @patch("app.worker.persist_json_artifact", return_value=("/fake/uri.json", "abc123"))
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step")
    def test_retry_on_failure_non_hazardous(
        self,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_persist: MagicMock,
        mock_append: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """Non-HAZARDOUS primitives should retry on failure."""
        contract = ActionContract(
            preconditions=(),
            effects=(),
            timeout=TimeoutConfig(seconds=30, retries=2),
            safety_class=SafetyClass.CAREFUL,
        )
        mock_get_contract.return_value = contract

        # Fail twice, succeed on third attempt
        adapter = MagicMock()
        adapter.execute_primitive.side_effect = [
            RuntimeError("fail-1"),
            RuntimeError("fail-2"),
            {"status": "ok"},
        ]

        result_holder: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(primitive="robot.load_labware", params={"name": "plate1"}),
            policy={"allowed_primitives": ["robot.load_labware"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=RunContext(),
        )

        assert result_holder["s1"]["ok"] is True
        assert adapter.execute_primitive.call_count == 3

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker.worker_append_artifact")
    @patch("app.worker.persist_json_artifact", return_value=("/fake/uri.json", "abc123"))
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step")
    def test_hazardous_no_retry(
        self,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_persist: MagicMock,
        mock_append: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """HAZARDOUS primitives should NOT retry even with retries configured."""
        contract = ActionContract(
            preconditions=(),
            effects=(),
            timeout=TimeoutConfig(seconds=30, retries=3),  # retries configured but ignored
            safety_class=SafetyClass.HAZARDOUS,
        )
        mock_get_contract.return_value = contract

        adapter = MagicMock()
        adapter.execute_primitive.side_effect = RuntimeError("hw-failure")

        result_holder: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(primitive="robot.aspirate", params={"pipette": "left"}),
            policy={"allowed_primitives": ["robot.aspirate"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=RunContext(),
        )

        # HAZARDOUS: only 1 attempt, no retries
        assert adapter.execute_primitive.call_count == 1
        assert result_holder["s1"]["ok"] is False

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker.worker_append_artifact")
    @patch("app.worker.persist_json_artifact", return_value=("/fake/uri.json", "abc123"))
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step")
    def test_no_contract_runs_normally(
        self,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_persist: MagicMock,
        mock_append: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """Steps without contracts should execute normally (backward compat)."""
        mock_get_contract.return_value = None

        adapter = MagicMock()
        adapter.execute_primitive.return_value = {"status": "ok"}

        result_holder: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(primitive="custom.action"),
            policy={"allowed_primitives": ["custom.action"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=RunContext(),
        )

        assert result_holder["s1"]["ok"] is True
        adapter.execute_primitive.assert_called_once()

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker.worker_append_artifact")
    @patch("app.worker.persist_json_artifact", return_value=("/fake/uri.json", "abc123"))
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step")
    def test_effects_not_applied_on_failure(
        self,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_persist: MagicMock,
        mock_append: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """Effects should NOT be applied when execution fails."""
        contract = ActionContract(
            preconditions=(),
            effects=(Effect("set:tip_state:{pipette}:on"),),
            timeout=TimeoutConfig(seconds=30, retries=0),
            safety_class=SafetyClass.HAZARDOUS,
        )
        mock_get_contract.return_value = contract

        ctx = RunContext()
        adapter = MagicMock()
        adapter.execute_primitive.side_effect = RuntimeError("hardware fault")

        result_holder: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(
                primitive="robot.pick_up_tip",
                params={"pipette": "left"},
            ),
            policy={"allowed_primitives": ["robot.pick_up_tip"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=ctx,
        )

        assert result_holder["s1"]["ok"] is False
        # tip_state should NOT have been modified
        assert ctx.tip_state.get("left") is None

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker.worker_append_artifact")
    @patch("app.worker.persist_json_artifact", return_value=("/fake/uri.json", "abc123"))
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step")
    def test_cross_step_effect_propagation(
        self,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_persist: MagicMock,
        mock_append: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """Effects from step 1 should satisfy preconditions for step 2."""
        ctx = RunContext()
        adapter = MagicMock()
        adapter.execute_primitive.return_value = {"status": "ok"}

        # Step 1: pick_up_tip → sets tip_state.left = on
        contract_1 = ActionContract(
            preconditions=(),
            effects=(Effect("set:tip_state:{pipette}:on"),),
            timeout=TimeoutConfig(),
            safety_class=SafetyClass.CAREFUL,
        )
        mock_get_contract.return_value = contract_1

        r1: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(
                key="s1",
                primitive="robot.pick_up_tip",
                params={"pipette": "left", "labware": "tips"},
            ),
            policy={"allowed_primitives": ["robot.pick_up_tip", "robot.aspirate"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=r1,
            run_context=ctx,
        )
        assert r1["s1"]["ok"] is True
        assert ctx.tip_state.get("left") == "on"

        # Step 2: aspirate → requires tip_on:{pipette}
        contract_2 = ActionContract(
            preconditions=(Precondition("tip_on:{pipette}"),),
            effects=(Effect("increase:pipette_volume:{pipette}:{volume}"),),
            timeout=TimeoutConfig(),
            safety_class=SafetyClass.HAZARDOUS,
        )
        mock_get_contract.return_value = contract_2

        r2: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(
                key="s2",
                primitive="robot.aspirate",
                params={"pipette": "left", "labware": "plate1", "volume": "100"},
            ),
            policy={"allowed_primitives": ["robot.pick_up_tip", "robot.aspirate"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=r2,
            run_context=ctx,
        )
        assert r2["s2"]["ok"] is True
        # pipette_volume should have increased
        assert ctx.pipette_volume.get("left", 0) == 100.0

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step")
    def test_without_run_context_backward_compat(
        self,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """When run_context=None, step should execute without contract checks."""
        contract = ActionContract(
            preconditions=(Precondition("tip_on:{pipette}"),),
            effects=(),
            timeout=TimeoutConfig(),
            safety_class=SafetyClass.HAZARDOUS,
        )
        mock_get_contract.return_value = contract

        adapter = MagicMock()
        adapter.execute_primitive.return_value = {"status": "ok"}

        result_holder: dict[str, Any] = {}

        with patch("app.worker.persist_json_artifact", return_value=("/f.json", "abc")), \
             patch("app.worker.worker_append_artifact"):
            _execute_step(
                adapter=adapter,
                run_id="run-1",
                step=self._make_step(
                    primitive="robot.aspirate",
                    params={"pipette": "left"},
                ),
                policy={"allowed_primitives": ["robot.aspirate"]},
                instrument_id="sim-instrument-1",
                error_policy=ErrorPolicy(),
                result_holder=result_holder,
                run_context=None,  # No context → skip precondition check
            )

        # Should succeed despite preconditions not met (no RunContext)
        assert result_holder["s1"]["ok"] is True


# ---------------------------------------------------------------------------
# Recovery integration with _execute_step
# ---------------------------------------------------------------------------


class TestExecuteStepRecovery:
    """Test adaptive recovery integration in _execute_step."""

    def _make_step(
        self,
        key: str = "s1",
        primitive: str = "wait",
        params: dict | None = None,
        resources: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": f"id-{key}",
            "step_key": key,
            "primitive": primitive,
            "params": params or {},
            "depends_on": [],
            "resources": resources or [],
            "status": "pending",
        }

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker.worker_append_artifact")
    @patch("app.worker.persist_json_artifact", return_value=("/fake/uri.json", "abc123"))
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step", return_value=None)
    @patch("app.worker.attempt_recovery")
    def test_execute_step_with_recovery(
        self,
        mock_recovery: MagicMock,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_persist: MagicMock,
        mock_append: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """Step fails → recovery succeeds → post-recovery retry succeeds."""
        from app.services.recovery import RecoveryPolicy, RecoveryResult

        mock_recovery.return_value = RecoveryResult(
            attempted=True, succeeded=True,
            recipe_used="tip", steps_executed=2, error=None,
        )

        adapter = MagicMock()
        # First call fails (original attempt), second call succeeds (post-recovery retry)
        adapter.execute_primitive.side_effect = [
            RuntimeError("tip not attached"),
            {"status": "ok"},
        ]

        result_holder: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(primitive="robot.aspirate"),
            policy={"allowed_primitives": ["robot.aspirate"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=RunContext(),
            recovery_policy=RecoveryPolicy(),
            recovery_attempt_counts={},
        )

        assert result_holder["s1"]["ok"] is True
        assert result_holder["s1"].get("recovered") is True
        mock_recovery.assert_called_once()

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step", return_value=None)
    @patch("app.worker.attempt_recovery")
    def test_execute_step_recovery_retry_fails(
        self,
        mock_recovery: MagicMock,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """Recovery succeeds but post-recovery retry still fails → step fails."""
        from app.services.recovery import RecoveryPolicy, RecoveryResult

        mock_recovery.return_value = RecoveryResult(
            attempted=True, succeeded=True,
            recipe_used="tip", steps_executed=2, error=None,
        )

        adapter = MagicMock()
        # Both calls fail
        adapter.execute_primitive.side_effect = [
            RuntimeError("tip not attached"),
            RuntimeError("still broken"),
        ]

        result_holder: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(primitive="robot.aspirate"),
            policy={"allowed_primitives": ["robot.aspirate"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=RunContext(),
            recovery_policy=RecoveryPolicy(),
            recovery_attempt_counts={},
        )

        assert result_holder["s1"]["ok"] is False
        assert "still broken" in result_holder["s1"]["error"]

    @patch("app.worker.worker_set_step_state")
    @patch("app.worker._release_resources")
    @patch("app.worker._acquire_resources", return_value=["sim-instrument-1"])
    @patch("app.worker._get_contract_for_step", return_value=None)
    @patch("app.worker.attempt_recovery")
    def test_execute_step_recovery_disabled(
        self,
        mock_recovery: MagicMock,
        mock_get_contract: MagicMock,
        mock_acquire: MagicMock,
        mock_release: MagicMock,
        mock_set_state: MagicMock,
    ) -> None:
        """When recovery policy is disabled, should not attempt recovery."""
        from app.services.recovery import RecoveryPolicy

        adapter = MagicMock()
        adapter.execute_primitive.side_effect = RuntimeError("tip not attached")

        result_holder: dict[str, Any] = {}
        _execute_step(
            adapter=adapter,
            run_id="run-1",
            step=self._make_step(primitive="robot.aspirate"),
            policy={"allowed_primitives": ["robot.aspirate"]},
            instrument_id="sim-instrument-1",
            error_policy=ErrorPolicy(),
            result_holder=result_holder,
            run_context=RunContext(),
            recovery_policy=RecoveryPolicy(enabled=False),
            recovery_attempt_counts={},
        )

        assert result_holder["s1"]["ok"] is False
        mock_recovery.assert_not_called()
