"""Tests for the CodeWriter agent."""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.agents.code_writer_agent import (
    CodeWriterAgent,
    CodeWriterInput,
    CodeWriterOutput,
    _ensure_ot2_agent_importable,
)


def _run(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Lightweight stand-ins that mirror ot2-nlp-agent dataclasses.
# Used by mock-based tests so we never need the real package installed.
# ---------------------------------------------------------------------------


@dataclass
class _FakeMissingInfo:
    parameter: str = "volume"
    question: str = "What volume?"
    question_zh: str = ""
    options: Optional[List[str]] = None
    default: Optional[Any] = None
    required: bool = True
    unit: Optional[str] = "uL"
    value_type: str = "number"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter,
            "question": self.question,
            "question_zh": self.question_zh,
            "options": self.options,
            "default": self.default,
            "required": self.required,
            "unit": self.unit,
            "value_type": self.value_type,
        }


@dataclass
class _FakeWorkflowDraft:
    name: str = "test-workflow"
    description: str = "A test workflow"
    description_zh: str = ""
    unit_operations: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    missing_info: list = field(default_factory=list)
    confidence: float = 0.85
    alternatives: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "description_zh": self.description_zh,
            "unit_operations": [],
            "assumptions": self.assumptions,
            "missing_info": [mi.to_dict() for mi in self.missing_info],
            "confidence": self.confidence,
            "alternatives": self.alternatives,
        }


@dataclass
class _FakePlannerOutput:
    candidates: list = field(default_factory=list)
    recommended_idx: int = 0


@dataclass
class _FakeDeviceAction:
    name: str = "pipette_transfer"
    description: str = "Transfer liquid"
    device_type: str = "liquid_handler"
    params: dict = field(default_factory=dict)
    requires_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "device_type": self.device_type,
            "params": self.params,
        }


@dataclass
class _FakeValidationResult:
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_count": 0,
            "warning_count": 0,
            "issues": [],
            "resource_conflicts": [],
            "topology_issues": [],
            "checkpoints": [],
        }


@dataclass
class _FakeCompilerOutput:
    python_code: str = "# OT-2 protocol\nfrom opentrons import protocol_api\n"
    workflow_json: str = '{"name": "test"}'
    validation_result: _FakeValidationResult = field(
        default_factory=_FakeValidationResult
    )
    device_actions: list = field(default_factory=list)
    primitives: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_validate_input_empty_intent(self):
        """Empty intent string must produce a validation error."""
        agent = CodeWriterAgent()
        errors = agent.validate_input(
            CodeWriterInput(intent="", context={})
        )
        assert any("non-empty" in e for e in errors)

    def test_validate_input_whitespace_intent(self):
        """Whitespace-only intent must produce a validation error."""
        agent = CodeWriterAgent()
        errors = agent.validate_input(
            CodeWriterInput(intent="   ", context={})
        )
        assert any("non-empty" in e for e in errors)

    def test_validate_input_valid(self):
        """A proper intent with ot2-nlp-agent available should pass."""
        agent = CodeWriterAgent()
        # The real ot2-nlp-agent is in our repo, so it should be importable
        # after _ensure_ot2_agent_importable() adds it to sys.path.
        if not _ensure_ot2_agent_importable():
            pytest.skip("ot2-nlp-agent not available")
        errors = agent.validate_input(
            CodeWriterInput(
                intent="Prepare a serial dilution of NaCl",
                context={"devices": ["OT-2"]},
            )
        )
        assert errors == []

    def test_validate_input_negative_candidate_idx(self):
        """candidate_idx < 0 must be rejected."""
        agent = CodeWriterAgent()
        errors = agent.validate_input(
            CodeWriterInput(
                intent="Do something",
                candidate_idx=-1,
            )
        )
        assert any("candidate_idx" in e for e in errors)


# ---------------------------------------------------------------------------
# Tests: plan_only mode (mocked)
# ---------------------------------------------------------------------------


class TestPlanOnlyMode:
    def test_plan_only_mode(self):
        """plan_only=True should call Planner but not Compiler."""
        agent = CodeWriterAgent()

        draft = _FakeWorkflowDraft(
            missing_info=[_FakeMissingInfo()],
        )
        fake_planner_output = _FakePlannerOutput(candidates=[draft])

        mock_planner_cls = MagicMock()
        mock_planner_instance = MagicMock()
        mock_planner_cls.return_value = mock_planner_instance
        mock_planner_instance.plan.return_value = fake_planner_output

        mock_compiler_cls = MagicMock()

        with patch(
            "app.agents.code_writer_agent._ensure_ot2_agent_importable",
            return_value=True,
        ), patch.dict("sys.modules", {
            "ot2_agent": MagicMock(),
            "ot2_agent.planner": MagicMock(
                Planner=mock_planner_cls,
                ConfirmedWorkflow=MagicMock,
            ),
            "ot2_agent.compiler": MagicMock(
                Compiler=mock_compiler_cls,
            ),
        }):
            input_data = CodeWriterInput(
                intent="Run a cyclic voltammetry experiment",
                context={"devices": ["OT-2"]},
                plan_only=True,
            )
            result = _run(agent.run(input_data))

        assert result.success
        output = result.output
        assert output is not None
        assert len(output.plan_candidates) == 1
        assert output.plan_candidates[0]["name"] == "test-workflow"
        assert output.python_code == ""  # no compilation
        assert len(output.missing_parameters) == 1
        assert output.missing_parameters[0]["parameter"] == "volume"

        # Compiler should NOT have been called
        mock_compiler_cls.return_value.compile.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: plan + compile (mocked)
# ---------------------------------------------------------------------------


class TestPlanAndCompile:
    def test_plan_and_compile(self):
        """Full flow: Planner -> Compiler with filled_parameters."""
        agent = CodeWriterAgent()

        draft = _FakeWorkflowDraft(
            missing_info=[_FakeMissingInfo()],
        )
        fake_planner_output = _FakePlannerOutput(candidates=[draft])

        fake_device_action = _FakeDeviceAction()
        fake_compiler_output = _FakeCompilerOutput(
            device_actions=[fake_device_action],
        )

        mock_planner_cls = MagicMock()
        mock_planner_instance = MagicMock()
        mock_planner_cls.return_value = mock_planner_instance
        mock_planner_instance.plan.return_value = fake_planner_output

        mock_compiler_cls = MagicMock()
        mock_compiler_instance = MagicMock()
        mock_compiler_cls.return_value = mock_compiler_instance
        mock_compiler_instance.compile.return_value = fake_compiler_output

        mock_confirmed_cls = MagicMock()

        with patch(
            "app.agents.code_writer_agent._ensure_ot2_agent_importable",
            return_value=True,
        ), patch.dict("sys.modules", {
            "ot2_agent": MagicMock(),
            "ot2_agent.planner": MagicMock(
                Planner=mock_planner_cls,
                ConfirmedWorkflow=mock_confirmed_cls,
            ),
            "ot2_agent.compiler": MagicMock(
                Compiler=mock_compiler_cls,
            ),
        }):
            input_data = CodeWriterInput(
                intent="Prepare serial dilution",
                context={"devices": ["OT-2"]},
                filled_parameters={"volume": 100},
            )
            result = _run(agent.run(input_data))

        assert result.success
        output = result.output
        assert output is not None

        # Plan candidates present
        assert len(output.plan_candidates) == 1

        # Compiled output present
        assert output.python_code == "# OT-2 protocol\nfrom opentrons import protocol_api\n"
        assert output.workflow_json == '{"name": "test"}'
        assert len(output.device_actions) == 1
        assert output.device_actions[0]["name"] == "pipette_transfer"

        # Compiler was called
        mock_compiler_instance.compile.assert_called_once()

    def test_compile_with_validation_errors(self):
        """Validation issues from the compiler should be surfaced."""
        agent = CodeWriterAgent()

        draft = _FakeWorkflowDraft()
        fake_planner_output = _FakePlannerOutput(candidates=[draft])

        vr = _FakeValidationResult(is_valid=False)
        # Inject issues into to_dict
        vr.to_dict = lambda: {
            "is_valid": False,
            "error_count": 1,
            "warning_count": 1,
            "issues": [
                {"severity": "ERROR", "message": "Missing labware", "step": 0},
                {"severity": "WARNING", "message": "Unusual volume", "step": 1},
            ],
            "resource_conflicts": [],
            "topology_issues": [],
            "checkpoints": [],
        }

        fake_compiler_output = _FakeCompilerOutput(
            validation_result=vr,
            python_code="# code with issues",
            workflow_json="{}",
        )

        mock_planner_cls = MagicMock()
        mock_planner_instance = MagicMock()
        mock_planner_cls.return_value = mock_planner_instance
        mock_planner_instance.plan.return_value = fake_planner_output

        mock_compiler_cls = MagicMock()
        mock_compiler_instance = MagicMock()
        mock_compiler_cls.return_value = mock_compiler_instance
        mock_compiler_instance.compile.return_value = fake_compiler_output

        with patch(
            "app.agents.code_writer_agent._ensure_ot2_agent_importable",
            return_value=True,
        ), patch.dict("sys.modules", {
            "ot2_agent": MagicMock(),
            "ot2_agent.planner": MagicMock(
                Planner=mock_planner_cls,
                ConfirmedWorkflow=MagicMock,
            ),
            "ot2_agent.compiler": MagicMock(
                Compiler=mock_compiler_cls,
            ),
        }):
            input_data = CodeWriterInput(
                intent="Test experiment",
                filled_parameters={"volume": 100},
            )
            result = _run(agent.run(input_data))

        assert result.success
        output = result.output
        assert "Missing labware" in output.validation_errors
        assert "Unusual volume" in output.validation_warnings


# ---------------------------------------------------------------------------
# Tests: import error handling
# ---------------------------------------------------------------------------


class TestImportErrorHandling:
    def test_import_error_handling(self):
        """When ot2-nlp-agent is not importable, validate_input should fail."""
        agent = CodeWriterAgent()

        with patch(
            "app.agents.code_writer_agent._ensure_ot2_agent_importable",
            return_value=False,
        ):
            errors = agent.validate_input(
                CodeWriterInput(
                    intent="Do something useful",
                    context={},
                )
            )

        assert any("ot2-nlp-agent" in e for e in errors)

    def test_import_error_via_run(self):
        """Running the agent when ot2-nlp-agent is absent should fail gracefully."""
        agent = CodeWriterAgent()

        with patch(
            "app.agents.code_writer_agent._ensure_ot2_agent_importable",
            return_value=False,
        ):
            result = _run(
                agent.run(
                    CodeWriterInput(intent="Test", context={})
                )
            )

        assert not result.success
        assert any("ot2-nlp-agent" in e for e in result.errors)
        assert result.agent_name == "code_writer_agent"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_candidate_idx_out_of_range_clamped(self):
        """candidate_idx beyond candidates list should be clamped."""
        agent = CodeWriterAgent()

        draft = _FakeWorkflowDraft()
        fake_planner_output = _FakePlannerOutput(candidates=[draft])

        mock_planner_cls = MagicMock()
        mock_planner_instance = MagicMock()
        mock_planner_cls.return_value = mock_planner_instance
        mock_planner_instance.plan.return_value = fake_planner_output

        with patch(
            "app.agents.code_writer_agent._ensure_ot2_agent_importable",
            return_value=True,
        ), patch.dict("sys.modules", {
            "ot2_agent": MagicMock(),
            "ot2_agent.planner": MagicMock(
                Planner=mock_planner_cls,
                ConfirmedWorkflow=MagicMock,
            ),
            "ot2_agent.compiler": MagicMock(),
        }):
            input_data = CodeWriterInput(
                intent="Something",
                candidate_idx=999,
                plan_only=True,
            )
            result = _run(agent.run(input_data))

        assert result.success
        assert result.output.selected_candidate_idx == 0  # clamped to max valid

    def test_agent_metadata(self):
        """Agent class attributes should be correct."""
        agent = CodeWriterAgent()
        assert agent.name == "code_writer_agent"
        assert agent.layer == "L1"
        assert "NL intent" in agent.description or "ot2-nlp-agent" in agent.description
