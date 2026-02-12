"""Tests for the InstrumentOnboardingService and OnboardingAgent.

Covers:
- InstrumentSpec data models
- Safety classification inference
- Communication confirmation builders
- KPI confirmation builders
- All 7 file generators
- Confirmation flow (generate → confirm → write)
- format_confirmations_for_chat()
- OnboardingAgent (generate/confirm/write phases)
- Serialisation round-trip
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

from app.services.instrument_onboarding import (
    CommunicationType,
    ConfirmationItem,
    ConfirmationType,
    GeneratedFile,
    InstrumentOnboardingService,
    InstrumentSpec,
    OnboardingResult,
    ParamInput,
    PrimitiveInput,
    _count_safety_classes,
    _default_port,
    _guess_kpi_name,
    _python_default,
    _python_type,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(
    name: str = "uv_vis",
    communication: CommunicationType = CommunicationType.USB,
    n_primitives: int = 2,
) -> InstrumentSpec:
    """Build a sample InstrumentSpec for testing."""
    primitives = []

    if n_primitives >= 1:
        primitives.append(PrimitiveInput(
            name="measure_spectrum",
            description="Capture UV-Vis absorption spectrum",
            params={
                "wavelength_start_nm": ParamInput(type="number", default=200),
                "wavelength_end_nm": ParamInput(type="number", default=800),
                "integration_time_ms": ParamInput(type="integer", default=100),
            },
            hazardous=False,
            generates_data=True,
        ))

    if n_primitives >= 2:
        primitives.append(PrimitiveInput(
            name="set_lamp",
            description="Turn UV-Vis lamp on or off",
            params={
                "state": ParamInput(type="boolean", default=True, description="Lamp on/off"),
            },
            hazardous=True,
            generates_data=False,
        ))

    return InstrumentSpec(
        name=name,
        manufacturer="Ocean Insight",
        model="Flame-S",
        communication=communication,
        description="UV-Vis spectrometer for absorbance measurements",
        primitives=primitives,
        sdk_package="seabreeze",
    )


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestInstrumentSpec:
    def test_prefix(self):
        spec = _make_spec(name="uv_vis")
        assert spec.prefix == "uv_vis"

    def test_class_name(self):
        spec = _make_spec(name="uv_vis")
        assert spec.class_name == "UvVisController"

    def test_class_name_single_word(self):
        spec = _make_spec(name="furnace")
        assert spec.class_name == "FurnaceController"

    def test_display_name_with_manufacturer(self):
        spec = _make_spec()
        assert spec.display_name == "Ocean Insight Flame-S"

    def test_display_name_without_manufacturer(self):
        spec = InstrumentSpec(name="furnace")
        assert "Furnace" in spec.display_name

    def test_resource_id_default(self):
        spec = _make_spec()
        svc = InstrumentOnboardingService()
        result = svc.generate(spec)
        assert result.spec.resource_id == "uv_vis"


class TestParamInput:
    def test_defaults(self):
        p = ParamInput()
        assert p.type == "number"
        assert p.description == ""
        assert p.default is None
        assert p.optional is False


class TestPrimitiveInput:
    def test_defaults(self):
        p = PrimitiveInput(name="test")
        assert p.hazardous is False
        assert p.generates_data is False
        assert p.error_class == ""
        assert p.safety_class == ""
        assert p.timeout_seconds == 30
        assert p.retries == 1


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_python_type_mapping(self):
        assert _python_type("string") == "str"
        assert _python_type("number") == "float"
        assert _python_type("integer") == "int"
        assert _python_type("boolean") == "bool"
        assert _python_type("array") == "list"
        assert _python_type("unknown") == "Any"

    def test_python_default_string(self):
        assert _python_default("hello") == '"hello"'

    def test_python_default_bool(self):
        assert _python_default(True) == "True"
        assert _python_default(False) == "False"

    def test_python_default_number(self):
        assert _python_default(42) == "42"
        assert _python_default(3.14) == "3.14"

    def test_default_port_usb(self):
        assert _default_port(CommunicationType.USB) == "/dev/ttyUSB0"

    def test_default_port_serial(self):
        assert _default_port(CommunicationType.SERIAL) == "/dev/ttyS0"

    def test_default_port_tcp(self):
        assert _default_port(CommunicationType.TCP) == ""

    def test_guess_kpi_spectrum(self):
        p = PrimitiveInput(name="measure_spectrum")
        assert _guess_kpi_name(p) == "peak_absorbance"

    def test_guess_kpi_impedance(self):
        p = PrimitiveInput(name="run_eis")
        assert _guess_kpi_name(p) == "impedance_ohm"

    def test_guess_kpi_current(self):
        p = PrimitiveInput(name="run_cv")
        assert _guess_kpi_name(p) == "peak_current_ma"

    def test_guess_kpi_temperature(self):
        p = PrimitiveInput(name="read_temperature")
        assert _guess_kpi_name(p) == "temperature_c"

    def test_guess_kpi_generic(self):
        p = PrimitiveInput(name="do_something")
        assert _guess_kpi_name(p) == "do_something_value"

    def test_count_safety_classes(self):
        primitives = [
            PrimitiveInput(name="a", safety_class="HAZARDOUS"),
            PrimitiveInput(name="b", safety_class="CAREFUL"),
            PrimitiveInput(name="c", safety_class="HAZARDOUS"),
        ]
        counts = _count_safety_classes(primitives)
        assert counts["HAZARDOUS"] == 2
        assert counts["CAREFUL"] == 1

    def test_count_safety_classes_empty(self):
        counts = _count_safety_classes([])
        assert counts == {}

    def test_count_safety_classes_defaults_to_reversible(self):
        primitives = [PrimitiveInput(name="a")]
        counts = _count_safety_classes(primitives)
        assert counts.get("REVERSIBLE", 0) == 1


# ---------------------------------------------------------------------------
# Service tests — generate
# ---------------------------------------------------------------------------


class TestServiceGenerate:
    def setup_method(self):
        self.svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        self.spec = _make_spec()

    def test_generate_returns_result(self):
        result = self.svc.generate(self.spec)
        assert isinstance(result, OnboardingResult)
        assert result.spec.name == "uv_vis"

    def test_generate_creates_7_files(self):
        result = self.svc.generate(self.spec)
        assert len(result.files) == 7

    def test_generate_file_paths(self):
        result = self.svc.generate(self.spec)
        paths = [f.path for f in result.files]
        assert "app/hardware/uv_vis_controller.py" in paths
        assert "agent/skills/uv_vis.md" in paths
        assert "tests/test_uv_vis_onboarded.py" in paths

    def test_generate_has_patch_files(self):
        result = self.svc.generate(self.spec)
        patches = [f for f in result.files if f.is_patch]
        assert len(patches) == 4  # dispatcher, simulated, adapter, dry-run

    def test_generate_creates_confirmations(self):
        result = self.svc.generate(self.spec)
        assert len(result.pending_confirmations) > 0

    def test_generate_has_safety_confirmations(self):
        result = self.svc.generate(self.spec)
        safety = [
            c for c in result.pending_confirmations
            if c.type == ConfirmationType.SAFETY_CLASSIFICATION
        ]
        # 2 primitives × 2 confirmations (safety_class + error_class) = 4
        assert len(safety) == 4

    def test_generate_has_kpi_confirmations(self):
        result = self.svc.generate(self.spec)
        kpi = [
            c for c in result.pending_confirmations
            if c.type == ConfirmationType.KPI_EXTRACTION
        ]
        # Only measure_spectrum generates data
        assert len(kpi) == 1

    def test_generate_has_communication_confirmations_usb(self):
        result = self.svc.generate(self.spec)
        comm = [
            c for c in result.pending_confirmations
            if c.type == ConfirmationType.COMMUNICATION_DETAILS
        ]
        # USB → port + baudrate
        assert len(comm) == 2

    def test_generate_communication_tcp(self):
        spec = _make_spec(communication=CommunicationType.TCP)
        result = self.svc.generate(spec)
        comm = [
            c for c in result.pending_confirmations
            if c.type == ConfirmationType.COMMUNICATION_DETAILS
        ]
        # TCP → host + port_number
        assert len(comm) == 2
        assert any("host" in c.id for c in comm)

    def test_generate_communication_modbus(self):
        spec = _make_spec(communication=CommunicationType.MODBUS)
        result = self.svc.generate(spec)
        comm = [
            c for c in result.pending_confirmations
            if c.type == ConfirmationType.COMMUNICATION_DETAILS
        ]
        assert len(comm) == 2
        # Modbus default port should be 502
        port_conf = [c for c in comm if "port_number" in c.id]
        assert len(port_conf) == 1
        assert port_conf[0].current_value == 502

    def test_generate_communication_gpib_no_extra(self):
        spec = _make_spec(communication=CommunicationType.GPIB)
        result = self.svc.generate(spec)
        comm = [
            c for c in result.pending_confirmations
            if c.type == ConfirmationType.COMMUNICATION_DETAILS
        ]
        assert len(comm) == 0

    def test_generate_communication_simulated_no_extra(self):
        spec = _make_spec(communication=CommunicationType.SIMULATED)
        result = self.svc.generate(spec)
        comm = [
            c for c in result.pending_confirmations
            if c.type == ConfirmationType.COMMUNICATION_DETAILS
        ]
        assert len(comm) == 0

    def test_generate_not_ready_to_write(self):
        result = self.svc.generate(self.spec)
        assert not result.ready_to_write

    def test_generate_manual_todos(self):
        result = self.svc.generate(self.spec)
        assert len(result.manual_todo) >= 2
        assert any("seabreeze" in t for t in result.manual_todo)

    def test_generate_manual_todo_no_sdk(self):
        spec = _make_spec()
        spec.sdk_package = ""
        result = self.svc.generate(spec)
        assert any("SDK" in t or "Python SDK" in t for t in result.manual_todo)


# ---------------------------------------------------------------------------
# Service tests — safety classification inference
# ---------------------------------------------------------------------------


class TestSafetyInference:
    def setup_method(self):
        self.svc = InstrumentOnboardingService(project_root="/tmp/test_project")

    def test_hazardous_inferred_as_hazardous(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        set_lamp = next(
            p for p in result.spec.primitives if p.name == "set_lamp"
        )
        assert set_lamp.safety_class == "HAZARDOUS"

    def test_hazardous_inferred_error_class_critical(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        set_lamp = next(
            p for p in result.spec.primitives if p.name == "set_lamp"
        )
        assert set_lamp.error_class == "CRITICAL"

    def test_data_generating_inferred_as_careful(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        measure = next(
            p for p in result.spec.primitives if p.name == "measure_spectrum"
        )
        assert measure.safety_class == "CAREFUL"

    def test_non_hazardous_non_data_inferred_reversible(self):
        spec = _make_spec(n_primitives=0)
        spec.primitives.append(PrimitiveInput(
            name="reset",
            hazardous=False,
            generates_data=False,
        ))
        result = self.svc.generate(spec)
        reset = next(p for p in result.spec.primitives if p.name == "reset")
        assert reset.safety_class == "REVERSIBLE"
        assert reset.error_class == "BYPASS"

    def test_explicit_safety_class_preserved(self):
        spec = _make_spec(n_primitives=0)
        spec.primitives.append(PrimitiveInput(
            name="calibrate",
            safety_class="INFORMATIONAL",
            error_class="BYPASS",
        ))
        result = self.svc.generate(spec)
        cal = next(p for p in result.spec.primitives if p.name == "calibrate")
        assert cal.safety_class == "INFORMATIONAL"
        assert cal.error_class == "BYPASS"


# ---------------------------------------------------------------------------
# Service tests — confirm
# ---------------------------------------------------------------------------


class TestServiceConfirm:
    def setup_method(self):
        self.svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        self.spec = _make_spec()
        self.result = self.svc.generate(self.spec)

    def test_confirm_marks_items(self):
        # Confirm all items with their current values
        confirmations = {
            c.id: c.current_value for c in self.result.pending_confirmations
        }
        updated = self.svc.confirm(self.result, confirmations)
        assert updated.ready_to_write

    def test_confirm_partial(self):
        # Confirm only the first item
        first = self.result.pending_confirmations[0]
        updated = self.svc.confirm(self.result, {first.id: first.current_value})
        assert updated.confirmed_count == 1
        if updated.total_confirmations > 1:
            assert not updated.ready_to_write

    def test_confirm_changes_safety_class(self):
        # Change set_lamp safety from HAZARDOUS to CAREFUL
        updated = self.svc.confirm(
            self.result,
            {"safety_class:set_lamp": "CAREFUL"},
        )
        set_lamp = next(
            p for p in updated.spec.primitives if p.name == "set_lamp"
        )
        assert set_lamp.safety_class == "CAREFUL"

    def test_confirm_changes_error_class(self):
        updated = self.svc.confirm(
            self.result,
            {"error_class:measure_spectrum": "BYPASS"},
        )
        measure = next(
            p for p in updated.spec.primitives if p.name == "measure_spectrum"
        )
        assert measure.error_class == "BYPASS"

    def test_confirm_regenerates_files(self):
        old_file_count = len(self.result.files)
        confirmations = {
            c.id: c.current_value for c in self.result.pending_confirmations
        }
        updated = self.svc.confirm(self.result, confirmations)
        # Files should be regenerated (same count)
        assert len(updated.files) == old_file_count

    def test_confirm_unknown_id_ignored(self):
        updated = self.svc.confirm(self.result, {"nonexistent:id": "value"})
        # No items should be confirmed
        assert updated.confirmed_count == 0


# ---------------------------------------------------------------------------
# Service tests — write_files
# ---------------------------------------------------------------------------


class TestServiceWriteFiles:
    def test_write_requires_confirmations(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        spec = _make_spec()
        result = svc.generate(spec)
        with pytest.raises(RuntimeError, match="pending"):
            svc.write_files(result)

    def test_write_force_bypasses_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = InstrumentOnboardingService(project_root=tmpdir)
            spec = _make_spec()
            result = svc.generate(spec)
            written = svc.write_files(result, force=True)
            assert len(written) > 0

    def test_write_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = InstrumentOnboardingService(project_root=tmpdir)
            spec = _make_spec()
            result = svc.generate(spec)
            confirmations = {
                c.id: c.current_value for c in result.pending_confirmations
            }
            result = svc.confirm(result, confirmations)
            written = svc.write_files(result)
            for path in written:
                assert os.path.exists(path), f"File not created: {path}"

    def test_write_controller_has_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = InstrumentOnboardingService(project_root=tmpdir)
            spec = _make_spec()
            result = svc.generate(spec)
            svc.write_files(result, force=True)
            ctrl_path = Path(tmpdir) / "app" / "hardware" / "uv_vis_controller.py"
            assert ctrl_path.exists()
            content = ctrl_path.read_text()
            assert "class UvVisController" in content
            assert "measure_spectrum" in content
            assert "set_lamp" in content

    def test_write_skill_has_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = InstrumentOnboardingService(project_root=tmpdir)
            spec = _make_spec()
            result = svc.generate(spec)
            svc.write_files(result, force=True)
            skill_path = Path(tmpdir) / "agent" / "skills" / "uv_vis.md"
            assert skill_path.exists()
            content = skill_path.read_text()
            assert "---" in content
            assert "uv_vis.measure_spectrum" in content


# ---------------------------------------------------------------------------
# Service tests — format_confirmations_for_chat
# ---------------------------------------------------------------------------


class TestFormatConfirmationsForChat:
    def test_no_pending_returns_empty(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        result = OnboardingResult(spec=_make_spec())
        assert svc.format_confirmations_for_chat(result) == ""

    def test_format_includes_instrument_name(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        result = svc.generate(_make_spec())
        msg = svc.format_confirmations_for_chat(result)
        assert "Ocean Insight Flame-S" in msg

    def test_format_includes_safety_section(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        result = svc.generate(_make_spec())
        msg = svc.format_confirmations_for_chat(result)
        assert "Safety" in msg

    def test_format_includes_kpi_section(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        result = svc.generate(_make_spec())
        msg = svc.format_confirmations_for_chat(result)
        assert "KPI" in msg

    def test_format_all_confirmed_returns_empty(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        result = svc.generate(_make_spec())
        confirmations = {
            c.id: c.current_value for c in result.pending_confirmations
        }
        result = svc.confirm(result, confirmations)
        msg = svc.format_confirmations_for_chat(result)
        assert msg == ""


# ---------------------------------------------------------------------------
# File generator content tests
# ---------------------------------------------------------------------------


class TestGeneratedFileContents:
    def setup_method(self):
        self.svc = InstrumentOnboardingService(project_root="/tmp/test_project")

    def test_controller_usb(self):
        spec = _make_spec(communication=CommunicationType.USB)
        result = self.svc.generate(spec)
        ctrl = next(f for f in result.files if "controller" in f.path)
        assert "port" in ctrl.content
        assert "baudrate" in ctrl.content
        assert "class UvVisController" in ctrl.content

    def test_controller_tcp(self):
        spec = _make_spec(communication=CommunicationType.TCP)
        result = self.svc.generate(spec)
        ctrl = next(f for f in result.files if "controller" in f.path)
        assert "host" in ctrl.content
        assert "port" in ctrl.content

    def test_controller_simulated(self):
        spec = _make_spec(communication=CommunicationType.SIMULATED)
        result = self.svc.generate(spec)
        ctrl = next(f for f in result.files if "controller" in f.path)
        assert "simulated" in ctrl.content.lower()

    def test_skill_md_has_all_primitives(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        skill = next(f for f in result.files if ".md" in f.path)
        assert "uv_vis.measure_spectrum" in skill.content
        assert "uv_vis.set_lamp" in skill.content

    def test_skill_md_has_safety_classes(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        skill = next(f for f in result.files if ".md" in f.path)
        assert "CAREFUL" in skill.content  # measure_spectrum
        assert "HAZARDOUS" in skill.content  # set_lamp

    def test_dispatcher_patch_has_handlers(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        patch = next(f for f in result.files if "dispatcher" in f.path)
        assert "_handle_uv_vis_measure_spectrum" in patch.content
        assert "_handle_uv_vis_set_lamp" in patch.content

    def test_simulated_patch_has_entries(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        patch = next(f for f in result.files if "simulated" in f.path)
        assert '"uv_vis.measure_spectrum"' in patch.content
        assert '"uv_vis.set_lamp"' in patch.content

    def test_adapter_patch_has_registration(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        patch = next(
            f for f in result.files if "adapter" in f.path and "simulated" not in f.path
        )
        assert "uv_vis_controller" in patch.content
        assert "UvVisController" in patch.content

    def test_dryrun_patch_has_actions(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        patch = next(f for f in result.files if "dryrun" in f.path)
        assert '"uv_vis.measure_spectrum"' in patch.content

    def test_tests_file_has_imports(self):
        spec = _make_spec()
        result = self.svc.generate(spec)
        tests = next(f for f in result.files if "test_" in f.path)
        assert "UvVisController" in tests.content
        assert "pytest" in tests.content


# ---------------------------------------------------------------------------
# OnboardingResult properties
# ---------------------------------------------------------------------------


class TestOnboardingResult:
    def test_ready_to_write_all_confirmed(self):
        result = OnboardingResult(
            spec=_make_spec(),
            pending_confirmations=[
                ConfirmationItem(
                    id="a", type=ConfirmationType.SAFETY_CLASSIFICATION,
                    primitive_name="x", question="?", current_value="V",
                    confirmed=True, confirmed_value="V",
                ),
            ],
        )
        assert result.ready_to_write

    def test_not_ready_if_unconfirmed(self):
        result = OnboardingResult(
            spec=_make_spec(),
            pending_confirmations=[
                ConfirmationItem(
                    id="a", type=ConfirmationType.SAFETY_CLASSIFICATION,
                    primitive_name="x", question="?", current_value="V",
                    confirmed=False,
                ),
            ],
        )
        assert not result.ready_to_write

    def test_confirmed_count(self):
        result = OnboardingResult(
            spec=_make_spec(),
            pending_confirmations=[
                ConfirmationItem(
                    id="a", type=ConfirmationType.SAFETY_CLASSIFICATION,
                    primitive_name="x", question="?", current_value="V",
                    confirmed=True, confirmed_value="V",
                ),
                ConfirmationItem(
                    id="b", type=ConfirmationType.SAFETY_CLASSIFICATION,
                    primitive_name="y", question="?", current_value="V",
                    confirmed=False,
                ),
            ],
        )
        assert result.confirmed_count == 1
        assert result.total_confirmations == 2


class TestConfirmationItem:
    def test_final_value_confirmed(self):
        item = ConfirmationItem(
            id="x", type=ConfirmationType.SAFETY_CLASSIFICATION,
            primitive_name="p", question="?", current_value="A",
            confirmed=True, confirmed_value="B",
        )
        assert item.final_value == "B"

    def test_final_value_unconfirmed(self):
        item = ConfirmationItem(
            id="x", type=ConfirmationType.SAFETY_CLASSIFICATION,
            primitive_name="p", question="?", current_value="A",
        )
        assert item.final_value == "A"


# ---------------------------------------------------------------------------
# OnboardingAgent tests
# ---------------------------------------------------------------------------


class TestOnboardingAgent:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from app.agents.onboarding_agent import OnboardingAgent
        self.agent = OnboardingAgent()

    def test_agent_name(self):
        assert self.agent.name == "onboarding"

    def test_validate_generate_requires_name(self):
        from app.agents.onboarding_agent import OnboardingInput
        inp = OnboardingInput(phase="generate", primitives=[])
        errors = self.agent.validate_input(inp)
        assert any("instrument_name" in e for e in errors)

    def test_validate_generate_requires_primitives(self):
        from app.agents.onboarding_agent import OnboardingInput
        inp = OnboardingInput(phase="generate", instrument_name="test")
        errors = self.agent.validate_input(inp)
        assert any("primitive" in e for e in errors)

    def test_validate_confirm_requires_previous(self):
        from app.agents.onboarding_agent import OnboardingInput
        inp = OnboardingInput(
            phase="confirm",
            confirmations={"a": "b"},
        )
        errors = self.agent.validate_input(inp)
        assert any("previous_result" in e for e in errors)

    def test_validate_confirm_requires_confirmations(self):
        from app.agents.onboarding_agent import OnboardingInput
        inp = OnboardingInput(
            phase="confirm",
            previous_result={"spec": {}},
        )
        errors = self.agent.validate_input(inp)
        assert any("confirmations" in e for e in errors)

    def test_validate_write_requires_previous(self):
        from app.agents.onboarding_agent import OnboardingInput
        inp = OnboardingInput(phase="write")
        errors = self.agent.validate_input(inp)
        assert any("previous_result" in e for e in errors)

    def test_validate_unknown_phase(self):
        from app.agents.onboarding_agent import OnboardingInput
        inp = OnboardingInput(phase="explode")
        errors = self.agent.validate_input(inp)
        assert any("Unknown phase" in e for e in errors)

    @pytest.mark.anyio
    async def test_generate_phase(self):
        from app.agents.onboarding_agent import OnboardingInput, PrimitiveSpec
        inp = OnboardingInput(
            phase="generate",
            instrument_name="test_sensor",
            manufacturer="TestCo",
            model="T-100",
            communication="usb",
            primitives=[
                PrimitiveSpec(
                    name="read_value",
                    description="Read sensor value",
                    params={"channel": {"type": "integer", "default": 1}},
                    generates_data=True,
                ),
            ],
        )
        result = await self.agent.run(inp)
        assert result.success
        assert result.output is not None
        assert result.output.status == "needs_confirmation"
        assert result.output.instrument_name == "test_sensor"
        assert len(result.output.pending_confirmations) > 0
        assert result.output.chat_message != ""

    @pytest.mark.anyio
    async def test_full_lifecycle(self):
        """Test generate → confirm all → check ready."""
        from app.agents.onboarding_agent import OnboardingInput, PrimitiveSpec

        # 1. Generate
        gen_input = OnboardingInput(
            phase="generate",
            instrument_name="lifecycle_test",
            manufacturer="Lab",
            model="X1",
            communication="simulated",
            primitives=[
                PrimitiveSpec(
                    name="measure",
                    generates_data=True,
                ),
            ],
        )
        gen_result = await self.agent.run(gen_input)
        assert gen_result.success
        assert gen_result.output.status == "needs_confirmation"

        # 2. Confirm all
        confirmations = {
            c.id: c.current_value
            for c in gen_result.output.pending_confirmations
        }
        confirm_input = OnboardingInput(
            phase="confirm",
            confirmations=confirmations,
            previous_result=gen_result.output.serialised_result,
        )
        confirm_result = await self.agent.run(confirm_input)
        assert confirm_result.success
        assert confirm_result.output.status == "ready_to_write"

    @pytest.mark.anyio
    async def test_write_phase_unconfirmed_fails(self):
        """Write without confirmations should return error status."""
        from app.agents.onboarding_agent import OnboardingInput, PrimitiveSpec

        gen_input = OnboardingInput(
            phase="generate",
            instrument_name="write_fail_test",
            communication="simulated",
            primitives=[PrimitiveSpec(name="test_prim")],
        )
        gen_result = await self.agent.run(gen_input)
        assert gen_result.success

        write_input = OnboardingInput(
            phase="write",
            previous_result=gen_result.output.serialised_result,
        )
        write_result = await self.agent.run(write_input)
        assert write_result.success
        assert write_result.output.status == "error"


# ---------------------------------------------------------------------------
# Serialisation round-trip tests
# ---------------------------------------------------------------------------


class TestSerialisation:
    @pytest.mark.anyio
    async def test_round_trip_preserves_spec(self):
        from app.agents.onboarding_agent import OnboardingAgent, OnboardingInput, PrimitiveSpec

        agent = OnboardingAgent()

        gen_input = OnboardingInput(
            phase="generate",
            instrument_name="serial_test",
            manufacturer="SerialCo",
            model="S1",
            communication="tcp",
            primitives=[
                PrimitiveSpec(
                    name="read",
                    description="Read data",
                    params={"timeout": {"type": "number", "default": 5.0}},
                    generates_data=True,
                ),
                PrimitiveSpec(
                    name="reset",
                    hazardous=True,
                ),
            ],
        )
        gen_result = await agent.run(gen_input)
        assert gen_result.success

        # Deserialise from the serialised result
        restored = agent._deserialise_result(gen_result.output.serialised_result)
        assert restored.spec.name == "serial_test"
        assert restored.spec.manufacturer == "SerialCo"
        assert restored.spec.communication == CommunicationType.TCP
        assert len(restored.spec.primitives) == 2
        assert len(restored.pending_confirmations) > 0
        assert len(restored.files) == 7

    @pytest.mark.anyio
    async def test_round_trip_preserves_confirmations(self):
        from app.agents.onboarding_agent import OnboardingAgent, OnboardingInput, PrimitiveSpec

        agent = OnboardingAgent()

        gen_input = OnboardingInput(
            phase="generate",
            instrument_name="conf_test",
            communication="usb",
            primitives=[PrimitiveSpec(name="measure", generates_data=True)],
        )
        gen_result = await agent.run(gen_input)

        serialised = gen_result.output.serialised_result
        restored = agent._deserialise_result(serialised)

        orig_ids = {c.id for c in gen_result.output.pending_confirmations}
        restored_ids = {c.id for c in restored.pending_confirmations}
        assert orig_ids == restored_ids


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_primitive_no_params(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        spec = InstrumentSpec(
            name="simple_relay",
            primitives=[PrimitiveInput(name="toggle")],
        )
        result = svc.generate(spec)
        assert len(result.files) == 7
        ctrl = next(f for f in result.files if "controller" in f.path)
        assert "toggle" in ctrl.content

    def test_many_primitives(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        prims = [
            PrimitiveInput(name=f"op_{i}", generates_data=(i % 2 == 0))
            for i in range(10)
        ]
        spec = InstrumentSpec(name="multi_op", primitives=prims)
        result = svc.generate(spec)
        assert len(result.files) == 7
        # 10 safety_class + 10 error_class + 5 KPI (even indices) = 25
        # plus comm confirmations
        safety = [
            c for c in result.pending_confirmations
            if c.type == ConfirmationType.SAFETY_CLASSIFICATION
        ]
        assert len(safety) == 20  # 10 primitives × 2

    def test_special_chars_in_name(self):
        svc = InstrumentOnboardingService(project_root="/tmp/test_project")
        spec = InstrumentSpec(
            name="uv_vis_2",
            primitives=[PrimitiveInput(name="measure")],
        )
        result = svc.generate(spec)
        assert result.spec.class_name == "UvVis2Controller"

    def test_empty_manufacturer_model(self):
        spec = InstrumentSpec(name="bare_sensor")
        assert "Bare Sensor" in spec.display_name
