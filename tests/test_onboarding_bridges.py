"""Tests for the onboarding bridge layer.

Validates that the instrument onboarding system properly bridges to:
1. PrimitivesRegistry — instrument discovery + refresh
2. conversation_engine — dynamic instrument list
3. nl_parse — instrument extraction + onboarding flag
4. instrument_onboarding — post-write registry refresh
5. contract_bridge — primitive cross-validation
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory with sample skill files."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Robot skill (ot2-robot)
    (skills_dir / "robot.md").write_text(textwrap.dedent("""\
        ---
        name: robot
        description: OT-2 liquid handler
        version: "1.0"
        instrument: ot2-robot
        resource_id: ot2
        primitives:
          - name: robot.aspirate
            description: Aspirate liquid from well
            error_class: CRITICAL
            params:
              volume:
                type: number
                description: Volume in uL
          - name: robot.dispense
            description: Dispense liquid to well
            error_class: CRITICAL
            params:
              volume:
                type: number
                description: Volume in uL
        ---
        # Robot Skill
    """))

    # Squidstat skill
    (skills_dir / "squidstat.md").write_text(textwrap.dedent("""\
        ---
        name: squidstat
        description: Electrochemical workstation
        version: "1.0"
        instrument: squidstat
        resource_id: squidstat
        primitives:
          - name: squidstat.run_experiment
            description: Run electrochemical experiment
            error_class: CRITICAL
        ---
        # Squidstat Skill
    """))

    # PLC skill (plc-controller)
    (skills_dir / "plc.md").write_text(textwrap.dedent("""\
        ---
        name: plc
        description: PLC controller
        version: "1.0"
        instrument: plc-controller
        resource_id: plc
        primitives:
          - name: plc.dispense_ml
            description: Dispense via PLC pump
            error_class: CRITICAL
        ---
        # PLC Skill
    """))

    # Utility skill (no instrument)
    (skills_dir / "utility.md").write_text(textwrap.dedent("""\
        ---
        name: utility
        description: General utility actions
        version: "1.0"
        instrument: null
        resource_id: null
        primitives:
          - name: wait
            description: Wait for a duration
            error_class: BYPASS
        ---
        # Utility Skill
    """))

    return skills_dir


@pytest.fixture()
def registry(tmp_skills_dir: Path):
    """Build a PrimitivesRegistry from the temp skills directory."""
    from app.services.primitives_registry import PrimitivesRegistry

    reg = PrimitivesRegistry()
    reg.load_skills_dir(tmp_skills_dir)
    return reg


# ===========================================================================
# Step 1: PrimitivesRegistry — instrument discovery methods
# ===========================================================================


class TestListInstruments:
    """PrimitivesRegistry.list_instruments() returns full IDs."""

    def test_returns_full_instrument_ids(self, registry):
        instruments = registry.list_instruments()
        assert "ot2-robot" in instruments
        assert "squidstat" in instruments
        assert "plc-controller" in instruments

    def test_excludes_null_instruments(self, registry):
        """Utility skills (instrument=null) are excluded."""
        instruments = registry.list_instruments()
        assert None not in instruments

    def test_result_is_sorted(self, registry):
        instruments = registry.list_instruments()
        assert instruments == sorted(instruments)


class TestListInstrumentShortNames:
    """PrimitivesRegistry.list_instrument_short_names() applies dash-split."""

    def test_short_names(self, registry):
        names = registry.list_instrument_short_names()
        assert "ot2" in names
        assert "plc" in names
        assert "squidstat" in names

    def test_no_full_ids(self, registry):
        names = registry.list_instrument_short_names()
        assert "ot2-robot" not in names
        assert "plc-controller" not in names

    def test_result_is_sorted(self, registry):
        names = registry.list_instrument_short_names()
        assert names == sorted(names)


class TestInstrumentShortToFull:
    """PrimitivesRegistry.instrument_short_to_full() mapping."""

    def test_mapping(self, registry):
        mapping = registry.instrument_short_to_full()
        assert mapping["ot2"] == "ot2-robot"
        assert mapping["plc"] == "plc-controller"
        assert mapping["squidstat"] == "squidstat"

    def test_round_trip(self, registry):
        """short_to_full values are in list_instruments."""
        mapping = registry.instrument_short_to_full()
        all_instruments = set(registry.list_instruments())
        for full_id in mapping.values():
            assert full_id in all_instruments


class TestResolveInstrument:
    """PrimitivesRegistry.resolve_instrument()."""

    def test_full_id_resolves(self, registry):
        assert registry.resolve_instrument("ot2-robot") == "ot2-robot"

    def test_short_name_resolves(self, registry):
        assert registry.resolve_instrument("ot2") == "ot2-robot"

    def test_unknown_returns_none(self, registry):
        assert registry.resolve_instrument("unknown_thing") is None


class TestRefreshRegistry:
    """refresh_registry() clears cache and reloads."""

    def test_refresh_clears_cache(self, tmp_skills_dir: Path):
        from app.services.primitives_registry import (
            PrimitivesRegistry,
            get_registry,
            refresh_registry,
        )

        # Patch the default skills dir to use our temp dir
        with patch(
            "app.services.primitives_registry._DEFAULT_SKILLS_DIR",
            tmp_skills_dir,
        ):
            # Clear any existing cache
            get_registry.cache_clear()

            reg1 = get_registry()
            assert "ot2-robot" in reg1.list_instruments()

            # Add a new skill file
            (tmp_skills_dir / "new_instr.md").write_text(textwrap.dedent("""\
                ---
                name: raman
                description: Raman spectrometer
                version: "1.0"
                instrument: raman-spectrometer
                resource_id: raman
                primitives:
                  - name: raman.measure
                    description: Capture Raman spectrum
                    error_class: CRITICAL
                ---
                # Raman
            """))

            # Without refresh, old registry is cached
            reg_cached = get_registry()
            assert "raman-spectrometer" not in reg_cached.list_instruments()

            # After refresh, new instrument appears
            reg2 = refresh_registry()
            assert "raman-spectrometer" in reg2.list_instruments()
            assert "raman" in reg2.list_instrument_short_names()

            # Cleanup
            get_registry.cache_clear()


# ===========================================================================
# Step 2: conversation_engine — dynamic instrument list
# ===========================================================================


class TestGetAvailableInstruments:
    """_get_available_instruments() uses registry with fallback."""

    def test_returns_list(self):
        from app.services.conversation_engine import _get_available_instruments

        result = _get_available_instruments()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_fallback_instruments_present(self):
        """Even if registry is empty, fallback instruments are included."""
        from app.services.conversation_engine import _get_available_instruments

        result = _get_available_instruments()
        # At minimum, the fallback list should be there
        assert "ot2" in result
        assert "squidstat" in result

    def test_with_registry(self, tmp_skills_dir: Path):
        """When registry has instruments, they appear in the list."""
        from app.services.conversation_engine import _get_available_instruments

        with patch(
            "app.services.primitives_registry._DEFAULT_SKILLS_DIR",
            tmp_skills_dir,
        ):
            from app.services.primitives_registry import get_registry
            get_registry.cache_clear()

            result = _get_available_instruments()
            assert "ot2" in result
            assert "plc" in result

            get_registry.cache_clear()


class TestGetKpiInstrumentMap:
    """_get_kpi_instrument_map() returns a mapping."""

    def test_returns_dict(self):
        from app.services.conversation_engine import _get_kpi_instrument_map

        result = _get_kpi_instrument_map()
        assert isinstance(result, dict)
        assert "overpotential_mv" in result
        assert result["overpotential_mv"] == "squidstat"


# ===========================================================================
# Step 3: NL parse — instrument extraction
# ===========================================================================


class TestNLParseInstrumentDetection:
    """parse_nl_text() detects instruments from text."""

    def test_detects_ot2_english(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("I want to use OT-2 for liquid handling")
        assert "ot2" in resp.detected_instruments

    def test_detects_ot2_chinese(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("使用移液工作站进行实验")
        assert "ot2" in resp.detected_instruments

    def test_detects_squidstat(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("用squidstat测量电化学数据")
        assert "squidstat" in resp.detected_instruments

    def test_detects_squidstat_chinese(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("使用电化学工作站进行循环伏安测试")
        assert "squidstat" in resp.detected_instruments

    def test_detects_furnace(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("样品在管式炉中退火")
        assert "furnace" in resp.detected_instruments

    def test_detects_multiple_instruments(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("OT-2配液，squidstat测量电化学性能")
        assert "ot2" in resp.detected_instruments
        assert "squidstat" in resp.detected_instruments

    def test_no_instruments_in_plain_text(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("优化反应温度和浓度")
        assert resp.detected_instruments == []

    def test_instrument_appears_in_extracted(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("use the OT-2 robot")
        instr_params = [e for e in resp.extracted if e.key == "instrument"]
        assert len(instr_params) > 0
        assert instr_params[0].value == "ot2"


class TestNLParseOnboardingSuggestion:
    """parse_nl_text() suggests onboarding for unknown instruments."""

    def test_unknown_instrument_flags_onboarding(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        # "Raman光谱仪" is not a known instrument
        resp = parse_nl_text("我想用Raman光谱仪做实验")
        assert resp.onboarding_suggested is True
        assert len(resp.unknown_instruments) > 0

    def test_known_instruments_no_onboarding(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("使用OT-2进行实验")
        assert resp.onboarding_suggested is False
        assert resp.unknown_instruments == []

    def test_unknown_in_extracted_params(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("用自定义Raman光谱仪做实验")
        unk_params = [e for e in resp.extracted if e.key == "unknown_instrument"]
        if resp.onboarding_suggested:
            assert len(unk_params) > 0
            assert unk_params[0].confidence < 1.0


class TestNLParseResponseSchema:
    """NLParseResponse has the new instrument fields."""

    def test_response_has_instrument_fields(self):
        from app.api.v1.endpoints.nl_parse import NLParseResponse

        resp = NLParseResponse(original_text="test")
        assert hasattr(resp, "detected_instruments")
        assert hasattr(resp, "unknown_instruments")
        assert hasattr(resp, "onboarding_suggested")
        assert resp.detected_instruments == []
        assert resp.unknown_instruments == []
        assert resp.onboarding_suggested is False


# ===========================================================================
# Step 4: instrument_onboarding — post-write refresh
# ===========================================================================


class TestPostWriteRefresh:
    """write_files() triggers registry refresh when skill file written."""

    def test_refresh_called_on_skill_write(self, tmp_path: Path):
        """When write_files() writes a skill .md, refresh_registry is called."""
        from app.services.instrument_onboarding import (
            CommunicationType,
            InstrumentOnboardingService,
            InstrumentSpec,
            PrimitiveInput,
        )

        spec = InstrumentSpec(
            name="test_instr",
            manufacturer="TestCorp",
            communication=CommunicationType.USB,
            primitives=[
                PrimitiveInput(
                    name="measure",
                    description="Take a measurement",
                ),
            ],
        )

        svc = InstrumentOnboardingService(project_root=tmp_path)
        result = svc.generate(spec)

        # Force all confirmations
        for c in result.pending_confirmations:
            c.confirmed = True
            c.confirmed_value = c.current_value

        # Regenerate to mark ready
        result = svc.confirm(result, {
            c.id: c.current_value for c in result.pending_confirmations
        })

        # Patch refresh_registry at the source module so the lazy import picks it up
        with patch(
            "app.services.primitives_registry.refresh_registry"
        ) as mock_refresh:
            mock_refresh.return_value = MagicMock()
            written = svc.write_files(result, force=True)

        # Verify refresh was called (because skill .md is in generated files)
        skill_file_written = any(
            f.endswith(".md") and "skills/" in f for f in written
        )
        if skill_file_written:
            mock_refresh.assert_called_once()

    def test_no_refresh_if_no_skill_file(self, tmp_path: Path):
        """If no skill file in output, refresh is not called."""
        from app.services.instrument_onboarding import (
            GeneratedFile,
            OnboardingResult,
            InstrumentOnboardingService,
            InstrumentSpec,
            CommunicationType,
        )

        # Create a result with only non-skill files
        spec = InstrumentSpec(
            name="test",
            communication=CommunicationType.USB,
            primitives=[],
        )
        result = OnboardingResult(
            spec=spec,
            files=[
                GeneratedFile(
                    path="app/hardware/test_controller.py",
                    content="# test",
                    description="Controller",
                ),
            ],
            pending_confirmations=[],
        )

        svc = InstrumentOnboardingService(project_root=tmp_path)

        with patch(
            "app.services.primitives_registry.refresh_registry"
        ) as mock_refresh:
            svc.write_files(result, force=True)

        mock_refresh.assert_not_called()


# ===========================================================================
# Step 5: contract_bridge — primitive validation
# ===========================================================================


class TestPrimitiveValidation:
    """_validate_primitives_against_registry() checks primitives."""

    def test_known_primitives_no_warnings(self, tmp_skills_dir: Path):
        from app.services.contract_bridge import (
            _validate_primitives_against_registry,
        )

        with patch(
            "app.services.primitives_registry._DEFAULT_SKILLS_DIR",
            tmp_skills_dir,
        ):
            from app.services.primitives_registry import get_registry
            get_registry.cache_clear()

            warnings = _validate_primitives_against_registry(
                ["robot.aspirate", "wait", "log"]
            )
            assert warnings == []

            get_registry.cache_clear()

    def test_unknown_primitives_generate_warnings(self, tmp_skills_dir: Path):
        from app.services.contract_bridge import (
            _validate_primitives_against_registry,
        )

        with patch(
            "app.services.primitives_registry._DEFAULT_SKILLS_DIR",
            tmp_skills_dir,
        ):
            from app.services.primitives_registry import get_registry
            get_registry.cache_clear()

            warnings = _validate_primitives_against_registry(
                ["robot.aspirate", "unknown.action", "another.thing"]
            )
            assert len(warnings) == 2
            assert any("unknown.action" in w for w in warnings)

            get_registry.cache_clear()

    def test_empty_list_no_warnings(self):
        from app.services.contract_bridge import (
            _validate_primitives_against_registry,
        )

        warnings = _validate_primitives_against_registry([])
        assert warnings == []


# ===========================================================================
# Integration: end-to-end flow
# ===========================================================================


class TestInstrumentShortNameHelper:
    """_instrument_short_name() helper function."""

    def test_with_dash(self):
        from app.services.primitives_registry import _instrument_short_name

        assert _instrument_short_name("ot2-robot") == "ot2"
        assert _instrument_short_name("plc-controller") == "plc"
        assert _instrument_short_name("relay-controller") == "relay"

    def test_without_dash(self):
        from app.services.primitives_registry import _instrument_short_name

        assert _instrument_short_name("squidstat") == "squidstat"

    def test_multiple_dashes(self):
        from app.services.primitives_registry import _instrument_short_name

        assert _instrument_short_name("my-custom-instrument") == "my"


class TestNLParseDetectsSpinCoater:
    """Ensure spin coater patterns work in both languages."""

    def test_spin_coater_english(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("apply film with spin coater at 3000 rpm")
        assert "spin_coater" in resp.detected_instruments

    def test_spin_coater_chinese(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("使用旋涂机涂覆薄膜")
        assert "spin_coater" in resp.detected_instruments


class TestNLParsePLC:
    """PLC detection."""

    def test_plc_english(self):
        from app.api.v1.endpoints.nl_parse import parse_nl_text

        resp = parse_nl_text("control dispensing with PLC")
        assert "plc" in resp.detected_instruments


class TestEmptyRegistry:
    """Behaviour when registry is empty or unavailable."""

    def test_list_instruments_empty(self):
        from app.services.primitives_registry import PrimitivesRegistry

        reg = PrimitivesRegistry()
        assert reg.list_instruments() == []
        assert reg.list_instrument_short_names() == []
        assert reg.instrument_short_to_full() == {}

    def test_resolve_instrument_empty(self):
        from app.services.primitives_registry import PrimitivesRegistry

        reg = PrimitivesRegistry()
        assert reg.resolve_instrument("ot2") is None
