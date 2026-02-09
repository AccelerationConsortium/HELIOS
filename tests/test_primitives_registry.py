"""Tests for app.services.primitives_registry — SKILL.md parsing and catalogue."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from app.services.action_contracts import SafetyClass
from app.services.primitives_registry import (
    PrimitiveParam,
    PrimitiveSpec,
    PrimitivesRegistry,
    SkillDescriptor,
    _extract_frontmatter,
    _parse_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_SKILL = textwrap.dedent("""\
    ---
    name: test-skill
    description: "A test skill"
    version: "1.0.0"
    instrument: test-instrument
    resource_id: test-resource
    primitives:
      - name: test.action
        error_class: CRITICAL
        params:
          volume: {type: number, description: "Volume in uL"}
          well: {type: string, description: "Target well"}
        description: "A test action"
      - name: test.helper
        error_class: BYPASS
        params: {}
        description: "A helper action"
    ---

    # Test Skill

    This is a test skill description.
""")

UTILITY_SKILL = textwrap.dedent("""\
    ---
    name: utility
    description: "Utility primitives"
    version: "1.0.0"
    instrument: null
    resource_id: null
    primitives:
      - name: wait
        error_class: BYPASS
        params:
          duration_seconds: {type: number, description: "Wait time"}
        description: "Pause execution"
      - name: log
        error_class: BYPASS
        params:
          message: {type: string, description: "Log message"}
        description: "Record a log entry"
    ---

    # Utility
""")


# ---------------------------------------------------------------------------
# _extract_frontmatter
# ---------------------------------------------------------------------------


class TestExtractFrontmatter:
    def test_valid_frontmatter(self) -> None:
        fm = _extract_frontmatter(MINIMAL_SKILL)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "A test skill"
        assert len(fm["primitives"]) == 2

    def test_no_frontmatter(self) -> None:
        assert _extract_frontmatter("# No frontmatter here") == {}

    def test_empty_string(self) -> None:
        assert _extract_frontmatter("") == {}


# ---------------------------------------------------------------------------
# _parse_params
# ---------------------------------------------------------------------------


class TestParseParams:
    def test_dict_params(self) -> None:
        raw = {
            "volume": {"type": "number", "description": "Volume in uL"},
            "well": {"type": "string", "description": "Target well", "optional": True},
        }
        params = _parse_params(raw)
        assert len(params) == 2
        vol = next(p for p in params if p.name == "volume")
        assert vol.type == "number"
        assert vol.optional is False

        well = next(p for p in params if p.name == "well")
        assert well.optional is True

    def test_empty_params(self) -> None:
        assert _parse_params({}) == ()
        assert _parse_params(None) == ()  # type: ignore[arg-type]

    def test_param_with_default(self) -> None:
        raw = {"offset_z": {"type": "number", "optional": True, "default": 0}}
        params = _parse_params(raw)
        assert params[0].default == 0
        assert params[0].optional is True


# ---------------------------------------------------------------------------
# PrimitivesRegistry — loading
# ---------------------------------------------------------------------------


class TestRegistryLoading:
    def test_load_skill_file(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "test.md"
        skill_file.write_text(MINIMAL_SKILL)

        registry = PrimitivesRegistry()
        skill = registry.load_skill_file(skill_file)

        assert skill is not None
        assert skill.name == "test-skill"
        assert len(skill.primitives) == 2
        assert skill.instrument == "test-instrument"

    def test_load_utility_null_instrument(self, tmp_path: Path) -> None:
        skill_file = tmp_path / "utility.md"
        skill_file.write_text(UTILITY_SKILL)

        registry = PrimitivesRegistry()
        skill = registry.load_skill_file(skill_file)

        assert skill is not None
        assert skill.instrument is None
        assert skill.resource_id is None

    def test_load_skills_dir(self, tmp_path: Path) -> None:
        (tmp_path / "test.md").write_text(MINIMAL_SKILL)
        (tmp_path / "utility.md").write_text(UTILITY_SKILL)
        (tmp_path / "not_a_skill.txt").write_text("ignore me")

        registry = PrimitivesRegistry()
        count = registry.load_skills_dir(tmp_path)

        assert count == 2
        assert len(registry.list_primitives()) == 4  # 2 + 2

    def test_load_nonexistent_dir(self) -> None:
        registry = PrimitivesRegistry()
        count = registry.load_skills_dir(Path("/nonexistent/path"))
        assert count == 0

    def test_load_file_no_frontmatter(self, tmp_path: Path) -> None:
        (tmp_path / "plain.md").write_text("# No frontmatter\nJust text.")
        registry = PrimitivesRegistry()
        skill = registry.load_skill_file(tmp_path / "plain.md")
        assert skill is None


# ---------------------------------------------------------------------------
# PrimitivesRegistry — queries
# ---------------------------------------------------------------------------


class TestRegistryQueries:
    @pytest.fixture()
    def registry(self, tmp_path: Path) -> PrimitivesRegistry:
        (tmp_path / "test.md").write_text(MINIMAL_SKILL)
        (tmp_path / "utility.md").write_text(UTILITY_SKILL)
        reg = PrimitivesRegistry()
        reg.load_skills_dir(tmp_path)
        return reg

    def test_get_primitive(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.action")
        assert p is not None
        assert p.error_class == "CRITICAL"
        assert p.instrument == "test-instrument"

    def test_get_primitive_not_found(self, registry: PrimitivesRegistry) -> None:
        assert registry.get_primitive("nonexistent") is None

    def test_list_primitive_names(self, registry: PrimitivesRegistry) -> None:
        names = registry.list_primitive_names()
        assert "test.action" in names
        assert "test.helper" in names
        assert "wait" in names
        assert "log" in names
        # Sorted
        assert names == sorted(names)

    def test_primitives_by_instrument(self, registry: PrimitivesRegistry) -> None:
        prims = registry.primitives_by_instrument("test-instrument")
        assert len(prims) == 2
        names = {p.name for p in prims}
        assert names == {"test.action", "test.helper"}

    def test_primitives_by_error_class(self, registry: PrimitivesRegistry) -> None:
        critical = registry.primitives_by_error_class("CRITICAL")
        bypass = registry.primitives_by_error_class("BYPASS")
        assert len(critical) == 1  # test.action
        assert len(bypass) == 3  # test.helper, wait, log

    def test_list_skills(self, registry: PrimitivesRegistry) -> None:
        skills = registry.list_skills()
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"test-skill", "utility"}

    def test_get_skill(self, registry: PrimitivesRegistry) -> None:
        skill = registry.get_skill("utility")
        assert skill is not None
        assert skill.description == "Utility primitives"

    def test_get_skill_not_found(self, registry: PrimitivesRegistry) -> None:
        assert registry.get_skill("nonexistent") is None

    def test_primitive_params(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.action")
        assert p is not None
        assert len(p.params) == 2
        vol = next(pp for pp in p.params if pp.name == "volume")
        assert vol.type == "number"
        assert vol.description == "Volume in uL"


# ---------------------------------------------------------------------------
# PrimitivesRegistry — serialization
# ---------------------------------------------------------------------------


class TestRegistrySerialization:
    @pytest.fixture()
    def registry(self, tmp_path: Path) -> PrimitivesRegistry:
        (tmp_path / "test.md").write_text(MINIMAL_SKILL)
        reg = PrimitivesRegistry()
        reg.load_skills_dir(tmp_path)
        return reg

    def test_to_dict(self, registry: PrimitivesRegistry) -> None:
        d = registry.to_dict()
        assert d["total_skills"] == 1
        assert d["total_primitives"] == 2
        assert len(d["skills"]) == 1
        skill = d["skills"][0]
        assert skill["name"] == "test-skill"
        assert skill["primitive_count"] == 2
        assert len(skill["primitives"]) == 2

    def test_to_dict_primitive_params(self, registry: PrimitivesRegistry) -> None:
        d = registry.to_dict()
        prim = d["skills"][0]["primitives"][0]
        assert len(prim["params"]) == 2
        vol = next(p for p in prim["params"] if p["name"] == "volume")
        assert vol["type"] == "number"

    def test_summary_for_llm(self, registry: PrimitivesRegistry) -> None:
        summary = registry.summary_for_llm()
        assert "test-skill" in summary
        assert "test.action" in summary
        assert "[CRITICAL]" in summary
        assert "[BYPASS]" in summary
        assert "volume" in summary


# ---------------------------------------------------------------------------
# Real skill files integration test
# ---------------------------------------------------------------------------


class TestRealSkillFiles:
    """Test against the actual agent/skills/ directory."""

    def test_load_real_skills(self) -> None:
        """Verify the real skill files in agent/skills/ parse correctly."""
        skills_dir = Path(__file__).resolve().parents[1] / "agent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("agent/skills/ directory not found")

        registry = PrimitivesRegistry()
        count = registry.load_skills_dir(skills_dir)

        # We should have 5 skill files
        assert count == 5, f"Expected 5 skills, loaded {count}"

        # Verify expected skills
        skill_names = {s.name for s in registry.list_skills()}
        assert "ot2-robot" in skill_names
        assert "plc-controller" in skill_names
        assert "relay-controller" in skill_names
        assert "squidstat-potentiostat" in skill_names
        assert "utility" in skill_names

    def test_real_primitive_count(self) -> None:
        """Verify total primitive count matches expectations."""
        skills_dir = Path(__file__).resolve().parents[1] / "agent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("agent/skills/ directory not found")

        registry = PrimitivesRegistry()
        registry.load_skills_dir(skills_dir)

        # 11 robot + 3 PLC + 4 relay + 4 squidstat + 8 utility = 30
        total = len(registry.list_primitives())
        assert total == 30, f"Expected 30 primitives, got {total}"

    def test_real_error_class_distribution(self) -> None:
        """Verify CRITICAL/BYPASS classification from real skill files."""
        skills_dir = Path(__file__).resolve().parents[1] / "agent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("agent/skills/ directory not found")

        registry = PrimitivesRegistry()
        registry.load_skills_dir(skills_dir)

        critical = registry.primitives_by_error_class("CRITICAL")
        bypass = registry.primitives_by_error_class("BYPASS")

        # CRITICAL: 7 robot + 2 PLC + 1 squidstat + 2 utility = 12
        assert len(critical) == 12, (
            f"Expected 12 CRITICAL, got {len(critical)}: "
            f"{[p.name for p in critical]}"
        )
        # BYPASS: 4 robot + 1 PLC + 4 relay + 3 squidstat + 6 utility = 18
        assert len(bypass) == 18, (
            f"Expected 18 BYPASS, got {len(bypass)}: "
            f"{[p.name for p in bypass]}"
        )

    def test_robot_aspirate_has_volume_param(self) -> None:
        """Verify robot.aspirate has volume parameter from real skill file."""
        skills_dir = Path(__file__).resolve().parents[1] / "agent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("agent/skills/ directory not found")

        registry = PrimitivesRegistry()
        registry.load_skills_dir(skills_dir)

        p = registry.get_primitive("robot.aspirate")
        assert p is not None
        assert p.error_class == "CRITICAL"
        param_names = {pp.name for pp in p.params}
        assert "volume" in param_names
        assert "labware" in param_names
        assert "well" in param_names
        assert "pipette" in param_names

    def test_summary_contains_all_instruments(self) -> None:
        """Verify LLM summary mentions all instrument categories."""
        skills_dir = Path(__file__).resolve().parents[1] / "agent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("agent/skills/ directory not found")

        registry = PrimitivesRegistry()
        registry.load_skills_dir(skills_dir)

        summary = registry.summary_for_llm()
        assert "ot2-robot" in summary
        assert "plc-controller" in summary
        assert "relay-controller" in summary
        assert "squidstat-potentiostat" in summary
        assert "utility" in summary

    def test_real_safety_class_distribution(self) -> None:
        """Verify SafetyClass mapping from legacy error_class."""
        skills_dir = Path(__file__).resolve().parents[1] / "agent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("agent/skills/ directory not found")

        registry = PrimitivesRegistry()
        registry.load_skills_dir(skills_dir)

        # All primitives should have safety_class set
        for p in registry.list_primitives():
            assert isinstance(p.safety_class, SafetyClass), (
                f"{p.name} has no safety_class"
            )

        # Verify distribution via LEGACY_SAFETY_MAP fallback
        hazardous = registry.primitives_by_safety_class(SafetyClass.HAZARDOUS)
        haz_names = {p.name for p in hazardous}
        assert "robot.aspirate" in haz_names
        assert "robot.dispense" in haz_names
        assert "squidstat.run_experiment" in haz_names

        info = registry.primitives_by_safety_class("INFORMATIONAL")
        info_names = {p.name for p in info}
        assert "robot.home" in info_names
        assert "wait" in info_names
        assert "log" in info_names


# ---------------------------------------------------------------------------
# Contract integration tests
# ---------------------------------------------------------------------------


SKILL_WITH_CONTRACT = textwrap.dedent("""\
    ---
    name: contract-test
    description: "Skill with action contracts"
    version: "1.0.0"
    instrument: test-instrument
    resource_id: test-resource
    primitives:
      - name: test.aspirate
        error_class: CRITICAL
        safety_class: HAZARDOUS
        params:
          labware: {type: string, description: "Target labware"}
          well: {type: string, description: "Target well"}
          pipette: {type: string, description: "Pipette mount"}
          volume: {type: number, description: "Volume in uL"}
        description: "Aspirate with contract"
        contract:
          preconditions:
            - "labware_loaded:{labware}"
            - "tip_on:{pipette}"
            - "pipettes_loaded"
          effects:
            - "increase:pipette_volume:{pipette}:{volume}"
        timeout:
          seconds: 30
          retries: 0
      - name: test.home
        error_class: BYPASS
        safety_class: INFORMATIONAL
        params: {}
        description: "Home the robot"
        contract:
          preconditions: []
          effects:
            - "set:robot_homed:true"
        timeout:
          seconds: 60
          retries: 1
    ---

    # Contract Test Skill
""")


class TestRegistryContractIntegration:
    @pytest.fixture()
    def registry(self, tmp_path: Path) -> PrimitivesRegistry:
        (tmp_path / "contract_test.md").write_text(SKILL_WITH_CONTRACT)
        reg = PrimitivesRegistry()
        reg.load_skills_dir(tmp_path)
        return reg

    def test_contract_parsed(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.aspirate")
        assert p is not None
        assert p.contract is not None
        assert len(p.contract.preconditions) == 3
        assert len(p.contract.effects) == 1

    def test_safety_class_explicit(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.aspirate")
        assert p is not None
        assert p.safety_class == SafetyClass.HAZARDOUS

    def test_safety_class_informational(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.home")
        assert p is not None
        assert p.safety_class == SafetyClass.INFORMATIONAL

    def test_timeout_parsed(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.aspirate")
        assert p is not None
        assert p.contract is not None
        assert p.contract.timeout.seconds == 30.0
        assert p.contract.timeout.retries == 0

    def test_timeout_with_retries(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.home")
        assert p is not None
        assert p.contract is not None
        assert p.contract.timeout.seconds == 60.0
        assert p.contract.timeout.retries == 1

    def test_precondition_predicates(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.aspirate")
        assert p is not None and p.contract is not None
        preds = [pc.predicate for pc in p.contract.preconditions]
        assert "labware_loaded:{labware}" in preds
        assert "tip_on:{pipette}" in preds
        assert "pipettes_loaded" in preds

    def test_effect_operations(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.aspirate")
        assert p is not None and p.contract is not None
        ops = [ef.operation for ef in p.contract.effects]
        assert "increase:pipette_volume:{pipette}:{volume}" in ops

    def test_to_dict_includes_contract(self, registry: PrimitivesRegistry) -> None:
        d = registry.to_dict()
        prim = d["skills"][0]["primitives"][0]
        assert "safety_class" in prim
        assert prim["safety_class"] == "HAZARDOUS"
        assert "contract" in prim
        assert len(prim["contract"]["preconditions"]) == 3
        assert prim["contract"]["timeout"]["seconds"] == 30.0

    def test_to_dict_home_contract(self, registry: PrimitivesRegistry) -> None:
        d = registry.to_dict()
        home = d["skills"][0]["primitives"][1]
        assert home["safety_class"] == "INFORMATIONAL"
        assert "contract" in home
        assert len(home["contract"]["effects"]) == 1

    def test_summary_includes_safety_class(self, registry: PrimitivesRegistry) -> None:
        summary = registry.summary_for_llm()
        assert "[HAZARDOUS]" in summary
        assert "[INFORMATIONAL]" in summary

    def test_summary_includes_preconditions(self, registry: PrimitivesRegistry) -> None:
        summary = registry.summary_for_llm()
        assert "preconditions:" in summary
        assert "labware_loaded" in summary

    def test_primitives_by_safety_class(self, registry: PrimitivesRegistry) -> None:
        haz = registry.primitives_by_safety_class(SafetyClass.HAZARDOUS)
        assert len(haz) == 1
        assert haz[0].name == "test.aspirate"

        info = registry.primitives_by_safety_class("INFORMATIONAL")
        assert len(info) == 1
        assert info[0].name == "test.home"

    def test_backward_compat_error_class_preserved(self, registry: PrimitivesRegistry) -> None:
        """error_class is still available alongside safety_class."""
        p = registry.get_primitive("test.aspirate")
        assert p is not None
        assert p.error_class == "CRITICAL"
        assert p.safety_class == SafetyClass.HAZARDOUS


class TestRegistryLegacyFallback:
    """Verify skills without contract/safety_class still get proper defaults."""

    @pytest.fixture()
    def registry(self, tmp_path: Path) -> PrimitivesRegistry:
        (tmp_path / "test.md").write_text(MINIMAL_SKILL)
        reg = PrimitivesRegistry()
        reg.load_skills_dir(tmp_path)
        return reg

    def test_legacy_skill_gets_safety_class(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.action")
        assert p is not None
        # test.action has error_class=CRITICAL, no safety_class → defaults to CAREFUL
        assert p.safety_class == SafetyClass.CAREFUL

    def test_legacy_skill_bypass_gets_reversible(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.helper")
        assert p is not None
        # test.helper has error_class=BYPASS → defaults to REVERSIBLE
        assert p.safety_class == SafetyClass.REVERSIBLE

    def test_legacy_skill_has_default_contract(self, registry: PrimitivesRegistry) -> None:
        p = registry.get_primitive("test.action")
        assert p is not None
        assert p.contract is not None
        # No contract in frontmatter → empty preconditions/effects
        assert p.contract.preconditions == ()
        assert p.contract.effects == ()
        # Default timeout
        assert p.contract.timeout.seconds == 300.0
