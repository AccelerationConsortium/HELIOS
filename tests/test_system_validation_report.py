from __future__ import annotations

import inspect

import app.services.system_validation_report as system_validation_report
from app.services.system_validation_report import (
    HELIOSSystemValidationReport,
    build_architecture_summary,
    build_backend_integration_summary,
    build_dynamic_strategy_summary,
    build_safety_boundary_summary,
    build_scenario_benchmark_summary,
    build_self_evolution_summary,
    build_system_validation_report,
)
from tests.test_offline_scenario_benchmarks import ScenarioRunner, _scenarios


def _flatten_text(value) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def test_system_validation_report_can_be_generated_from_mocked_summaries():
    report = build_system_validation_report(
        offline_test_summary={
            "offline_closed_loop_sdl": {"passed": 11, "file": "tests/test_offline_closed_loop_sdl.py"},
            "policy_evolution_workflow": {"passed": 8},
        },
        scenario_benchmark_summary=build_scenario_benchmark_summary(
            aggregate_summary={
                "total_scenarios": 10,
                "pass_count": 10,
                "fail_count": 0,
                "trace_completeness_score": 1.0,
            }
        ),
        remaining_limitations=("real campaign evidence still required",),
    )

    assert isinstance(report, HELIOSSystemValidationReport)
    assert report.offline_test_summary["offline_closed_loop_sdl"]["passed"] == 11
    assert report.scenario_benchmark_summary["aggregate"]["total_scenarios"] == 10
    assert report.remaining_limitations == ("real campaign evidence still required",)
    assert report.to_dict()["report_version"] == "helios_system_validation_report_v1"


def test_system_validation_report_includes_all_required_sections():
    report = build_system_validation_report()
    raw = report.to_dict()

    assert {
        "architecture_summary",
        "dynamic_strategy_summary",
        "backend_integration_summary",
        "no_human_closed_loop_summary",
        "learned_policy_summary",
        "self_evolution_summary",
        "safety_boundary_summary",
        "offline_test_summary",
        "scenario_benchmark_summary",
        "remaining_limitations",
        "generated_at",
    } <= set(raw)


def test_report_states_default_live_behavior_unchanged():
    text = _flatten_text(build_system_validation_report().to_dict()).lower()

    assert "default live behavior is unchanged" in text
    assert "default bo mcp/nexus/backend behavior remains unchanged" in text
    assert "space revisions are approval-only" in text
    assert "not auto-applied" in text


def test_report_states_architecture_and_dynamic_strategy_boundaries():
    architecture = build_architecture_summary()
    dynamic = build_dynamic_strategy_summary()
    text = _flatten_text({"architecture": architecture, "dynamic": dynamic}).lower()

    assert "orchestrator-agnostic adaptive campaign decision layer" in text
    assert "closed-loop experimentation" in text
    assert "hierarchical" in text
    assert "graph/state-machine" in text
    assert "specialist role boundaries" in text
    assert "human/llm query" in text
    assert "rule-based / auditable by default" in text
    assert "strategydecision" in text
    assert "strategytrace" in text


def test_report_states_nexus_and_bomcp_are_not_campaign_authority():
    backend = build_backend_integration_summary()
    text = _flatten_text(backend).lower()

    assert "nexus is advisor/backend/evidence source, not campaign decision authority" in text
    assert "bo mcp is backend/tool, not campaign decision authority" in text


def test_report_states_learned_policy_is_not_default_live_override():
    report = build_system_validation_report()
    text = _flatten_text(report.learned_policy_summary).lower()

    assert "not default live override" in text
    assert "disabled by default" in text
    assert "no hard veto" in text
    assert "backend addition" in text


def test_report_includes_scenario_benchmark_aggregate_summary():
    runner = ScenarioRunner()
    aggregate = runner.aggregate(tuple(runner.run(spec) for spec in _scenarios()))
    report = build_system_validation_report(
        scenario_benchmark_summary=build_scenario_benchmark_summary(aggregate_summary=aggregate)
    )

    assert report.scenario_benchmark_summary["aggregate"]["total_scenarios"] == 10
    assert report.scenario_benchmark_summary["aggregate"]["pass_count"] == 10
    assert report.scenario_benchmark_summary["aggregate"]["unexpected_live_behavior_changes"] == 0


def test_report_includes_self_evolution_approval_boundary():
    text = _flatten_text(build_self_evolution_summary()).lower()

    assert "human/config-approved" in text
    assert "proposal/approval gated" in text
    assert "metadata does not alter default live behavior" in text
    assert "no auto-promotion" in text


def test_report_includes_safety_boundary_and_failure_attribution():
    text = _flatten_text(build_safety_boundary_summary()).lower()

    assert "not connected to rank_backends" in text
    assert "live strategy_selector" in text
    assert "scientific_negative is evidence" in text
    assert "hardware and measurement failures do not directly penalize optimizer backends" in text


def test_system_validation_report_does_not_import_or_call_live_selector_or_ranker():
    source = inspect.getsource(system_validation_report)

    assert "from app.services.strategy_selector" not in source
    assert "import app.services.strategy_selector" not in source
    assert "from app.optimization.backend_selection" not in source
    assert "rank_backends(" not in source
    assert "select_strategy(" not in source


def test_system_validation_report_markdown_contains_required_evidence_pack_claims():
    markdown = build_system_validation_report(
        scenario_benchmark_summary=build_scenario_benchmark_summary(
            aggregate_summary={"total_scenarios": 10, "pass_count": 10}
        )
    ).to_markdown().lower()

    assert "# helios architecture validation" in markdown
    assert "nexus is advisor/backend/evidence source" in markdown
    assert "learned policies are offline/shadow/canary/gated" in markdown
    assert "no-human closed-loop sdl path has offline tests" in markdown
