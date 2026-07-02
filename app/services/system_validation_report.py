"""System validation report builders for HELIOS architecture evidence.

This module is intentionally reporting-only. It does not import or call the
live strategy selector, backend ranking, optimizers, policy-evolution managers,
or runtime configuration mutators.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

REPORT_VERSION = "helios_system_validation_report_v1"


@dataclass(frozen=True)
class HELIOSSystemValidationReport:
    """Structured evidence pack for architecture and safety-boundary review."""

    architecture_summary: dict[str, Any]
    dynamic_strategy_summary: dict[str, Any]
    backend_integration_summary: dict[str, Any]
    no_human_closed_loop_summary: dict[str, Any]
    learned_policy_summary: dict[str, Any]
    self_evolution_summary: dict[str, Any]
    safety_boundary_summary: dict[str, Any]
    offline_test_summary: dict[str, Any]
    scenario_benchmark_summary: dict[str, Any]
    remaining_limitations: tuple[str, ...]
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    report_version: str = REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render a compact Markdown evidence pack."""
        sections = (
            ("Architecture", self.architecture_summary),
            ("Dynamic Strategy", self.dynamic_strategy_summary),
            ("Backend Integration", self.backend_integration_summary),
            ("No-Human Closed Loop", self.no_human_closed_loop_summary),
            ("Learned Policy", self.learned_policy_summary),
            ("Self Evolution", self.self_evolution_summary),
            ("Safety Boundary", self.safety_boundary_summary),
            ("Offline Tests", self.offline_test_summary),
            ("Scenario Benchmarks", self.scenario_benchmark_summary),
        )
        lines = [
            "# HELIOS Architecture Validation",
            "",
            f"- report_version: `{self.report_version}`",
            f"- generated_at: `{self.generated_at}`",
            "",
        ]
        for title, summary in sections:
            lines.append(f"## {title}")
            for key, value in summary.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        lines.append("## Remaining Limitations")
        for item in self.remaining_limitations:
            lines.append(f"- {item}")
        return "\n".join(lines).rstrip() + "\n"


def build_architecture_summary() -> dict[str, Any]:
    return {
        "claim": (
            "HELIOS is agent-native, hierarchical, graph/state-machine routed, "
            "role-based multi-agent controller."
        ),
        "controller_shape": "hierarchical graph/state-machine routing with specialist role boundaries",
        "agent_contract": "shared typed campaign state, traces, outcomes, and safety/proposal contracts",
        "decision_authority": "campaign controller and auditable strategy policy retain decision authority",
        "runtime_boundary": "reporting-only evidence; no runtime behavior is modified",
    }


def build_dynamic_strategy_summary() -> dict[str, Any]:
    return {
        "default_controller": "Live controller remains rule-based / auditable by default.",
        "decision_trace": "StrategyDecision and StrategyTrace record intent, mode, backend, evidence, proposals, outcome, and reward.",
        "dynamic_context": (
            "CampaignContext includes objective hierarchy, failure taxonomy, route, budget, "
            "data-quality, parameter-space, and prior-campaign context."
        ),
        "proposal_boundary": "route_switch, revise_space, objective transitions, and hypothesis actions remain proposal/evidence paths unless explicitly supported.",
    }


def build_backend_integration_summary() -> dict[str, Any]:
    return {
        "nexus_boundary": "Nexus is advisor/backend/evidence source, not campaign decision authority.",
        "nexus_runtime_path": (
            "HELIOS uses lightweight in-process Nexus optimization core/advisor paths where configured; "
            "server/API/MCP/platform/LLM Nexus components stay outside the default runtime path."
        ),
        "bomcp_boundary": "BO MCP is backend/tool, not the top-level controller.",
        "fallback_behavior": "Backend failures degrade through typed failure attribution and deterministic fallback paths.",
        "default_behavior": "Default BO MCP/Nexus/backend behavior remains unchanged by this report layer.",
    }


def build_offline_closed_loop_summary(
    *,
    test_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "claim": "No-human closed-loop SDL path has offline tests.",
        "offline_only": "tests use simulated observations, simulated execution, mocked backends, no hardware, and no network",
        "coverage": (
            "multi-round campaign loop, candidate generation, compile/safety/execute/outcome provenance, "
            "typed failures, replay records, and default behavior invariance"
        ),
    }
    if test_summary:
        summary.update(test_summary)
    return summary


def build_learned_policy_summary() -> dict[str, Any]:
    return {
        "default_boundary": "Learned policies are offline/shadow/canary/gated, not default live override.",
        "influence_boundary": "learned-policy score deltas require explicit SAFE_SOFT/canary gating and remain bounded",
        "disallowed_actions": "no hard veto, backend addition, action override, objective override, or space revision auto-apply",
        "default_behavior": "learned policy influence is disabled by default",
    }


def build_self_evolution_summary() -> dict[str, Any]:
    return {
        "approval_boundary": "Self-evolution is human/config-approved and proposal/approval gated.",
        "workflow": "trigger -> plan -> offline candidate -> shadow proposal/approval -> canary proposal/approval -> final proposal/approval",
        "metadata_boundary": "self-evolution metadata does not alter default live behavior",
        "non_goals": "no auto-training into live selector, no auto-promotion, no auto-application of tuning/structure/space proposals",
    }


def build_safety_boundary_summary() -> dict[str, Any]:
    return {
        "live_behavior": "Default live behavior is unchanged.",
        "space_revision": "Space revisions are approval-only and are not auto-applied.",
        "policy_evolution_boundary": "Policy evolution workflow is not connected to rank_backends or live strategy_selector execution.",
        "approval_gates": "safety gates and approval requirements are not weakened by reporting or metadata",
        "failure_attribution": "hardware and measurement failures do not directly penalize optimizer backends; scientific_negative is evidence, not backend failure",
    }


def build_scenario_benchmark_summary(
    *,
    aggregate_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "claim": "Deterministic offline scenario benchmark validates common no-human SDL campaign conditions.",
        "scenario_count": 10,
        "scenarios": (
            "tiny_data_baseline",
            "high_noise_measurement_drift",
            "constraint_heavy_campaign",
            "backend_unavailable_fallback",
            "scientific_negative_campaign",
            "plateau_to_pivot",
            "mechanism_validation",
            "generalization_transfer",
            "execution_instability_recovery",
            "legacy_fallback_campaign",
        ),
    }
    if aggregate_summary:
        summary["aggregate"] = dict(aggregate_summary)
    return summary


def build_offline_test_summary(
    *,
    closed_loop: dict[str, Any] | None = None,
    scenario_benchmarks: dict[str, Any] | None = None,
    policy_evolution: dict[str, Any] | None = None,
    learned_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "offline_closed_loop_sdl": dict(closed_loop or {}),
        "offline_scenario_benchmarks": dict(scenario_benchmarks or {}),
        "policy_evolution_workflow": dict(policy_evolution or {}),
        "learned_policy": dict(learned_policy or {}),
        "test_boundary": "offline tests do not require hardware, network, live BO MCP, or live Nexus services",
    }


def build_system_validation_report(
    *,
    offline_test_summary: dict[str, Any] | None = None,
    scenario_benchmark_summary: dict[str, Any] | None = None,
    remaining_limitations: tuple[str, ...] | None = None,
) -> HELIOSSystemValidationReport:
    offline = offline_test_summary or build_offline_test_summary()
    scenario = scenario_benchmark_summary or build_scenario_benchmark_summary()
    return HELIOSSystemValidationReport(
        architecture_summary=build_architecture_summary(),
        dynamic_strategy_summary=build_dynamic_strategy_summary(),
        backend_integration_summary=build_backend_integration_summary(),
        no_human_closed_loop_summary=build_offline_closed_loop_summary(test_summary=offline.get("offline_closed_loop_sdl")),
        learned_policy_summary=build_learned_policy_summary(),
        self_evolution_summary=build_self_evolution_summary(),
        safety_boundary_summary=build_safety_boundary_summary(),
        offline_test_summary=offline,
        scenario_benchmark_summary=scenario,
        remaining_limitations=remaining_limitations or _default_remaining_limitations(),
    )


def _default_remaining_limitations() -> tuple[str, ...]:
    return (
        "Offline tests validate deterministic simulated campaign behavior, not paper-level proof of real-world superiority.",
        "Real campaign data, shadow/canary outcomes, reward correlation, failure-rate comparison, ablations, and external benchmarks remain required for research-grade claims.",
        "Report summaries are evidence metadata and do not activate learned policies or self-evolution workflows.",
    )
