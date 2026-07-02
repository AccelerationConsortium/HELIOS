from __future__ import annotations

from app.services.strategy_models import CampaignContext, CampaignSnapshot, FailureEvent

AVAIL = {
    "lhs": True,
    "built_in": True,
    "random_sampling": True,
    "optuna_tpe": True,
    "optuna_cmaes": False,
    "scipy_de": False,
    "pymoo_nsga2": False,
    "nexus_lhs": True,
    "nexus_sobol": True,
    "nexus_tpe": True,
    "nexus_gp_bo": True,
    "nexus_turbo": True,
}


def snapshot(
    *,
    round_number: int,
    kpis: tuple[float, ...],
    context: CampaignContext | None = None,
    failures: tuple[FailureEvent, ...] = (),
    campaign_id: str = "replay",
    previous_intent: str | None = None,
) -> CampaignSnapshot:
    params = tuple({"x": i / max(len(kpis), 1), "y": (i % 3) / 3} for i in range(len(kpis)))
    return CampaignSnapshot(
        round_number=round_number,
        max_rounds=12,
        n_observations=len(kpis),
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=kpis,
        direction="maximize",
        available_backends=dict(AVAIL),
        last_batch_kpis=kpis[-3:],
        last_batch_params=params[-3:],
        best_kpi_so_far=max(kpis) if kpis else None,
        all_params=params,
        all_kpis=kpis,
        campaign_context=context,
        failure_events=failures,
        campaign_id=campaign_id,
        previous_intent=previous_intent,
    )


def tiny_data_to_baseline() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=1,
            kpis=(0.1,),
            context=CampaignContext(current_objective_level="baseline"),
        ),
        snapshot(
            round_number=2,
            kpis=(0.1, 0.15),
            context=CampaignContext(current_objective_level="baseline"),
            previous_intent="discover",
        ),
        snapshot(
            round_number=3,
            kpis=(0.1, 0.15, 0.21, 0.29, 0.35, 0.41),
            context=CampaignContext(current_objective_level="performance"),
            previous_intent="discover",
        ),
    ]


def high_noise_to_validation_stabilization() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=3,
            kpis=(1.0, 2.0, 0.2),
            context=CampaignContext(current_objective_level="data_quality"),
            failures=(FailureEvent(failure_type="measurement", reason="blank drift"),),
            previous_intent="optimize",
        )
    ]


def constraint_failure_to_space_revision() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=4,
            kpis=(0.1, 0.2, 0.25, 0.27),
            failures=(
                FailureEvent(
                    failure_type="constraint",
                    reason="voltage exceeded window",
                    backend_name="nexus_gp_bo",
                    params={"voltage": 3.4},
                    penalize_backend=True,
                ),
            ),
            previous_intent="optimize",
        )
    ]


def plateau_to_pivot_route_switch() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=8,
            kpis=tuple([0.1, 0.3, 0.5, 0.55] + [0.56] * 4),
            context=CampaignContext(
                current_objective_level="generalization",
                synthesis_routes=("electrodeposition", "gel"),
            ),
            previous_intent="optimize",
        )
    ]


def promising_best_to_mechanism_validation() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=6,
            kpis=tuple(0.1 + 0.03 * i for i in range(12)),
            context=CampaignContext(
                current_objective_level="mechanism",
                domain_hypotheses=(
                    "Fe stabilizes NiOOH",
                    "Ni vacancy suppresses degradation",
                ),
            ),
            previous_intent="optimize",
        )
    ]


def hardware_instability_recovery() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=5,
            kpis=(0.2, 0.24, 0.25, 0.26),
            failures=(
                FailureEvent(
                    failure_type="hardware",
                    reason="pump pressure oscillation",
                    backend_name="nexus_gp_bo",
                    penalize_backend=False,
                ),
            ),
            previous_intent="optimize",
        )
    ]


def scientific_negative_hypothesis_update() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=7,
            kpis=tuple(0.2 + 0.01 * i for i in range(8)),
            context=CampaignContext(
                current_objective_level="performance",
                domain_hypotheses=("surface phase controls activity",),
            ),
            failures=(
                FailureEvent(
                    failure_type="scientific_negative",
                    reason="clean negative result contradicts active hypothesis",
                    backend_name="nexus_tpe",
                    penalize_backend=False,
                ),
            ),
            previous_intent="optimize",
        )
    ]


def measurement_drift_stabilization() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=6,
            kpis=(1.1, 0.4, 1.3, 0.35, 1.2, 0.45),
            context=CampaignContext(current_objective_level="data_quality"),
            failures=(
                FailureEvent(
                    failure_type="measurement",
                    reason="reference electrode drift",
                    backend_name="nexus_gp_bo",
                    penalize_backend=False,
                ),
            ),
            previous_intent="optimize",
        )
    ]


def performance_to_mechanism_validation() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=9,
            kpis=tuple(0.15 + 0.025 * i for i in range(12)),
            context=CampaignContext(
                current_objective_level="mechanism",
                domain_hypotheses=("adsorbate coverage mediates selectivity",),
            ),
            previous_intent="optimize",
        )
    ]


def generalization_to_transfer() -> list[CampaignSnapshot]:
    return [
        snapshot(
            round_number=10,
            kpis=tuple(0.3 + 0.005 * i for i in range(12)),
            context=CampaignContext(
                current_objective_level="generalization",
                synthesis_routes=("electrodeposition", "sol-gel"),
                prior_campaigns=({"campaign_id": "prior-1", "material_family": "NiFe"},),
            ),
            previous_intent="pivot",
        )
    ]


def all_replay_scenarios() -> list[CampaignSnapshot]:
    return (
        tiny_data_to_baseline()
        + high_noise_to_validation_stabilization()
        + constraint_failure_to_space_revision()
        + plateau_to_pivot_route_switch()
        + promising_best_to_mechanism_validation()
        + hardware_instability_recovery()
        + scientific_negative_hypothesis_update()
        + measurement_drift_stabilization()
        + performance_to_mechanism_validation()
        + generalization_to_transfer()
    )
