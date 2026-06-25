"""Optimization intelligence layer for HELIOS strategy selection.

This module turns optional Nexus signals into local, structured evidence for
the HELIOS optimizer.  It deliberately keeps Nexus as an advisory dependency:
if Nexus is unavailable or returns partial data, strategy selection continues
with local diagnostics and RL/rule-based logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.nexus_advisor import NexusAdvisor
from app.services.strategy_models import CampaignSnapshot, EvidenceItem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationIntelligence:
    """Structured optimization advice consumed by the strategy selector."""

    evidence: tuple[EvidenceItem, ...] = ()
    weight_adjustments: dict[str, float] = field(default_factory=dict)
    recommended_phase: str | None = None
    recommended_backends: tuple[str, ...] = ()  # fingerprint-driven soft bias (Δ2)
    similar_campaigns: tuple[dict[str, Any], ...] = ()
    sources: tuple[str, ...] = ()

    @property
    def has_signal(self) -> bool:
        return bool(
            self.evidence
            or self.weight_adjustments
            or self.recommended_phase
            or self.recommended_backends
            or self.similar_campaigns
        )


class OptimizationIntelligenceAdvisor:
    """Adapter that enriches HELIOS optimization decisions with Nexus advice."""

    def __init__(self, nexus: NexusAdvisor | None = None) -> None:
        self._nexus = nexus or NexusAdvisor()

    def advise(
        self,
        snapshot: CampaignSnapshot,
        *,
        campaign_id: str | None = None,
        tracker_state: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> OptimizationIntelligence:
        """Return optional cross-campaign and causal advice for a snapshot."""
        resolved_campaign_id = campaign_id or getattr(snapshot, "nexus_campaign_id", None) or "default"
        causal_data, var_names = _build_causal_inputs(snapshot)

        evidence: list[EvidenceItem] = []
        sources: list[str] = []
        weight_adjustments: dict[str, float] = {}
        recommended_phase: str | None = None
        similar_campaigns: tuple[dict[str, Any], ...] = ()

        try:
            insights = self._nexus.get_enhanced_diagnostics(
                campaign_id=resolved_campaign_id,
                causal_data=causal_data,
                var_names=var_names,
                tracker_state=tracker_state,
            )
            if insights is not None:
                sources.append("nexus_diagnostics")
                evidence.extend(_evidence_from_causal_edges(insights.causal_edges))
                evidence.extend(_evidence_from_hypotheses(insights.hypotheses))
        except Exception:
            logger.debug("Optimization intelligence diagnostics skipped", exc_info=True)

        try:
            meta = self._nexus.get_meta_learning_advice(resolved_campaign_id)
            if meta is not None:
                sources.append("nexus_meta_learning")
                weight_adjustments = dict(meta.weight_adjustments)
                recommended_phase = meta.recommended_phase
                if weight_adjustments:
                    evidence.append(_evidence_from_weight_adjustments(weight_adjustments))
                if recommended_phase:
                    evidence.append(EvidenceItem(
                        signal_name="nexus_recommended_phase",
                        signal_value=None,
                        target_action=_phase_to_action(recommended_phase),
                        contribution=0.05,
                        description=f"Nexus recommends phase '{recommended_phase}' from similar campaigns.",
                    ))
        except Exception:
            logger.debug("Optimization intelligence meta-learning skipped", exc_info=True)

        if record_id:
            try:
                similar = self._nexus.get_similar_experiments(
                    resolved_campaign_id,
                    record_id,
                    top_k=5,
                )
                if similar:
                    sources.append("nexus_similarity")
                    similar_campaigns = tuple(similar)
                    evidence.append(EvidenceItem(
                        signal_name="nexus_similar_experiments",
                        signal_value=float(len(similar)),
                        target_action="exploit",
                        contribution=min(0.12, 0.02 * len(similar)),
                        description=f"Nexus found {len(similar)} similar historical experiments for transfer.",
                    ))
            except Exception:
                logger.debug("Optimization intelligence similarity skipped", exc_info=True)

        # Fingerprint-driven backend recommendation (Δ2, in-process Nexus profiler).
        recommended_backends: tuple[str, ...] = ()
        try:
            from app.optimization.recommendation import recommend_backends

            recommended_backends = recommend_backends(snapshot)
            if recommended_backends:
                sources.append("nexus_fingerprint")
        except Exception:
            logger.debug("Backend recommendation skipped", exc_info=True)

        return OptimizationIntelligence(
            evidence=tuple(evidence),
            weight_adjustments=weight_adjustments,
            recommended_phase=recommended_phase,
            recommended_backends=recommended_backends,
            similar_campaigns=similar_campaigns,
            sources=tuple(dict.fromkeys(sources)),
        )


def _build_causal_inputs(
    snapshot: CampaignSnapshot,
) -> tuple[list[list[float]] | None, list[str] | None]:
    if not snapshot.all_params or not snapshot.all_kpis:
        return None, None

    sample = snapshot.all_params[0]
    numeric_keys = [
        key for key in sorted(sample)
        if isinstance(sample.get(key), (int, float))
    ]
    if not numeric_keys:
        return None, None

    rows = [
        [float(params.get(key, 0.0)) for key in numeric_keys] + [float(kpi)]
        for params, kpi in zip(snapshot.all_params, snapshot.all_kpis, strict=False)
    ]
    return rows, numeric_keys + ["kpi"]


def _evidence_from_causal_edges(edges: tuple[Any, ...]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for edge in edges:
        strength = float(getattr(edge, "strength", 0.0))
        if strength <= 0.3:
            continue
        target_action = "exploit" if strength > 0.7 else "explore"
        evidence.append(EvidenceItem(
            signal_name=f"nexus_causal_{edge.source}_to_{edge.target}",
            signal_value=strength,
            target_action=target_action,
            contribution=round(strength * 0.15, 4),
            description=(
                f"Nexus causal edge {edge.source}->{edge.target} "
                f"strength={strength:.2f} supports {target_action}."
            ),
        ))
    return evidence


def _evidence_from_hypotheses(hypotheses: tuple[Any, ...]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for hyp in hypotheses:
        status = str(getattr(hyp, "status", "")).upper()
        evidence_count = int(getattr(hyp, "evidence_count", 0))
        if status not in {"SUPPORTED", "TESTING"} or evidence_count <= 0:
            continue
        contribution = 0.08 if status == "SUPPORTED" else 0.04
        evidence.append(EvidenceItem(
            signal_name=f"nexus_hypothesis_{status.lower()}",
            signal_value=float(evidence_count),
            target_action="exploit" if status == "SUPPORTED" else "explore",
            contribution=contribution,
            description=(
                f"Nexus hypothesis '{getattr(hyp, 'statement', '')}' is {status.lower()} "
                f"with {evidence_count} evidence items."
            ),
        ))
    return evidence


def _evidence_from_weight_adjustments(adjustments: dict[str, float]) -> EvidenceItem:
    info_gain = adjustments.get("w_info_gain", 0.0)
    improvement = adjustments.get("w_improvement", 0.0)
    risk = adjustments.get("w_risk", 0.0)
    target_action = "explore" if info_gain >= improvement else "exploit"
    if risk > max(info_gain, improvement):
        target_action = "stabilize"
    contribution = min(0.15, abs(info_gain) + abs(improvement) + abs(risk))
    return EvidenceItem(
        signal_name="nexus_weight_adjustments",
        signal_value=round(contribution, 4),
        target_action=target_action,
        contribution=round(contribution, 4),
        description=f"Nexus meta-learning adjusted utility weights: {adjustments}.",
    )


def _phase_to_action(phase: str) -> str:
    normalized = phase.lower().strip()
    if "explor" in normalized or normalized in {"cold_start", "learning"}:
        return "explore"
    if "exploit" in normalized:
        return "exploit"
    if "refin" in normalized:
        return "refine"
    if "stabil" in normalized or "replic" in normalized:
        return "stabilize"
    return "explore"
