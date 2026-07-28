"""Nexus early-stage advisory adapter for imperfect campaign data.

Nexus owns messy-data intake and early-stage system characterization. HELIOS
keeps campaign authority by adapting Nexus reports into local evidence, mode
hints, action-space adjustments, and audit metadata.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.core.config import get_settings
from app.services.campaign_mode import CampaignMode
from app.services.strategy_models import EvidenceItem

logger = logging.getLogger(__name__)

SUPPORTED_CONTRACT_VERSIONS = {"early_stage_system_characterization.v1"}


class NexusEarlyStageErrorType(StrEnum):
    """Typed Nexus early-stage failure classes for safe fallback decisions."""

    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    UNSUPPORTED_CONTRACT_VERSION = "unsupported_contract_version"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True)
class NexusEarlyStageResponse:
    """Typed result from a Nexus early-stage endpoint.

    ``ok=False`` is expected for fallback paths. Callers should inspect
    ``error_type`` and continue with local HELIOS policy when automation is not
    safe.
    """

    ok: bool
    endpoint: str
    status_code: int | None = None
    campaign_id: str | None = None
    report: dict[str, Any] | None = None
    intake_report: dict[str, Any] | None = None
    observations: tuple[dict[str, Any], ...] = ()
    parameter_specs: tuple[dict[str, Any], ...] = ()
    error_type: NexusEarlyStageErrorType | None = None
    error_message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def contract_version(self) -> str | None:
        return _string_or_none((self.report or {}).get("contract_version"))

    @property
    def recommended_campaign_mode(self) -> str | None:
        return _string_or_none((self.report or {}).get("recommended_campaign_mode"))

    @property
    def confidence(self) -> float | None:
        value = (self.report or {}).get("confidence")
        return float(value) if isinstance(value, int | float) else None

    @property
    def risk_flags(self) -> tuple[str, ...]:
        return _string_tuple((self.report or {}).get("risk_flags"))


class NexusEarlyStageClient:
    """Small REST wrapper around Nexus early-stage endpoints."""

    def __init__(self, base_url: str | None = None, timeout_seconds: float | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.nexus_url).rstrip("/")
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else settings.nexus_timeout_seconds
        )

    def intake(self, payload: dict[str, Any]) -> NexusEarlyStageResponse:
        """Call ``POST /early-stage/intake``."""
        return self._post("/early-stage/intake", payload)

    def analyze(self, payload: dict[str, Any]) -> NexusEarlyStageResponse:
        """Call ``POST /early-stage/analyze``."""
        return self._post("/early-stage/analyze", payload)

    def report(self, campaign_id: str) -> NexusEarlyStageResponse:
        """Call ``GET /campaigns/{campaign_id}/early-stage-report``."""
        escaped = quote(campaign_id, safe="")
        return self._request(
            "GET",
            f"/campaigns/{escaped}/early-stage-report",
            campaign_id=campaign_id,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> NexusEarlyStageResponse:
        campaign_id = _string_or_none(payload.get("campaign_id"))
        return self._request("POST", path, payload=payload, campaign_id=campaign_id)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        campaign_id: str | None = None,
    ) -> NexusEarlyStageResponse:
        endpoint = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            endpoint,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
                decoded = json.loads(raw_body) if raw_body else {}
                if not isinstance(decoded, dict):
                    raise TypeError("Nexus early-stage response must be a JSON object.")
                return self._build_response(
                    endpoint=endpoint,
                    status_code=response.status,
                    raw=decoded,
                    campaign_id=campaign_id,
                )
        except HTTPError as exc:
            raw = _decode_http_error(exc)
            error_type = (
                NexusEarlyStageErrorType.BAD_REQUEST
                if exc.code == 400
                else NexusEarlyStageErrorType.NOT_FOUND
                if exc.code == 404
                else NexusEarlyStageErrorType.UNAVAILABLE
            )
            response = NexusEarlyStageResponse(
                ok=False,
                endpoint=endpoint,
                status_code=exc.code,
                campaign_id=campaign_id,
                error_type=error_type,
                error_message=_error_message(raw, exc.reason),
                raw=raw,
            )
            self._log_response(response)
            return response
        except TimeoutError:
            response = NexusEarlyStageResponse(
                ok=False,
                endpoint=endpoint,
                campaign_id=campaign_id,
                error_type=NexusEarlyStageErrorType.TIMEOUT,
                error_message=f"Nexus early-stage request timed out after {self.timeout_seconds}s.",
            )
            self._log_response(response)
            return response
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            error_type = (
                NexusEarlyStageErrorType.TIMEOUT
                if isinstance(reason, TimeoutError)
                else NexusEarlyStageErrorType.UNAVAILABLE
            )
            response = NexusEarlyStageResponse(
                ok=False,
                endpoint=endpoint,
                campaign_id=campaign_id,
                error_type=error_type,
                error_message=str(reason),
            )
            self._log_response(response)
            return response
        except (json.JSONDecodeError, TypeError) as exc:
            response = NexusEarlyStageResponse(
                ok=False,
                endpoint=endpoint,
                campaign_id=campaign_id,
                error_type=NexusEarlyStageErrorType.INVALID_RESPONSE,
                error_message=str(exc),
            )
            self._log_response(response)
            return response

    def _build_response(
        self,
        *,
        endpoint: str,
        status_code: int,
        raw: dict[str, Any],
        campaign_id: str | None,
    ) -> NexusEarlyStageResponse:
        report = _report_from_raw(raw)
        contract_version = _string_or_none((report or {}).get("contract_version"))
        error_type = None
        error_message = ""
        ok = True
        if contract_version and contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            ok = False
            error_type = NexusEarlyStageErrorType.UNSUPPORTED_CONTRACT_VERSION
            error_message = f"Unsupported Nexus early-stage contract: {contract_version}."

        response = NexusEarlyStageResponse(
            ok=ok,
            endpoint=endpoint,
            status_code=status_code,
            campaign_id=campaign_id or _string_or_none(raw.get("campaign_id")),
            report=report,
            intake_report=_dict_or_none(raw.get("intake_report")),
            observations=_dict_tuple(raw.get("observations")),
            parameter_specs=_dict_tuple(raw.get("parameter_specs")),
            error_type=error_type,
            error_message=error_message,
            raw=raw,
        )
        self._log_response(response)
        return response

    def _log_response(self, response: NexusEarlyStageResponse) -> None:
        logger.info(
            "Nexus early-stage response: campaign_id=%s contract=%s mode=%s risks=%s ok=%s error=%s",
            response.campaign_id,
            response.contract_version,
            response.recommended_campaign_mode,
            list(response.risk_flags)[:5],
            response.ok,
            response.error_type,
        )


@dataclass(frozen=True)
class ActionSpaceAdjustment:
    """Advisory adjustment derived from Nexus early-stage characterization."""

    adjustment_type: str
    source_field: str
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)
    reject_by_default: bool = False


@dataclass(frozen=True)
class NexusEarlyStageAdvice:
    """HELIOS-native view of a Nexus early-stage report."""

    evidence: tuple[EvidenceItem, ...] = ()
    campaign_mode_hint: CampaignMode | None = None
    nexus_campaign_mode: str | None = None
    action_space_adjustments: tuple[ActionSpaceAdjustment, ...] = ()
    objective_candidates: tuple[dict[str, Any], ...] = ()
    operator_messages: tuple[str, ...] = ()
    audit_metadata: dict[str, Any] = field(default_factory=dict)
    ordinary_bo_allowed: bool = True
    requires_operator_approval: bool = False


class NexusEarlyStageAdapter:
    """Convert Nexus characterization reports into HELIOS advisory artifacts."""

    def adapt(
        self,
        report: dict[str, Any] | NexusEarlyStageResponse | None,
        *,
        endpoint_used: str | None = None,
        campaign_id: str | None = None,
    ) -> NexusEarlyStageAdvice:
        response = report if isinstance(report, NexusEarlyStageResponse) else None
        raw_report = response.report if response is not None else report
        if not isinstance(raw_report, dict):
            return NexusEarlyStageAdvice(
                requires_operator_approval=True,
                audit_metadata={
                    "nexus_endpoint_used": endpoint_used or getattr(response, "endpoint", None),
                    "campaign_id": campaign_id or getattr(response, "campaign_id", None),
                    "error_type": getattr(response, "error_type", None),
                    "error_message": getattr(response, "error_message", ""),
                },
            )

        contract_version = _string_or_none(raw_report.get("contract_version"))
        if contract_version and contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            return NexusEarlyStageAdvice(
                requires_operator_approval=True,
                audit_metadata={
                    "nexus_endpoint_used": endpoint_used or getattr(response, "endpoint", None),
                    "contract_version": contract_version,
                    "campaign_id": campaign_id or getattr(response, "campaign_id", None),
                    "error_type": NexusEarlyStageErrorType.UNSUPPORTED_CONTRACT_VERSION,
                    "error_message": (
                        f"Unsupported Nexus early-stage contract: {contract_version}."
                    ),
                },
            )
        risk_flags = _string_tuple(raw_report.get("risk_flags"))
        recommendations = _dict_tuple(raw_report.get("diagnostic_recommendations"))
        mode = _select_campaign_mode(raw_report)
        evidence = _build_strategy_evidence(raw_report, mode, risk_flags, recommendations)
        adjustments = _build_action_space_adjustments(raw_report, risk_flags)
        objective_candidates = _dict_tuple(raw_report.get("candidate_kpis"))
        confidence = _confidence(raw_report)
        ordinary_bo_allowed = _ordinary_bo_allowed(raw_report, risk_flags, confidence)

        operator_messages = tuple(
            str(item)
            for item in raw_report.get("insights", ())
            if isinstance(item, str) and item.strip()
        )
        if not ordinary_bo_allowed:
            operator_messages = (
                *operator_messages,
                "Nexus early-stage characterization gates ordinary final-KPI BO.",
            )
        if confidence < 0.5:
            operator_messages = (
                *operator_messages,
                "Nexus confidence is low; treat recommendations as advisory only.",
            )

        return NexusEarlyStageAdvice(
            evidence=evidence,
            campaign_mode_hint=mode,
            nexus_campaign_mode=_string_or_none(raw_report.get("recommended_campaign_mode")),
            action_space_adjustments=adjustments,
            objective_candidates=objective_candidates,
            operator_messages=operator_messages,
            audit_metadata={
                "nexus_endpoint_used": endpoint_used or getattr(response, "endpoint", None),
                "contract_version": contract_version,
                "campaign_id": campaign_id or getattr(response, "campaign_id", None),
                "recommended_campaign_mode": raw_report.get("recommended_campaign_mode"),
                "confidence": confidence,
                "risk_flags": list(risk_flags),
                "top_diagnostic_recommendations": [
                    rec for rec in recommendations[:3]
                ],
                "action_space_adjustments": [
                    {
                        "adjustment_type": adj.adjustment_type,
                        "source_field": adj.source_field,
                        "reject_by_default": adj.reject_by_default,
                        "payload": adj.payload,
                    }
                    for adj in adjustments
                ],
            },
            ordinary_bo_allowed=ordinary_bo_allowed,
            requires_operator_approval=(confidence < 0.5 or not ordinary_bo_allowed),
        )


def _select_campaign_mode(report: dict[str, Any]) -> CampaignMode:
    mode = _string_or_none(report.get("recommended_campaign_mode"))
    risk_flags = set(_string_tuple(report.get("risk_flags")))

    if "objective_missing" in risk_flags or "low_confidence_objective_candidates" in risk_flags:
        return CampaignMode.OBJECTIVE_DISCOVERY
    if "poor_controllability" in risk_flags or "target_reachability_low" in risk_flags:
        return CampaignMode.CONTROLLABILITY_MAPPING
    if "hardware_failures_dominate" in risk_flags or "hardware_design_changed" in risk_flags:
        return CampaignMode.HARDWARE_FEASIBILITY_DISCOVERY
    if risk_flags & {"low_data_quality", "batch_effect_detected", "instrument_drift_detected"}:
        return CampaignMode.DATA_QUALITY_DIAGNOSTIC

    mapping = {
        "optimization_ready": CampaignMode.BO_OPTIMIZATION,
        "early_stage_system_characterization": (
            CampaignMode.EARLY_STAGE_SYSTEM_CHARACTERIZATION
        ),
        "hardware_feasibility_discovery": CampaignMode.HARDWARE_FEASIBILITY_DISCOVERY,
        "controllability_mapping": CampaignMode.CONTROLLABILITY_MAPPING,
        "data_quality_diagnostic": CampaignMode.DATA_QUALITY_DIAGNOSTIC,
        "objective_discovery": CampaignMode.OBJECTIVE_DISCOVERY,
    }
    return mapping.get(mode or "", CampaignMode.EARLY_STAGE_SYSTEM_CHARACTERIZATION)


def _build_strategy_evidence(
    report: dict[str, Any],
    mode: CampaignMode,
    risk_flags: tuple[str, ...],
    recommendations: tuple[dict[str, Any], ...],
) -> tuple[EvidenceItem, ...]:
    evidence: list[EvidenceItem] = []
    confidence = _confidence(report)
    for flag in risk_flags:
        evidence.append(
            EvidenceItem(
                signal_name=f"nexus_early_stage_risk_{flag}",
                signal_value=confidence,
                target_action=_risk_flag_target_action(flag),
                contribution=_risk_flag_contribution(flag, confidence),
                description=f"Nexus early-stage risk flag '{flag}' supports {mode.value}.",
            )
        )
    for rec in recommendations[:3]:
        action_type = _string_or_none(rec.get("action_type")) or "diagnostic"
        priority = _priority(rec)
        evidence.append(
            EvidenceItem(
                signal_name=f"nexus_early_stage_recommendation_{action_type}",
                signal_value=priority,
                target_action=_recommendation_target_action(action_type),
                contribution=round(min(0.2, 0.05 + priority * 0.1), 4),
                description=f"Nexus recommends '{action_type}' for early-stage characterization.",
            )
        )
    if not evidence and mode == CampaignMode.BO_OPTIMIZATION:
        evidence.append(
            EvidenceItem(
                signal_name="nexus_early_stage_optimization_ready",
                signal_value=confidence,
                target_action="exploit",
                contribution=round(0.05 * confidence, 4),
                description="Nexus reports optimization_ready; HELIOS may continue guarded BO.",
            )
        )
    return tuple(evidence)


def _build_action_space_adjustments(
    report: dict[str, Any],
    risk_flags: tuple[str, ...],
) -> tuple[ActionSpaceAdjustment, ...]:
    adjustments: list[ActionSpaceAdjustment] = []
    feasibility = _dict_or_none(report.get("feasibility_summary")) or {}
    for zone in _dict_tuple(feasibility.get("danger_zones")):
        adjustments.append(
            ActionSpaceAdjustment(
                adjustment_type="reject_or_annotate_danger_zone",
                source_field="feasibility_summary.danger_zones",
                reason="Nexus identified an early-stage failure danger zone.",
                payload=zone,
                reject_by_default=True,
            )
        )

    target_feasibility = report.get("target_feasibility_summary")
    target_rows = (
        _dict_tuple(target_feasibility)
        if isinstance(target_feasibility, list | tuple)
        else _dict_tuple([target_feasibility] if isinstance(target_feasibility, dict) else [])
    )
    for row in target_rows:
        lower = row.get("infeasible_target_min")
        upper = row.get("infeasible_target_max")
        if lower is None and upper is None:
            continue
        adjustments.append(
            ActionSpaceAdjustment(
                adjustment_type="narrow_target_range",
                source_field="target_feasibility_summary",
                reason="Nexus marked part of the target range as infeasible.",
                payload=row,
                reject_by_default="target_reachability_low" in risk_flags,
            )
        )

    hardware = _dict_or_none(report.get("hardware_summary")) or {}
    worst_design_id = _string_or_none(hardware.get("worst_design_id"))
    if worst_design_id:
        adjustments.append(
            ActionSpaceAdjustment(
                adjustment_type="route_by_hardware_design",
                source_field="hardware_summary.worst_design_id",
                reason="Nexus identified a weak hardware or reactor design.",
                payload={"worst_design_id": worst_design_id},
                reject_by_default="hardware_failures_dominate" in risk_flags,
            )
        )
    return tuple(adjustments)


def _ordinary_bo_allowed(
    report: dict[str, Any],
    risk_flags: tuple[str, ...],
    confidence: float,
) -> bool:
    mode = _string_or_none(report.get("recommended_campaign_mode"))
    blockers = {
        "poor_controllability",
        "objective_missing",
        "target_reachability_low",
        "hardware_failures_dominate",
    }
    if blockers & set(risk_flags):
        return False
    if mode and mode != "optimization_ready" and confidence >= 0.5:
        return False
    return True


def _risk_flag_target_action(flag: str) -> str:
    if flag in {"objective_missing", "low_confidence_objective_candidates"}:
        return "objective_discovery"
    if flag in {"poor_controllability", "target_reachability_low"}:
        return "controllability_mapping"
    if flag in {"hardware_failures_dominate", "hardware_design_changed"}:
        return "hardware_feasibility"
    if flag in {"low_data_quality", "batch_effect_detected", "instrument_drift_detected"}:
        return "data_quality_diagnostic"
    return "diagnose"


def _risk_flag_contribution(flag: str, confidence: float) -> float:
    high_impact = {
        "poor_controllability",
        "objective_missing",
        "target_reachability_low",
        "hardware_failures_dominate",
    }
    base = 0.18 if flag in high_impact else 0.1
    return round(base * max(0.2, confidence), 4)


def _recommendation_target_action(action_type: str) -> str:
    mapping = {
        "run_controllability_mapping": "controllability_mapping",
        "map_hardware_feasibility": "hardware_feasibility",
        "run_data_quality_diagnostic": "data_quality_diagnostic",
        "run_objective_discovery": "objective_discovery",
        "shrink_or_annotate_action_space": "revise_space",
        "validate_target_reachability": "controllability_mapping",
        "proceed_with_guarded_optimization": "exploit",
    }
    return mapping.get(action_type, "diagnose")


def _priority(recommendation: dict[str, Any]) -> float:
    raw = recommendation.get("priority", recommendation.get("score", 0.5))
    if isinstance(raw, int | float):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(raw.lower(), 0.5)
    return 0.5


def _report_from_raw(raw: dict[str, Any]) -> dict[str, Any] | None:
    report = raw.get("analysis_report", raw.get("report"))
    return _dict_or_none(report)


def _decode_http_error(exc: HTTPError) -> dict[str, Any]:
    try:
        raw_body = exc.read().decode("utf-8")
        decoded = json.loads(raw_body) if raw_body else {}
        return decoded if isinstance(decoded, dict) else {"detail": decoded}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def _error_message(raw: dict[str, Any], fallback: str) -> str:
    detail = raw.get("detail") or raw.get("error") or raw.get("message")
    return str(detail or fallback)


def _confidence(report: dict[str, Any]) -> float:
    value = report.get("confidence", 0.5)
    return max(0.0, min(1.0, float(value))) if isinstance(value, int | float) else 0.5


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None
