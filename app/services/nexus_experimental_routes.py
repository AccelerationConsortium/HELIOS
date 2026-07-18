"""Typed REST boundary for Nexus experimental-route intelligence.

Nexus characterizes route evidence and returns advisory-only reports.  This
module deliberately stops at that boundary: route selection and live campaign
mutation belong to HELIOS (see :mod:`experimental_route_policy`).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_CONTRACT_VERSIONS = {"experimental_route_intelligence.v1"}
REQUIRED_AUTHORITY = "advisory_only"
MAX_EXPERIMENTAL_ROUTE_OBSERVATIONS = 10_000
MAX_NEXUS_RESPONSE_BYTES = 8 * 1024 * 1024


class NexusExperimentalRouteErrorType(StrEnum):
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNSUPPORTED_CONTRACT_VERSION = "unsupported_contract_version"
    INVALID_AUTHORITY = "invalid_authority"
    INVALID_RESPONSE = "invalid_response"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class NexusExperimentalRouteResponse:
    ok: bool
    endpoint: str
    status_code: int | None = None
    campaign_id: str | None = None
    report: dict[str, Any] | None = None
    error_type: NexusExperimentalRouteErrorType | None = None
    error_message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def contract_version(self) -> str | None:
        value = (self.report or {}).get("contract_version")
        return str(value) if value is not None else None

    @property
    def authority(self) -> str | None:
        value = (self.report or {}).get("authority")
        return str(value) if value is not None else None


class NexusExperimentalRouteClient:
    """Small fail-closed client for ``POST /experimental-routes/analyze``."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.nexus_url).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else settings.nexus_timeout_seconds
        )
        self.api_key = api_key if api_key is not None else settings.nexus_api_key

    def analyze(self, payload: dict[str, Any]) -> NexusExperimentalRouteResponse:
        endpoint = f"{self.base_url}/experimental-routes/analyze"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        campaign_id = _text(payload.get("campaign_id"))
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_bytes = response.read(MAX_NEXUS_RESPONSE_BYTES + 1)
                if len(raw_bytes) > MAX_NEXUS_RESPONSE_BYTES:
                    raise TypeError("Nexus experimental-route response exceeded 8 MiB.")
                raw_body = raw_bytes.decode("utf-8")
                decoded = json.loads(raw_body) if raw_body else {}
                if not isinstance(decoded, dict):
                    raise TypeError("Nexus experimental-route response must be a JSON object.")
                return self._build_response(
                    endpoint=endpoint,
                    status_code=response.status,
                    raw=decoded,
                    campaign_id=campaign_id,
                )
        except HTTPError as exc:
            raw = _decode_http_error(exc)
            error_type = {
                400: NexusExperimentalRouteErrorType.BAD_REQUEST,
                401: NexusExperimentalRouteErrorType.UNAUTHORIZED,
                413: NexusExperimentalRouteErrorType.PAYLOAD_TOO_LARGE,
                422: NexusExperimentalRouteErrorType.BAD_REQUEST,
                429: NexusExperimentalRouteErrorType.RATE_LIMITED,
            }.get(exc.code, NexusExperimentalRouteErrorType.UNAVAILABLE)
            return self._failed(
                endpoint,
                campaign_id,
                error_type,
                _error_message(raw, exc.reason),
                status_code=exc.code,
                raw=raw,
            )
        except TimeoutError:
            return self._failed(
                endpoint,
                campaign_id,
                NexusExperimentalRouteErrorType.TIMEOUT,
                f"Nexus experimental-route request timed out after {self.timeout_seconds}s.",
            )
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            error_type = (
                NexusExperimentalRouteErrorType.TIMEOUT
                if isinstance(reason, TimeoutError)
                else NexusExperimentalRouteErrorType.UNAVAILABLE
            )
            return self._failed(endpoint, campaign_id, error_type, str(reason))
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            return self._failed(
                endpoint,
                campaign_id,
                NexusExperimentalRouteErrorType.INVALID_RESPONSE,
                str(exc),
            )

    def _build_response(
        self,
        *,
        endpoint: str,
        status_code: int,
        raw: dict[str, Any],
        campaign_id: str | None,
    ) -> NexusExperimentalRouteResponse:
        report = raw.get("report")
        if not isinstance(report, dict):
            return self._failed(
                endpoint,
                campaign_id,
                NexusExperimentalRouteErrorType.INVALID_RESPONSE,
                "Nexus response is missing a report object.",
                status_code=status_code,
                raw=raw,
            )
        contract = _text(report.get("contract_version"))
        if contract not in SUPPORTED_CONTRACT_VERSIONS:
            return self._failed(
                endpoint,
                campaign_id,
                NexusExperimentalRouteErrorType.UNSUPPORTED_CONTRACT_VERSION,
                f"Unsupported Nexus experimental-route contract: {contract!r}.",
                status_code=status_code,
                raw=raw,
            )
        authority = _text(report.get("authority"))
        if authority != REQUIRED_AUTHORITY:
            return self._failed(
                endpoint,
                campaign_id,
                NexusExperimentalRouteErrorType.INVALID_AUTHORITY,
                f"Nexus report authority must be {REQUIRED_AUTHORITY!r}, got {authority!r}.",
                status_code=status_code,
                raw=raw,
            )
        result = NexusExperimentalRouteResponse(
            ok=True,
            endpoint=endpoint,
            status_code=status_code,
            campaign_id=campaign_id or _text(report.get("campaign_id")),
            report=dict(report),
            raw=raw,
        )
        self._log(result)
        return result

    def _failed(
        self,
        endpoint: str,
        campaign_id: str | None,
        error_type: NexusExperimentalRouteErrorType,
        message: str,
        *,
        status_code: int | None = None,
        raw: dict[str, Any] | None = None,
    ) -> NexusExperimentalRouteResponse:
        result = NexusExperimentalRouteResponse(
            ok=False,
            endpoint=endpoint,
            status_code=status_code,
            campaign_id=campaign_id,
            error_type=error_type,
            error_message=message,
            raw=dict(raw or {}),
        )
        self._log(result)
        return result

    @staticmethod
    def _log(response: NexusExperimentalRouteResponse) -> None:
        logger.info(
            "Nexus experimental-route response: campaign=%s ok=%s contract=%s authority=%s error=%s",
            response.campaign_id,
            response.ok,
            response.contract_version,
            response.authority,
            response.error_type,
        )


def build_experimental_route_payload(
    *,
    campaign_id: str,
    graph: dict[str, Any],
    observations: list[dict[str, Any]],
    objective: str,
    direction: str,
    available_capabilities: list[str] | None,
) -> dict[str, Any]:
    """Build only fields accepted by Nexus's strict v1 request contract."""
    nodes = []
    for raw in graph.get("nodes", []) or []:
        if not isinstance(raw, dict):
            continue
        nodes.append({
            key: raw[key]
            for key in (
                "node_id", "label", "kind", "parameter_space", "protocol_ref",
                "required_capabilities", "expected_cost", "expected_duration_s",
                "safety_risk", "prior_weight", "prior_evidence", "metadata",
            )
            if key in raw
        })
    transitions = []
    for raw in graph.get("transitions", []) or []:
        if not isinstance(raw, dict):
            continue
        transitions.append({
            key: raw[key]
            for key in (
                "source_id", "target_id", "switch_cost", "switch_duration_s",
                "approval_required", "constraints", "evidence", "metadata",
            )
            if key in raw
        })
    clean_observations = []
    for raw in observations[-MAX_EXPERIMENTAL_ROUTE_OBSERVATIONS:]:
        if not isinstance(raw, dict):
            continue
        clean_observations.append({
            key: raw[key]
            for key in (
                "iteration", "parameters", "kpi_values", "qc_passed", "is_failure",
                "failure_reason", "timestamp", "metadata",
            )
            if key in raw
        })
    clean_graph: dict[str, Any] = {
        "graph_id": graph.get("graph_id", "experimental-routes"),
        "nodes": nodes,
        "transitions": transitions,
        "metadata": dict(graph.get("metadata") or {}),
    }
    if graph.get("active_node_id"):
        clean_graph["active_node_id"] = graph["active_node_id"]
    return {
        "campaign_id": campaign_id,
        "graph": clean_graph,
        "observations": clean_observations,
        "objectives": [objective],
        "objective_directions": [direction],
        "available_capabilities": available_capabilities,
    }


def _decode_http_error(exc: HTTPError) -> dict[str, Any]:
    try:
        decoded = json.loads(
            exc.read(MAX_NEXUS_RESPONSE_BYTES + 1)[
                :MAX_NEXUS_RESPONSE_BYTES
            ].decode("utf-8")
        )
        return decoded if isinstance(decoded, dict) else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _error_message(raw: dict[str, Any], fallback: Any) -> str:
    detail = raw.get("detail")
    if isinstance(detail, str):
        return detail
    return str(fallback)


def _text(value: Any) -> str | None:
    return str(value) if value is not None and str(value) else None
