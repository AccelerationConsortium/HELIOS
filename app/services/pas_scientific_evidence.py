"""Fail-closed Paper Attribution System scientific-evidence adapter.

PAS is an external evidence provider. This module validates its versioned
payload, derives deterministic HELIOS assessment signals, and exposes bounded
decision evidence. It never creates executable routes or hardware commands.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import ValidationError

from app.contracts.scientific_evidence import (
    SCIENTIFIC_EVIDENCE_CONTRACT_VERSION,
    ApplicabilityStatus,
    EvidenceCentrality,
    EvidenceNamespace,
    EvidencePolarity,
    ScientificEvidenceAssessment,
    ScientificEvidenceBundle,
    ScientificEvidencePolicyMode,
    ScientificEvidenceRecommendedAction,
    ScientificEvidenceStatus,
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_BUNDLE_BYTES = 262_144

_CENTRALITY_WEIGHT = {
    EvidenceCentrality.CORE_CONTRIBUTION: 1.0,
    EvidenceCentrality.SUPPORTING_METHOD: 0.75,
    EvidenceCentrality.BACKGROUND_ONLY: 0.25,
    EvidenceCentrality.INCIDENTAL_MENTION: 0.1,
    EvidenceCentrality.UNRELATED: 0.0,
}
_APPLICABILITY_SCORE = {
    ApplicabilityStatus.APPLICABLE: 1.0,
    ApplicabilityStatus.PARTIAL: 0.5,
    ApplicabilityStatus.MISMATCH: 0.0,
    ApplicabilityStatus.UNKNOWN: 0.25,
}


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_PAS_OPENER = build_opener(_NoRedirectHandler())


def _open_pas_request(request: Request, *, timeout: float):
    return _PAS_OPENER.open(request, timeout=timeout)


class PasScientificEvidenceErrorType(StrEnum):
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    UNSUPPORTED_CONTRACT_VERSION = "unsupported_contract_version"
    OVERSIZED_PAYLOAD = "oversized_payload"


@dataclass(frozen=True)
class PasScientificEvidenceResponse:
    """Typed response from the PAS evidence endpoint."""

    ok: bool
    endpoint: str
    status_code: int | None = None
    bundle: ScientificEvidenceBundle | None = None
    error_type: PasScientificEvidenceErrorType | None = None
    error_message: str = ""


@dataclass(frozen=True)
class PasScientificEvidenceAdvice:
    """HELIOS-native advisory projection of one PAS response."""

    bundle: ScientificEvidenceBundle | None
    assessment: ScientificEvidenceAssessment
    requires_operator_approval: bool = False
    audit_metadata: dict[str, Any] = field(default_factory=dict)


class PasScientificEvidenceClient:
    """Small synchronous REST client used only behind explicit config gates."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        api_key: str | None = None,
        max_bundle_bytes: int | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.pas_evidence_url).rstrip("/")
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else settings.pas_evidence_timeout_seconds
        )
        self.api_key = api_key if api_key is not None else settings.pas_evidence_api_key
        self.max_bundle_bytes = (
            int(max_bundle_bytes)
            if max_bundle_bytes is not None
            else settings.pas_evidence_max_bundle_bytes
        )
        parsed_url = urlsplit(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("PAS evidence URL must be an absolute HTTP(S) URL")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("PAS evidence URL cannot contain user information")
        loopback_hosts = {"localhost", "127.0.0.1", "::1"}
        if (
            parsed_url.scheme != "https"
            and parsed_url.hostname not in loopback_hosts
        ):
            raise ValueError(
                "PAS evidence URL must use HTTPS except for loopback hosts"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("PAS evidence timeout must be positive")
        if self.max_bundle_bytes < 1024:
            raise ValueError("PAS evidence byte limit must be at least 1024")
        if "\r" in self.api_key or "\n" in self.api_key:
            raise ValueError("PAS evidence API key cannot contain header newlines")

    def query(self, payload: dict[str, Any]) -> PasScientificEvidenceResponse:
        endpoint = f"{self.base_url}/scientific-evidence/query"
        try:
            body = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            return self._failure(
                endpoint,
                PasScientificEvidenceErrorType.BAD_REQUEST,
                f"PAS query payload is not valid bounded JSON: {exc}",
            )
        if len(body) > self.max_bundle_bytes:
            return self._failure(
                endpoint,
                PasScientificEvidenceErrorType.OVERSIZED_PAYLOAD,
                "PAS query payload exceeds configured byte limit.",
            )
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = Request(endpoint, data=body, method="POST", headers=headers)
        try:
            with _open_pas_request(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_body = response.read(self.max_bundle_bytes + 1)
                if len(raw_body) > self.max_bundle_bytes:
                    return self._failure(
                        endpoint,
                        PasScientificEvidenceErrorType.OVERSIZED_PAYLOAD,
                        "PAS response exceeds configured byte limit.",
                        status_code=getattr(response, "status", None),
                    )
                decoded = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                return self._build_response(
                    endpoint=endpoint,
                    status_code=getattr(response, "status", 200),
                    decoded=decoded,
                )
        except HTTPError as exc:
            error_type = (
                PasScientificEvidenceErrorType.BAD_REQUEST
                if exc.code == 400
                else PasScientificEvidenceErrorType.NOT_FOUND
                if exc.code == 404
                else PasScientificEvidenceErrorType.UNAVAILABLE
            )
            return self._failure(
                endpoint,
                error_type,
                f"PAS returned HTTP {exc.code}.",
                status_code=exc.code,
            )
        except TimeoutError:
            return self._failure(
                endpoint,
                PasScientificEvidenceErrorType.TIMEOUT,
                f"PAS evidence request timed out after {self.timeout_seconds}s.",
            )
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            error_type = (
                PasScientificEvidenceErrorType.TIMEOUT
                if isinstance(reason, TimeoutError)
                else PasScientificEvidenceErrorType.UNAVAILABLE
            )
            return self._failure(
                endpoint,
                error_type,
                "PAS evidence endpoint timed out."
                if error_type == PasScientificEvidenceErrorType.TIMEOUT
                else "PAS evidence endpoint is unavailable.",
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return self._failure(
                endpoint,
                PasScientificEvidenceErrorType.INVALID_RESPONSE,
                f"PAS returned an invalid response ({type(exc).__name__}).",
            )

    def _build_response(
        self,
        *,
        endpoint: str,
        status_code: int,
        decoded: Any,
    ) -> PasScientificEvidenceResponse:
        if not isinstance(decoded, dict):
            return self._failure(
                endpoint,
                PasScientificEvidenceErrorType.INVALID_RESPONSE,
                "PAS response must be a JSON object.",
                status_code=status_code,
            )
        candidate = decoded.get(
            "bundle", decoded.get("scientific_evidence_bundle", decoded)
        )
        if not isinstance(candidate, dict):
            return self._failure(
                endpoint,
                PasScientificEvidenceErrorType.INVALID_RESPONSE,
                "PAS response does not contain a scientific evidence bundle.",
                status_code=status_code,
            )
        contract_version = candidate.get("contract_version")
        if contract_version != SCIENTIFIC_EVIDENCE_CONTRACT_VERSION:
            return self._failure(
                endpoint,
                PasScientificEvidenceErrorType.UNSUPPORTED_CONTRACT_VERSION,
                f"Unsupported PAS evidence contract: {contract_version!r}.",
                status_code=status_code,
            )
        try:
            bundle = ScientificEvidenceBundle.model_validate(candidate)
        except ValidationError as exc:
            return self._failure(
                endpoint,
                PasScientificEvidenceErrorType.INVALID_RESPONSE,
                _validation_error_message(exc),
                status_code=status_code,
            )
        response = PasScientificEvidenceResponse(
            ok=True,
            endpoint=endpoint,
            status_code=status_code,
            bundle=bundle,
        )
        self._log_response(response)
        return response

    def _failure(
        self,
        endpoint: str,
        error_type: PasScientificEvidenceErrorType,
        error_message: str,
        *,
        status_code: int | None = None,
    ) -> PasScientificEvidenceResponse:
        response = PasScientificEvidenceResponse(
            ok=False,
            endpoint=endpoint,
            status_code=status_code,
            error_type=error_type,
            error_message=error_message[:2000],
        )
        self._log_response(response)
        return response

    @staticmethod
    def _log_response(response: PasScientificEvidenceResponse) -> None:
        logger.info(
            "PAS scientific evidence response: bundle=%s contract=%s ok=%s error=%s",
            response.bundle.bundle_id if response.bundle else None,
            response.bundle.contract_version if response.bundle else None,
            response.ok,
            response.error_type,
        )


class PasScientificEvidenceAdapter:
    """Validate PAS input and derive bounded HELIOS decision evidence."""

    def adapt(
        self,
        value: (
            dict[str, Any]
            | ScientificEvidenceBundle
            | PasScientificEvidenceResponse
            | None
        ),
        *,
        policy_mode: ScientificEvidencePolicyMode = ScientificEvidencePolicyMode.OFF,
        now: datetime | None = None,
    ) -> PasScientificEvidenceAdvice:
        response = value if isinstance(value, PasScientificEvidenceResponse) else None
        if response is not None and (not response.ok or response.bundle is None):
            assessment = unavailable_evidence_assessment(
                policy_mode=policy_mode,
                reason=response.error_message or "PAS evidence unavailable.",
                error_type=response.error_type,
            )
            return PasScientificEvidenceAdvice(
                bundle=None,
                assessment=assessment,
                audit_metadata={
                    "endpoint": response.endpoint,
                    "status_code": response.status_code,
                    "error_type": response.error_type,
                    "error_message": response.error_message,
                },
            )

        try:
            bundle = (
                response.bundle
                if response is not None
                else value
                if isinstance(value, ScientificEvidenceBundle)
                else ScientificEvidenceBundle.model_validate(
                    value.get("bundle", value) if isinstance(value, dict) else value
                )
            )
        except (ValidationError, TypeError, AttributeError) as exc:
            error_message = (
                _validation_error_message(exc)
                if isinstance(exc, ValidationError)
                else f"Invalid PAS evidence bundle ({type(exc).__name__})."
            )
            assessment = unavailable_evidence_assessment(
                policy_mode=policy_mode,
                reason=error_message,
                status=ScientificEvidenceStatus.INVALID,
                error_type=PasScientificEvidenceErrorType.INVALID_RESPONSE,
            )
            return PasScientificEvidenceAdvice(
                bundle=None,
                assessment=assessment,
                audit_metadata={
                    "error_type": PasScientificEvidenceErrorType.INVALID_RESPONSE,
                    "error_message": error_message,
                },
            )

        assessment = assess_scientific_evidence(
            bundle,
            policy_mode=policy_mode,
            now=now,
        )
        return PasScientificEvidenceAdvice(
            bundle=bundle,
            assessment=assessment,
            requires_operator_approval=assessment.requires_human_review,
            audit_metadata={
                "bundle_id": bundle.bundle_id,
                "contract_version": bundle.contract_version,
                "corpus_version": bundle.corpus_version,
                "ontology_version": bundle.ontology_version,
                "policy_mode": policy_mode.value,
            },
        )


def assess_scientific_evidence(
    bundle: ScientificEvidenceBundle,
    *,
    policy_mode: ScientificEvidencePolicyMode = ScientificEvidencePolicyMode.OFF,
    now: datetime | None = None,
) -> ScientificEvidenceAssessment:
    """Derive deterministic evidence strength, applicability, and next action."""
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
        raise ValueError("assessment time must be timezone-aware")

    stale = bundle.expires_at is not None and bundle.expires_at <= timestamp
    claim_count = len(bundle.claims)
    source_grounded = sum(
        1
        for claim in bundle.claims
        if claim.source_refs
        and all(source.source_id and source.chunk_ids for source in claim.source_refs)
    )
    source_coverage = source_grounded / claim_count if claim_count else 0.0
    applicability_score = (
        _mean(
            [
                _APPLICABILITY_SCORE[claim.applicability.status]
                for claim in bundle.claims
            ]
        )
        if claim_count
        else 0.0
    )
    support_strength = max(
        (
            claim.confidence * _CENTRALITY_WEIGHT[claim.centrality]
            for claim in bundle.claims
            if claim.polarity == EvidencePolarity.POSITIVE
            and claim.namespace != EvidenceNamespace.REFUTED
        ),
        default=0.0,
    )
    negative_claim_strength = max(
        (
            claim.confidence * _CENTRALITY_WEIGHT[claim.centrality]
            for claim in bundle.claims
            if claim.polarity == EvidencePolarity.NEGATIVE
            or claim.namespace == EvidenceNamespace.REFUTED
        ),
        default=0.0,
    )
    conflict_strength = max(
        (conflict.confidence for conflict in bundle.conflicts),
        default=0.0,
    )
    contradiction_strength = max(negative_claim_strength, conflict_strength)
    has_mismatch = any(
        claim.applicability.status == ApplicabilityStatus.MISMATCH
        for claim in bundle.claims
    )
    has_unknown_applicability = any(
        claim.applicability.status == ApplicabilityStatus.UNKNOWN
        for claim in bundle.claims
    )

    reasons: list[str] = []
    requires_human_review = False
    if stale:
        status = ScientificEvidenceStatus.STALE
        action = ScientificEvidenceRecommendedAction.QUERY_LITERATURE
        reasons.append("Evidence bundle is expired.")
    elif not bundle.claims:
        status = ScientificEvidenceStatus.INSUFFICIENT
        action = ScientificEvidenceRecommendedAction.QUERY_LITERATURE
        reasons.append("Evidence bundle contains no claims.")
    elif bundle.conflicts or (support_strength > 0.0 and contradiction_strength >= 0.5):
        status = ScientificEvidenceStatus.CONFLICTING
        action = ScientificEvidenceRecommendedAction.RUN_VALIDATION
        requires_human_review = True
        reasons.append("Source-grounded claims contain material conflict.")
    elif has_mismatch:
        status = ScientificEvidenceStatus.APPLICABILITY_MISMATCH
        action = ScientificEvidenceRecommendedAction.REQUEST_HUMAN_OBSERVATION
        requires_human_review = True
        reasons.append("At least one claim is not applicable to the current context.")
    else:
        status = ScientificEvidenceStatus.USABLE
        if has_unknown_applicability:
            action = (
                ScientificEvidenceRecommendedAction.REQUEST_HUMAN_OBSERVATION
            )
            reasons.append("Some applicability conditions remain unknown.")
            requires_human_review = True
        else:
            action = ScientificEvidenceRecommendedAction.NONE

    return ScientificEvidenceAssessment(
        bundle_id=bundle.bundle_id,
        status=status,
        policy_mode=policy_mode,
        support_strength=round(support_strength, 10),
        contradiction_strength=round(contradiction_strength, 10),
        applicability_score=round(applicability_score, 10),
        source_coverage=round(source_coverage, 10),
        claim_count=claim_count,
        evidence_path_count=len(bundle.evidence_paths),
        conflict_count=len(bundle.conflicts),
        stale=stale,
        requires_human_review=requires_human_review,
        recommended_action=action,
        reasons=reasons,
        metadata={
            "contract_version": bundle.contract_version,
            "corpus_version": bundle.corpus_version,
            "ontology_version": bundle.ontology_version,
        },
    )


def unavailable_evidence_assessment(
    *,
    policy_mode: ScientificEvidencePolicyMode,
    reason: str,
    status: ScientificEvidenceStatus = ScientificEvidenceStatus.UNAVAILABLE,
    error_type: PasScientificEvidenceErrorType | None = None,
) -> ScientificEvidenceAssessment:
    """Create a non-influencing assessment for unavailable or invalid evidence."""
    return ScientificEvidenceAssessment(
        status=status,
        policy_mode=policy_mode,
        recommended_action=ScientificEvidenceRecommendedAction.NONE,
        reasons=[reason[:2000]],
        metadata={"error_type": error_type.value if error_type else None},
    )


def _validation_error_message(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    return json.dumps(errors, separators=(",", ":"))[:2000]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
