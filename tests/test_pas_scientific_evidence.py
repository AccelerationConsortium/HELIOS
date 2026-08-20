from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from io import BytesIO
from urllib.error import HTTPError

import pytest

from app.contracts.scientific_evidence import (
    ScientificEvidenceBundle,
    ScientificEvidencePolicyMode,
    ScientificEvidenceRecommendedAction,
    ScientificEvidenceStatus,
)
from app.services.pas_scientific_evidence import (
    _PAS_OPENER,
    PasScientificEvidenceAdapter,
    PasScientificEvidenceClient,
    PasScientificEvidenceErrorType,
    _NoRedirectHandler,
    assess_scientific_evidence,
)
from tests.fixtures.scientific_evidence import scientific_evidence_bundle_payload


class _FakeHTTPResponse:
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self.payload
        return self.payload[:size]


def _bundle(**overrides) -> ScientificEvidenceBundle:
    payload = scientific_evidence_bundle_payload()
    payload.update(overrides)
    return ScientificEvidenceBundle.model_validate(payload)


def _conflicting_bundle() -> ScientificEvidenceBundle:
    payload = scientific_evidence_bundle_payload()
    second_claim = deepcopy(payload["claims"][0])
    second_claim["claim_id"] = "claim-2"
    second_claim["statement"] = "Lower flow rate does not improve uniformity."
    second_claim["polarity"] = "negative"
    second_claim["confidence"] = 0.9
    second_claim["source_refs"][0]["ref_id"] = "source-2"
    second_claim["source_refs"][0]["source_id"] = "doi:10.1000/refutation"
    second_claim["source_refs"][0]["paper_id"] = "doi:10.1000/refutation"
    second_claim["source_refs"][0]["chunk_ids"] = ["chunk-20"]
    payload["claims"].append(second_claim)
    payload["conflicts"] = [
        {
            "conflict_id": "conflict-1",
            "claim_ids": ["claim-1", "claim-2"],
            "reason": "The source-grounded conclusions disagree.",
            "confidence": 0.9,
        }
    ]
    return ScientificEvidenceBundle.model_validate(payload)


def test_client_sends_api_key_and_validates_success_response(monkeypatch):
    captured = {}
    response_body = json.dumps(
        {"bundle": scientific_evidence_bundle_payload()}
    ).encode()

    def fake_open(request, timeout):  # noqa: ANN001, ANN202
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHTTPResponse(response_body)

    monkeypatch.setattr(
        "app.services.pas_scientific_evidence._open_pas_request",
        fake_open,
    )
    response = PasScientificEvidenceClient(
        base_url="https://pas.test/api/v1",
        timeout_seconds=2.5,
        api_key="pas-secret",
        max_bundle_bytes=32_768,
    ).query({"objective": "maximize yield"})

    assert response.ok is True
    assert response.bundle is not None
    assert response.bundle.bundle_id == "pas-bundle-1"
    assert response.endpoint == "https://pas.test/api/v1/scientific-evidence/query"
    assert captured["timeout"] == 2.5
    assert captured["request"].get_header("X-api-key") == "pas-secret"


def test_client_returns_typed_failures_without_raising(monkeypatch):
    def timeout_open(_request, timeout):  # noqa: ARG001
        raise TimeoutError

    monkeypatch.setattr(
        "app.services.pas_scientific_evidence._open_pas_request",
        timeout_open,
    )
    client = PasScientificEvidenceClient(
        base_url="https://pas.test",
        max_bundle_bytes=1024,
    )
    timeout = client.query({"query": "evidence"})
    assert timeout.ok is False
    assert timeout.error_type == PasScientificEvidenceErrorType.TIMEOUT

    invalid_query = client.query({"score": float("nan")})
    assert invalid_query.ok is False
    assert invalid_query.error_type == PasScientificEvidenceErrorType.BAD_REQUEST


def test_client_does_not_persist_remote_error_body(monkeypatch):
    def error_open(_request, timeout):  # noqa: ARG001
        raise HTTPError(
            "https://pas.test/scientific-evidence/query",
            500,
            "server error",
            {},
            BytesIO(b'{"detail":"credential-that-must-not-persist"}'),
        )

    monkeypatch.setattr(
        "app.services.pas_scientific_evidence._open_pas_request",
        error_open,
    )
    response = PasScientificEvidenceClient(
        base_url="https://pas.test",
    ).query({"query": "evidence"})

    assert response.error_type == PasScientificEvidenceErrorType.UNAVAILABLE
    assert response.error_message == "PAS returned HTTP 500."
    assert "credential-that-must-not-persist" not in response.error_message


def test_client_rejects_unsupported_contract_and_oversized_response(monkeypatch):
    unsupported = scientific_evidence_bundle_payload()
    unsupported["contract_version"] = "scientific_evidence_bundle.v2"

    monkeypatch.setattr(
        "app.services.pas_scientific_evidence._open_pas_request",
        lambda *_args, **_kwargs: _FakeHTTPResponse(
            json.dumps({"bundle": unsupported}).encode()
        ),
    )
    client = PasScientificEvidenceClient(
        base_url="https://pas.test",
        max_bundle_bytes=4096,
    )
    response = client.query({"query": "evidence"})
    assert response.error_type == (
        PasScientificEvidenceErrorType.UNSUPPORTED_CONTRACT_VERSION
    )

    monkeypatch.setattr(
        "app.services.pas_scientific_evidence._open_pas_request",
        lambda *_args, **_kwargs: _FakeHTTPResponse(b"x" * 1025),
    )
    oversized = PasScientificEvidenceClient(
        base_url="https://pas.test",
        max_bundle_bytes=1024,
    ).query({"query": "evidence"})
    assert oversized.error_type == PasScientificEvidenceErrorType.OVERSIZED_PAYLOAD


def test_client_rejects_unsafe_or_unbounded_configuration():
    with pytest.raises(ValueError, match="absolute HTTP"):
        PasScientificEvidenceClient(base_url="file:///tmp/pas.json")
    with pytest.raises(ValueError, match="must use HTTPS"):
        PasScientificEvidenceClient(base_url="http://pas.example")
    with pytest.raises(ValueError, match="user information"):
        PasScientificEvidenceClient(base_url="https://user:pass@pas.test")
    with pytest.raises(ValueError, match="timeout must be positive"):
        PasScientificEvidenceClient(
            base_url="https://pas.test",
            timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="at least 1024"):
        PasScientificEvidenceClient(
            base_url="https://pas.test",
            max_bundle_bytes=100,
        )
    with pytest.raises(ValueError, match="header newlines"):
        PasScientificEvidenceClient(
            base_url="https://pas.test",
            api_key="unsafe\nheader",
        )


def test_pas_transport_disables_redirects():
    redirect_handler = next(
        handler
        for handler in _PAS_OPENER.handlers
        if isinstance(handler, _NoRedirectHandler)
    )

    assert redirect_handler.redirect_request(None, None, None) is None


def test_assessment_classifies_usable_conflicting_mismatch_stale_and_empty():
    now = datetime(2026, 8, 1, tzinfo=UTC)
    usable = assess_scientific_evidence(
        _bundle(),
        policy_mode=ScientificEvidencePolicyMode.SHADOW,
        now=now,
    )
    assert usable.status == ScientificEvidenceStatus.USABLE
    assert usable.recommended_action == ScientificEvidenceRecommendedAction.NONE
    assert usable.support_strength == 0.8
    assert usable.source_coverage == 1.0

    conflict = assess_scientific_evidence(
        _conflicting_bundle(),
        policy_mode=ScientificEvidencePolicyMode.BOUNDED,
        now=now,
    )
    assert conflict.status == ScientificEvidenceStatus.CONFLICTING
    assert conflict.recommended_action == (
        ScientificEvidenceRecommendedAction.RUN_VALIDATION
    )
    assert conflict.requires_human_review is True

    mismatch_payload = scientific_evidence_bundle_payload()
    mismatch_payload["claims"][0]["applicability"]["status"] = "mismatch"
    mismatch = assess_scientific_evidence(
        ScientificEvidenceBundle.model_validate(mismatch_payload),
        now=now,
    )
    assert mismatch.status == ScientificEvidenceStatus.APPLICABILITY_MISMATCH
    assert mismatch.recommended_action == (
        ScientificEvidenceRecommendedAction.REQUEST_HUMAN_OBSERVATION
    )

    unknown_payload = scientific_evidence_bundle_payload()
    unknown_payload["claims"][0]["applicability"]["status"] = "unknown"
    unknown = assess_scientific_evidence(
        ScientificEvidenceBundle.model_validate(unknown_payload),
        now=now,
    )
    assert unknown.status == ScientificEvidenceStatus.USABLE
    assert unknown.requires_human_review is True
    assert unknown.recommended_action == (
        ScientificEvidenceRecommendedAction.REQUEST_HUMAN_OBSERVATION
    )

    stale = assess_scientific_evidence(
        _bundle(expires_at="2026-07-29T00:00:00Z"),
        now=now,
    )
    assert stale.status == ScientificEvidenceStatus.STALE
    assert stale.recommended_action == (
        ScientificEvidenceRecommendedAction.QUERY_LITERATURE
    )

    empty = assess_scientific_evidence(
        _bundle(claims=[], evidence_paths=[]),
        now=now,
    )
    assert empty.status == ScientificEvidenceStatus.INSUFFICIENT


def test_adapter_returns_only_validated_domain_assessment():
    advice = PasScientificEvidenceAdapter().adapt(
        _bundle(),
        policy_mode=ScientificEvidencePolicyMode.BOUNDED,
        now=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert advice.bundle is not None
    assert advice.assessment.policy_mode == ScientificEvidencePolicyMode.BOUNDED
    assert advice.assessment.bundle_id == "pas-bundle-1"
    assert advice.audit_metadata["contract_version"] == (
        "scientific_evidence_bundle.v1"
    )
    assert "protocol" not in advice.audit_metadata
    assert "execution_graph" not in advice.audit_metadata


def test_adapter_rejects_invalid_input_without_raising():
    advice = PasScientificEvidenceAdapter().adapt(
        {"contract_version": "unexpected"},
        policy_mode=ScientificEvidencePolicyMode.SHADOW,
    )

    assert advice.bundle is None
    assert advice.assessment.status == ScientificEvidenceStatus.INVALID
    assert advice.assessment.recommended_action == (
        ScientificEvidenceRecommendedAction.NONE
    )


def test_pas_settings_default_off_and_independently_gated(monkeypatch, request):
    from app.core.config import get_settings

    request.addfinalizer(get_settings.cache_clear)
    for name in (
        "PAS_EVIDENCE_FETCH_ENABLED",
        "PAS_EVIDENCE_SHADOW_ENABLED",
        "PAS_EVIDENCE_INFLUENCE_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.pas_evidence_fetch_enabled is False
    assert settings.pas_evidence_shadow_enabled is False
    assert settings.pas_evidence_influence_enabled is False

    monkeypatch.setenv("PAS_EVIDENCE_FETCH_ENABLED", "true")
    monkeypatch.setenv("PAS_EVIDENCE_SHADOW_ENABLED", "true")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.pas_evidence_fetch_enabled is True
    assert settings.pas_evidence_shadow_enabled is True
    assert settings.pas_evidence_influence_enabled is False


def test_orchestrator_pas_collection_is_default_off(monkeypatch, request):
    from app.agents.orchestrator import _maybe_collect_pas_scientific_evidence
    from app.core.config import get_settings

    request.addfinalizer(get_settings.cache_clear)
    monkeypatch.delenv("PAS_EVIDENCE_FETCH_ENABLED", raising=False)
    monkeypatch.delenv("PAS_EVIDENCE_SHADOW_ENABLED", raising=False)
    monkeypatch.delenv("PAS_EVIDENCE_INFLUENCE_ENABLED", raising=False)
    get_settings.cache_clear()

    bundle, assessment, audit = _maybe_collect_pas_scientific_evidence(
        supplied_bundle=scientific_evidence_bundle_payload(),
        query_payload={"objective": "yield"},
    )

    assert bundle is None
    assert assessment is None
    assert audit == {"policy_mode": "off", "collected": False}


def test_orchestrator_pas_collection_fails_closed(monkeypatch, request):
    from app.agents.orchestrator import _maybe_collect_pas_scientific_evidence
    from app.core.config import get_settings

    request.addfinalizer(get_settings.cache_clear)
    monkeypatch.setenv("PAS_EVIDENCE_FETCH_ENABLED", "true")
    monkeypatch.setenv("PAS_EVIDENCE_SHADOW_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "app.agents.orchestrator.PasScientificEvidenceClient.query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    bundle, assessment, audit = _maybe_collect_pas_scientific_evidence(
        supplied_bundle=None,
        query_payload={"objective": "yield"},
    )

    assert bundle is None
    assert assessment is not None
    assert assessment.status == ScientificEvidenceStatus.UNAVAILABLE
    assert assessment.policy_mode == ScientificEvidencePolicyMode.SHADOW
    assert audit["collected"] is False


def test_pas_shadow_gate_alone_records_contextual_trace(monkeypatch, request):
    from app.agents.orchestrator import (
        _maybe_collect_pas_scientific_evidence,
        _maybe_record_contextual_shadow_decision,
    )
    from app.core.config import get_settings

    request.addfinalizer(get_settings.cache_clear)
    monkeypatch.setenv("PAS_EVIDENCE_FETCH_ENABLED", "false")
    monkeypatch.setenv("PAS_EVIDENCE_SHADOW_ENABLED", "true")
    monkeypatch.setenv("PAS_EVIDENCE_INFLUENCE_ENABLED", "false")
    monkeypatch.setenv("CONTEXTUAL_DECISION_SHADOW_ENABLED", "false")
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "false")
    monkeypatch.setenv("CAMPAIGN_DECISION_AUTHORITY_ENABLED", "false")
    monkeypatch.setenv("CLOSED_LOOP_DRIFT_MONITOR_ENABLED", "false")
    get_settings.cache_clear()

    bundle, assessment, _audit = _maybe_collect_pas_scientific_evidence(
        supplied_bundle=scientific_evidence_bundle_payload(),
        query_payload={"objective": "yield"},
    )
    trace = _maybe_record_contextual_shadow_decision(
        campaign_id="campaign-pas-shadow",
        round_index=1,
        strategy_selection_result={"backend": "bo_mcp"},
        scientific_evidence=bundle,
        scientific_evidence_assessment=assessment,
    )

    assert trace is not None
    assert trace.context.scientific_evidence.bundle_id == "pas-bundle-1"
    assert trace.decision_plan.metadata["scientific_evidence_policy_mode"] == (
        "shadow"
    )


def test_pas_bounded_influence_requests_validation_without_live_mutation(
    monkeypatch,
    request,
):
    from app.agents.orchestrator import (
        _maybe_collect_pas_scientific_evidence,
        _maybe_record_contextual_shadow_decision,
    )
    from app.core.config import get_settings

    request.addfinalizer(get_settings.cache_clear)
    monkeypatch.setenv("PAS_EVIDENCE_FETCH_ENABLED", "false")
    monkeypatch.setenv("PAS_EVIDENCE_SHADOW_ENABLED", "false")
    monkeypatch.setenv("PAS_EVIDENCE_INFLUENCE_ENABLED", "true")
    monkeypatch.setenv("CONTEXTUAL_DECISION_SHADOW_ENABLED", "false")
    monkeypatch.setenv("SCIENTIFIC_LEDGER_ENABLED", "false")
    monkeypatch.setenv("CAMPAIGN_DECISION_AUTHORITY_ENABLED", "false")
    monkeypatch.setenv("CLOSED_LOOP_DRIFT_MONITOR_ENABLED", "false")
    get_settings.cache_clear()

    bundle, assessment, _audit = _maybe_collect_pas_scientific_evidence(
        supplied_bundle=_conflicting_bundle(),
        query_payload={"objective": "yield"},
    )
    trace = _maybe_record_contextual_shadow_decision(
        campaign_id="campaign-pas-bounded",
        round_index=1,
        strategy_selection_result={"backend": "bo_mcp"},
        scientific_evidence=bundle,
        scientific_evidence_assessment=assessment,
    )

    assert trace is not None
    assert trace.decision_plan.action_type.value == "run_validation"
    assert trace.decision_plan.shadow_only is True
    assert trace.decision_plan.metadata["scientific_evidence_policy_mode"] == (
        "bounded"
    )
