from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.contracts.scientific_evidence import ScientificEvidenceBundle
from tests.fixtures.scientific_evidence import scientific_evidence_bundle_payload


def test_valid_scientific_evidence_bundle_is_versioned_and_traceable():
    bundle = ScientificEvidenceBundle.model_validate(
        scientific_evidence_bundle_payload()
    )

    assert bundle.contract_version == "scientific_evidence_bundle.v1"
    assert bundle.authority == "advisory_only"
    assert bundle.claims[0].source_refs[0].source_id == "doi:10.1000/example"
    assert bundle.claims[0].source_refs[0].chunk_ids == ["chunk-10", "chunk-11"]
    assert bundle.evidence_paths[0].claim_id == bundle.claims[0].claim_id
    created_at = bundle.model_dump(mode="json")["created_at"]
    assert created_at.endswith("Z") or created_at.endswith("+00:00")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", "scientific_evidence_bundle.v2"),
        ("authority", "execution_authority"),
    ],
)
def test_contract_rejects_unsupported_version_or_external_authority(field, value):
    payload = scientific_evidence_bundle_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        ScientificEvidenceBundle.model_validate(payload)


@pytest.mark.parametrize(
    "metadata",
    [
        {"protocol": "execute-this"},
        {"nested": {"hardware_command": "move-axis"}},
        {"nested": {"executionGraph": {"node": "unsafe"}}},
        {"nested": {"protocol-template": "unsafe"}},
        {"nested": {"steps": [{"name": "dispense"}]}},
    ],
)
def test_contract_rejects_executable_metadata(metadata):
    payload = scientific_evidence_bundle_payload()
    payload["metadata"] = metadata

    with pytest.raises(ValidationError, match="cannot include executable field"):
        ScientificEvidenceBundle.model_validate(payload)


def test_contract_rejects_unknown_claim_and_source_references():
    unknown_claim = scientific_evidence_bundle_payload()
    unknown_claim["evidence_paths"][0]["claim_id"] = "missing-claim"
    with pytest.raises(ValidationError, match="references unknown claim"):
        ScientificEvidenceBundle.model_validate(unknown_claim)

    unknown_source = scientific_evidence_bundle_payload()
    unknown_source["evidence_paths"][0]["source_ref_ids"] = ["missing-source"]
    with pytest.raises(ValidationError, match="references sources outside claim"):
        ScientificEvidenceBundle.model_validate(unknown_source)


def test_contract_rejects_conflicts_that_reference_unknown_claims():
    payload = scientific_evidence_bundle_payload()
    second_claim = deepcopy(payload["claims"][0])
    second_claim["claim_id"] = "claim-2"
    second_claim["source_refs"][0]["ref_id"] = "source-2"
    payload["claims"].append(second_claim)
    payload["conflicts"] = [
        {
            "conflict_id": "conflict-1",
            "claim_ids": ["claim-1", "missing-claim"],
            "reason": "Sources disagree.",
            "confidence": 0.9,
        }
    ]

    with pytest.raises(ValidationError, match="references unknown claims"):
        ScientificEvidenceBundle.model_validate(payload)


def test_contract_supports_typed_local_experimental_sources():
    payload = scientific_evidence_bundle_payload()
    payload["claims"][0]["namespace"] = "local_experimental_evidence"
    payload["claims"][0]["source_refs"] = [
        {
            "ref_id": "experiment-source-1",
            "source_type": "experiment",
            "source_id": "helios:campaign-1:round-4",
            "experiment_id": "campaign-1-round-4",
            "chunk_ids": ["result-packet-4"],
        }
    ]
    payload["evidence_paths"][0]["source_ref_ids"] = ["experiment-source-1"]

    bundle = ScientificEvidenceBundle.model_validate(payload)

    assert bundle.claims[0].source_refs[0].paper_id is None
    assert bundle.claims[0].source_refs[0].experiment_id == "campaign-1-round-4"


def test_contract_requires_source_type_specific_identifier():
    payload = scientific_evidence_bundle_payload()
    payload["claims"][0]["source_refs"][0].pop("paper_id")

    with pytest.raises(ValidationError, match="require paper_id"):
        ScientificEvidenceBundle.model_validate(payload)


def test_contract_rejects_duplicate_path_or_conflict_references():
    duplicate_path = scientific_evidence_bundle_payload()
    duplicate_path["evidence_paths"][0]["source_ref_ids"] = [
        "source-1",
        "source-1",
    ]
    with pytest.raises(ValidationError, match="must be unique"):
        ScientificEvidenceBundle.model_validate(duplicate_path)

    duplicate_conflict = scientific_evidence_bundle_payload()
    duplicate_conflict["conflicts"] = [
        {
            "conflict_id": "conflict-1",
            "claim_ids": ["claim-1", "claim-1"],
            "reason": "Invalid self-conflict.",
            "confidence": 0.9,
        }
    ]
    with pytest.raises(ValidationError, match="must be unique"):
        ScientificEvidenceBundle.model_validate(duplicate_conflict)


def test_contract_requires_timezone_aware_and_ordered_timestamps():
    naive = scientific_evidence_bundle_payload()
    naive["created_at"] = "2026-07-28T12:00:00"
    with pytest.raises(ValidationError, match="timezone-aware"):
        ScientificEvidenceBundle.model_validate(naive)

    reversed_window = scientific_evidence_bundle_payload()
    reversed_window["expires_at"] = "2026-07-27T00:00:00Z"
    with pytest.raises(ValidationError, match="later than created_at"):
        ScientificEvidenceBundle.model_validate(reversed_window)


def test_contract_rejects_unbounded_or_nonfinite_metadata():
    oversized = scientific_evidence_bundle_payload()
    oversized["metadata"] = {f"key-{index}": index for index in range(33)}
    with pytest.raises(ValidationError, match="exceeds 32 items"):
        ScientificEvidenceBundle.model_validate(oversized)

    nonfinite = scientific_evidence_bundle_payload()
    nonfinite["metadata"] = {"score": float("nan")}
    with pytest.raises(ValidationError, match="must be finite"):
        ScientificEvidenceBundle.model_validate(nonfinite)


@pytest.mark.parametrize(
    "key",
    ["api_key", "Authorization", "private-key", "accessToken"],
)
def test_contract_rejects_sensitive_metadata_keys(key):
    payload = scientific_evidence_bundle_payload()
    payload["metadata"] = {key: "must-not-persist"}

    with pytest.raises(ValidationError, match="sensitive field"):
        ScientificEvidenceBundle.model_validate(payload)
