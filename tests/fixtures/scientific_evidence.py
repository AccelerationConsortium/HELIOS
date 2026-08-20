from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def scientific_evidence_bundle_payload(
    *,
    claims: list[dict[str, Any]] | None = None,
    conflicts: list[dict[str, Any]] | None = None,
    expires_at: str | None = "2030-01-01T00:00:00Z",
) -> dict[str, Any]:
    default_claims = [
        {
            "claim_id": "claim-1",
            "claim_type": "process_outcome",
            "statement": "Lower flow rate improves film uniformity in the studied range.",
            "namespace": "published_evidence",
            "polarity": "positive",
            "centrality": "core_contribution",
            "confidence": 0.8,
            "applicability": {
                "status": "applicable",
                "material_families": ["perovskite"],
                "methods": ["spin_coating"],
                "conditions": {"temperature_c": 25},
            },
            "source_refs": [
                {
                    "ref_id": "source-1",
                    "source_type": "paper",
                    "source_id": "doi:10.1000/example",
                    "paper_id": "doi:10.1000/example",
                    "chunk_ids": ["chunk-10", "chunk-11"],
                    "title": "Example source",
                    "doi": "10.1000/example",
                    "pages": ["4-5"],
                }
            ],
            "tags": ["flow_rate", "uniformity"],
        }
    ]
    effective_claims = default_claims if claims is None else claims
    evidence_paths = (
        [
            {
                "path_id": "path-1",
                "claim_id": "claim-1",
                "relation_types": ["supports"],
                "source_ref_ids": ["source-1"],
                "summary": "The attributed source chunks directly support the claim.",
                "confidence": 0.78,
            }
        ]
        if effective_claims
        and effective_claims[0].get("claim_id") == "claim-1"
        and effective_claims[0].get("source_refs")
        else []
    )
    return {
        "contract_version": "scientific_evidence_bundle.v1",
        "authority": "advisory_only",
        "bundle_id": "pas-bundle-1",
        "query_id": "pas-query-1",
        "claims": effective_claims,
        "evidence_paths": evidence_paths,
        "conflicts": conflicts or [],
        "corpus_version": "pas-corpus-2026-07-28",
        "ontology_version": "pas-aet-v1",
        "created_at": datetime(2026, 7, 28, tzinfo=UTC).isoformat(),
        "expires_at": expires_at,
        "metadata": {"retrieval_mode": "knowledge_graph"},
    }
