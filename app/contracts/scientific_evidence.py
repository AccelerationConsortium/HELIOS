"""Typed advisory scientific-evidence contract consumed by HELIOS.

The contract carries literature and validated experimental evidence into the
campaign decision layer. It deliberately excludes executable protocols,
commands, and workflow mappings: external evidence may inform a HELIOS
decision, but it never owns execution.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCIENTIFIC_EVIDENCE_CONTRACT_VERSION = "scientific_evidence_bundle.v1"

MAX_CLAIMS = 32
MAX_PATHS = 64
MAX_CONFLICTS = 32
MAX_SOURCE_REFS_PER_CLAIM = 8
MAX_CHUNKS_PER_SOURCE = 16
MAX_METADATA_ITEMS = 32
MAX_METADATA_DEPTH = 3

_EXECUTION_KEYS = {
    "command",
    "commands",
    "execution_graph",
    "execution_mapping",
    "hardware_command",
    "primitive",
    "protocol",
    "protocol_template",
    "steps",
    "workflow",
}
_EXECUTION_KEYS_COMPACT = {
    "".join(character for character in key if character.isalnum())
    for key in _EXECUTION_KEYS
}
_SENSITIVE_KEYS_COMPACT = {
    "accesstoken",
    "apikey",
    "authtoken",
    "authorization",
    "bearertoken",
    "cookie",
    "credential",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "setcookie",
    "token",
}


class EvidenceNamespace(StrEnum):
    PUBLISHED = "published_evidence"
    LOCAL_EXPERIMENTAL = "local_experimental_evidence"
    HYPOTHESIS = "hypothesis"
    DERIVED_INFERENCE = "derived_inference"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"


class ScientificSourceType(StrEnum):
    PAPER = "paper"
    EXPERIMENT = "experiment"
    DATASET = "dataset"
    OTHER = "other"


class EvidencePolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class EvidenceCentrality(StrEnum):
    CORE_CONTRIBUTION = "core_contribution"
    SUPPORTING_METHOD = "supporting_method"
    BACKGROUND_ONLY = "background_only"
    INCIDENTAL_MENTION = "incidental_mention"
    UNRELATED = "unrelated"


class ApplicabilityStatus(StrEnum):
    APPLICABLE = "applicable"
    PARTIAL = "partial"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class ScientificEvidenceStatus(StrEnum):
    USABLE = "usable"
    INSUFFICIENT = "insufficient_evidence"
    CONFLICTING = "conflicting_evidence"
    APPLICABILITY_MISMATCH = "applicability_mismatch"
    STALE = "stale"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


class ScientificEvidencePolicyMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"
    BOUNDED = "bounded"


class ScientificEvidenceRecommendedAction(StrEnum):
    NONE = "none"
    QUERY_LITERATURE = "query_literature"
    RUN_VALIDATION = "run_validation"
    REQUEST_HUMAN_OBSERVATION = "request_human_observation"


class ScientificSourceRef(BaseModel):
    """Traceable source for one scientific claim."""

    model_config = ConfigDict(extra="forbid")

    ref_id: str = Field(min_length=1, max_length=160)
    source_type: ScientificSourceType = ScientificSourceType.PAPER
    source_id: str = Field(min_length=1, max_length=160)
    paper_id: str | None = Field(default=None, min_length=1, max_length=160)
    experiment_id: str | None = Field(default=None, min_length=1, max_length=160)
    chunk_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_CHUNKS_PER_SOURCE,
    )
    title: str | None = Field(default=None, max_length=512)
    doi: str | None = Field(default=None, max_length=256)
    pages: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("chunk_ids", "pages")
    @classmethod
    def _unique_non_empty_values(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("source reference values must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("source reference values must be unique")
        return normalized

    @model_validator(mode="after")
    def _type_specific_identifier(self) -> ScientificSourceRef:
        if self.source_type == ScientificSourceType.PAPER and not self.paper_id:
            raise ValueError("paper source references require paper_id")
        if (
            self.source_type == ScientificSourceType.EXPERIMENT
            and not self.experiment_id
        ):
            raise ValueError("experiment source references require experiment_id")
        return self


class ApplicabilityContext(BaseModel):
    """Conditions under which a claim can be reused for the current campaign."""

    model_config = ConfigDict(extra="forbid")

    status: ApplicabilityStatus = ApplicabilityStatus.UNKNOWN
    material_families: list[str] = Field(default_factory=list, max_length=16)
    methods: list[str] = Field(default_factory=list, max_length=16)
    instruments: list[str] = Field(default_factory=list, max_length=16)
    protocols: list[str] = Field(default_factory=list, max_length=16)
    conditions: dict[str, str | int | float | bool] = Field(default_factory=dict)
    mismatches: list[str] = Field(default_factory=list, max_length=16)

    @field_validator(
        "material_families",
        "methods",
        "instruments",
        "protocols",
        "mismatches",
    )
    @classmethod
    def _bounded_strings(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 256 for value in normalized):
            raise ValueError(
                "applicability values must be non-empty and <=256 characters"
            )
        return normalized

    @field_validator("conditions")
    @classmethod
    def _bounded_conditions(
        cls,
        value: dict[str, str | int | float | bool],
    ) -> dict[str, str | int | float | bool]:
        if len(value) > 16:
            raise ValueError("applicability conditions exceed the 16-item limit")
        for key, item in value.items():
            if not key or len(key) > 128:
                raise ValueError(
                    "applicability condition keys must be 1-128 characters"
                )
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("applicability condition numbers must be finite")
            if isinstance(item, str) and len(item) > 512:
                raise ValueError(
                    "applicability condition strings must be <=512 characters"
                )
        return value


class ScientificClaim(BaseModel):
    """One source-grounded claim in a scientific evidence bundle."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=160)
    claim_type: str = Field(min_length=1, max_length=160)
    statement: str = Field(min_length=1, max_length=2000)
    namespace: EvidenceNamespace
    polarity: EvidencePolarity
    centrality: EvidenceCentrality
    confidence: float = Field(ge=0.0, le=1.0)
    applicability: ApplicabilityContext = Field(default_factory=ApplicabilityContext)
    source_refs: list[ScientificSourceRef] = Field(
        min_length=1,
        max_length=MAX_SOURCE_REFS_PER_CLAIM,
    )
    tags: list[str] = Field(default_factory=list, max_length=16)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def _bounded_tags(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 128 for value in normalized):
            raise ValueError("claim tags must be non-empty and <=128 characters")
        return normalized

    @field_validator("metadata")
    @classmethod
    def _safe_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_bounded_json(value)
        return value

    @model_validator(mode="after")
    def _source_refs_are_unique(self) -> ScientificClaim:
        ref_ids = [ref.ref_id for ref in self.source_refs]
        if len(set(ref_ids)) != len(ref_ids):
            raise ValueError(f"claim {self.claim_id} has duplicate source ref ids")
        if self.namespace == EvidenceNamespace.PUBLISHED and any(
            ref.source_type != ScientificSourceType.PAPER
            for ref in self.source_refs
        ):
            raise ValueError("published evidence claims require paper sources")
        if self.namespace == EvidenceNamespace.LOCAL_EXPERIMENTAL and not any(
            ref.source_type == ScientificSourceType.EXPERIMENT
            for ref in self.source_refs
        ):
            raise ValueError(
                "local experimental claims require an experiment source"
            )
        return self


class ScientificEvidencePath(BaseModel):
    """Auditable path connecting a claim to graph relations and source chunks."""

    model_config = ConfigDict(extra="forbid")

    path_id: str = Field(min_length=1, max_length=160)
    claim_id: str = Field(min_length=1, max_length=160)
    relation_types: list[str] = Field(min_length=1, max_length=16)
    source_ref_ids: list[str] = Field(min_length=1, max_length=16)
    summary: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("relation_types", "source_ref_ids")
    @classmethod
    def _unique_non_empty_values(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 160 for value in normalized):
            raise ValueError(
                "evidence path values must be non-empty and <=160 characters"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("evidence path values must be unique")
        return normalized


class EvidenceConflict(BaseModel):
    """Explicitly preserved disagreement between source-grounded claims."""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str = Field(min_length=1, max_length=160)
    claim_ids: list[str] = Field(min_length=2, max_length=8)
    reason: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("claim_ids")
    @classmethod
    def _unique_claim_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 160 for value in normalized):
            raise ValueError(
                "conflict claim ids must be non-empty and <=160 characters"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("conflict claim ids must be unique")
        return normalized


class ScientificEvidenceBundle(BaseModel):
    """Versioned, bounded, advisory-only evidence package."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["scientific_evidence_bundle.v1"]
    authority: Literal["advisory_only"] = "advisory_only"
    bundle_id: str = Field(min_length=1, max_length=160)
    query_id: str | None = Field(default=None, max_length=160)
    claims: list[ScientificClaim] = Field(default_factory=list, max_length=MAX_CLAIMS)
    evidence_paths: list[ScientificEvidencePath] = Field(
        default_factory=list,
        max_length=MAX_PATHS,
    )
    conflicts: list[EvidenceConflict] = Field(
        default_factory=list,
        max_length=MAX_CONFLICTS,
    )
    corpus_version: str = Field(min_length=1, max_length=160)
    ontology_version: str = Field(min_length=1, max_length=160)
    created_at: datetime
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at", "expires_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.tzinfo.utcoffset(value) is None
        ):
            raise ValueError("scientific evidence timestamps must be timezone-aware")
        return value

    @field_validator("metadata")
    @classmethod
    def _safe_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_bounded_json(value)
        return value

    @model_validator(mode="after")
    def _validate_graph_references(self) -> ScientificEvidenceBundle:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")

        claim_ids = [claim.claim_id for claim in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim ids must be unique")
        path_ids = [path.path_id for path in self.evidence_paths]
        if len(set(path_ids)) != len(path_ids):
            raise ValueError("evidence path ids must be unique")
        conflict_ids = [conflict.conflict_id for conflict in self.conflicts]
        if len(set(conflict_ids)) != len(conflict_ids):
            raise ValueError("conflict ids must be unique")

        claims_by_id = {claim.claim_id: claim for claim in self.claims}
        for path in self.evidence_paths:
            claim = claims_by_id.get(path.claim_id)
            if claim is None:
                raise ValueError(f"path {path.path_id} references unknown claim")
            available_refs = {ref.ref_id for ref in claim.source_refs}
            if not set(path.source_ref_ids).issubset(available_refs):
                raise ValueError(
                    f"path {path.path_id} references sources outside claim {path.claim_id}"
                )
        for conflict in self.conflicts:
            if not set(conflict.claim_ids).issubset(claims_by_id):
                raise ValueError(
                    f"conflict {conflict.conflict_id} references unknown claims"
                )
        if not self.claims and (self.evidence_paths or self.conflicts):
            raise ValueError("paths and conflicts require at least one claim")
        return self


class ScientificEvidenceAssessment(BaseModel):
    """Deterministic HELIOS assessment of one external evidence bundle."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str | None = None
    status: ScientificEvidenceStatus
    policy_mode: ScientificEvidencePolicyMode = ScientificEvidencePolicyMode.OFF
    support_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    applicability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_count: int = Field(default=0, ge=0, le=MAX_CLAIMS)
    evidence_path_count: int = Field(default=0, ge=0, le=MAX_PATHS)
    conflict_count: int = Field(default=0, ge=0, le=MAX_CONFLICTS)
    stale: bool = False
    requires_human_review: bool = False
    recommended_action: ScientificEvidenceRecommendedAction = (
        ScientificEvidenceRecommendedAction.NONE
    )
    reasons: list[str] = Field(default_factory=list, max_length=16)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _safe_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_bounded_json(value)
        return value

    @field_validator("reasons")
    @classmethod
    def _bounded_reasons(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 2000 for value in normalized):
            raise ValueError(
                "assessment reasons must be non-empty and <=2000 characters"
            )
        return normalized


def _validate_bounded_json(
    value: Any,
    *,
    depth: int = 0,
) -> None:
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(f"metadata exceeds depth limit {MAX_METADATA_DEPTH}")
    if isinstance(value, dict):
        if len(value) > MAX_METADATA_ITEMS:
            raise ValueError(f"metadata exceeds {MAX_METADATA_ITEMS} items")
        for key, item in value.items():
            normalized = str(key).strip().lower()
            compact = "".join(
                character for character in normalized if character.isalnum()
            )
            if not normalized or len(normalized) > 128:
                raise ValueError("metadata keys must be 1-128 characters")
            if (
                normalized in _EXECUTION_KEYS
                or compact in _EXECUTION_KEYS_COMPACT
            ):
                raise ValueError(
                    f"external scientific evidence cannot include executable field {key!r}"
                )
            if compact in _SENSITIVE_KEYS_COMPACT:
                raise ValueError(
                    f"external scientific evidence cannot include sensitive field {key!r}"
                )
            _validate_bounded_json(item, depth=depth + 1)
        return
    if isinstance(value, list | tuple):
        if len(value) > MAX_METADATA_ITEMS:
            raise ValueError(f"metadata list exceeds {MAX_METADATA_ITEMS} items")
        for item in value:
            _validate_bounded_json(item, depth=depth + 1)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metadata numbers must be finite")
    if isinstance(value, str) and len(value) > 2000:
        raise ValueError("metadata strings must be <=2000 characters")
    if value is not None and not isinstance(value, str | int | float | bool):
        raise ValueError(f"unsupported metadata type {type(value).__name__}")
