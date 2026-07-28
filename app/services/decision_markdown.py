"""Deterministic Markdown projections for HELIOS scientific decisions.

The runtime DTOs and SQLite rows remain the transactional source of truth.
This module produces a human-readable, Git-friendly projection of the same
decision accounting bundle without calling databases, Git, or live services.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

import yaml

from app.services.decision_outcome import CampaignDecisionAccounting
from app.services.decision_trace import CampaignDecisionTrace

DECISION_CARD_SCHEMA_VERSION = "helios.decision-card/v1"
LEDGER_RENDERER_VERSION = "1"
REDACTED = "[REDACTED]"

_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "set_cookie",
    "token",
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")


@dataclass(frozen=True)
class LedgerProvenance:
    """Version identities attached to every rendered scientific artifact."""

    code_commit: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    nexus_contract_version: str | None = None
    rubric_version: str | None = None
    renderer_version: str = LEDGER_RENDERER_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedDecisionBundle:
    """A complete set of relative Markdown files for one decision lifecycle."""

    campaign_id: str
    trace_id: str
    round_index: int
    status: str
    source_sha256: str
    files: Mapping[str, str]


def redact_sensitive(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact credentials while preserving scientific structure."""
    if key is not None and _is_secret_key(key):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_sensitive(item_value, key=str(item_key))
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, Enum):
        return redact_sensitive(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return redact_sensitive(value.model_dump(mode="json"))
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


def render_pending_decision(
    trace: CampaignDecisionTrace,
    *,
    provenance: LedgerProvenance | None = None,
    campaign_metadata: Mapping[str, Any] | None = None,
) -> RenderedDecisionBundle:
    """Render a decision immediately, before its experimental outcome exists."""
    return _render_bundle(
        trace=trace,
        accounting=None,
        provenance=provenance or LedgerProvenance(),
        campaign_metadata=campaign_metadata or {},
        observations=(),
        failures=(),
        recovery_events=(),
    )


def render_completed_decision(
    accounting: CampaignDecisionAccounting,
    *,
    provenance: LedgerProvenance | None = None,
    campaign_metadata: Mapping[str, Any] | None = None,
    observations: Sequence[Any] | None = None,
    failures: Sequence[Any] | None = None,
    recovery_events: Sequence[Any] | None = None,
) -> RenderedDecisionBundle:
    """Render the completed trace, outcome, verifiers, reward, and recovery."""
    return _render_bundle(
        trace=accounting.trace,
        accounting=accounting,
        provenance=provenance or LedgerProvenance(),
        campaign_metadata=campaign_metadata or {},
        observations=observations or (),
        failures=failures or (),
        recovery_events=recovery_events or (),
    )


def _render_bundle(
    *,
    trace: CampaignDecisionTrace,
    accounting: CampaignDecisionAccounting | None,
    provenance: LedgerProvenance,
    campaign_metadata: Mapping[str, Any],
    observations: Sequence[Any],
    failures: Sequence[Any],
    recovery_events: Sequence[Any],
) -> RenderedDecisionBundle:
    status = "completed" if accounting is not None else "pending"
    source = {
        "trace": trace.model_dump(mode="json"),
        "outcome": accounting.outcome.model_dump(mode="json") if accounting else None,
        "reward": accounting.reward.model_dump(mode="json") if accounting else None,
        "observations": list(observations),
        "failures": list(failures),
        "recovery_events": list(recovery_events),
        "campaign_metadata": dict(campaign_metadata),
        "provenance": _provenance_dict(provenance),
    }
    redacted_source = redact_sensitive(source)
    source_hash = _stable_hash(redacted_source)
    prefix = f"rounds/{trace.round_index:03d}"
    metadata = _card_metadata(trace, status, source_hash, provenance)

    files = {
        f"{prefix}/objective.md": _render_objective(trace, campaign_metadata, metadata),
        f"{prefix}/observations.md": _render_observations(trace, observations, metadata),
        f"{prefix}/decision_{trace.round_index:03d}.md": _render_decision_card(
            trace,
            accounting,
            provenance,
            source_hash,
            failures,
            recovery_events,
        ),
        f"{prefix}/strategy.md": _render_strategy(trace, metadata),
        f"{prefix}/evidence.md": _render_evidence(trace, metadata),
        f"{prefix}/failure.md": _render_failures(failures, metadata),
        f"{prefix}/recovery.md": _render_recovery(recovery_events, metadata),
        f"{prefix}/summary.md": _render_round_summary(
            trace, accounting, failures, recovery_events, metadata
        ),
    }
    return RenderedDecisionBundle(
        campaign_id=trace.campaign_id,
        trace_id=trace.trace_id,
        round_index=trace.round_index,
        status=status,
        source_sha256=source_hash,
        files=files,
    )


def _render_decision_card(
    trace: CampaignDecisionTrace,
    accounting: CampaignDecisionAccounting | None,
    provenance: LedgerProvenance,
    source_hash: str,
    failures: Sequence[Any],
    recovery_events: Sequence[Any],
) -> str:
    plan = trace.decision_plan
    metadata = _card_metadata(
        trace,
        "completed" if accounting else "pending",
        source_hash,
        provenance,
    )
    metadata.update(
        {
            "confidence": plan.confidence,
            "reward": accounting.reward.reward if accounting else None,
            "selected_action": plan.action_type.value,
            "selected_backend": plan.candidate_generation_backend or "unspecified",
        }
    )
    strategy = redact_sensitive(plan.strategy_trace)
    actions = _candidate_actions(strategy)
    selected = plan.action_type.value
    if plan.candidate_generation_backend:
        selected = f"{selected} via {plan.candidate_generation_backend}"

    lines = [
        _front_matter(metadata),
        f"# Decision Card {trace.round_index:03d}",
        "",
        "## Question",
        "",
        "What should happen next in this scientific campaign?",
        "",
        "## Context",
        "",
        _mapping_markdown(redact_sensitive(trace.context.model_dump(mode="json"))),
        "",
        "## Evidence",
        "",
        _evidence_table(trace),
        "",
        "## Candidate Actions",
        "",
        _action_table(actions, plan),
        "",
        "## Chosen",
        "",
        f"**{_safe_inline(selected)}**",
        "",
        "## Decision Rationale",
        "",
        _safe_paragraph(plan.rationale),
        "",
        "## Confidence and Expected Gain",
        "",
        f"- Confidence: {_format_scalar(plan.confidence)}",
        f"- Expected gain: {_expected_gain(actions, plan)}",
        f"- Fallback: {_safe_inline(plan.fallback_action.value if plan.fallback_action else 'None')}",
        f"- Shadow only: {'yes' if plan.shadow_only else 'no'}",
        "",
        "## Outcome",
        "",
    ]
    if accounting is None:
        lines.extend(["**Pending** — the decision has been recorded before execution.", ""])
    else:
        lines.extend([_outcome_table(accounting), ""])
    lines.extend(["## Reward and Verification", ""])
    if accounting is None:
        lines.extend(["Pending outcome evaluation.", ""])
    else:
        lines.extend([_reward_section(accounting), ""])
    lines.extend(
        [
            "## Failure and Recovery",
            "",
            f"- Failure events: {len(failures)}",
            f"- Recovery events: {len(recovery_events)}",
            "",
            "## Reproducibility",
            "",
            _mapping_markdown(_provenance_dict(provenance)),
            "",
        ]
    )
    return _finish(lines)


def _render_objective(
    trace: CampaignDecisionTrace,
    campaign_metadata: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    objective: Any = trace.context.objective_summary
    if not objective:
        objective = campaign_metadata.get("objective", {})
    if not isinstance(objective, Mapping):
        objective = {"value": objective}
    return _finish(
        [
            _front_matter({**metadata, "artifact_type": "objective"}),
            f"# Objective — Round {trace.round_index}",
            "",
            _mapping_markdown(redact_sensitive(objective)),
            "",
        ]
    )


def _render_observations(
    trace: CampaignDecisionTrace,
    observations: Sequence[Any],
    metadata: Mapping[str, Any],
) -> str:
    combined = [*trace.context.human_observations, *list(observations)]
    lines = [
        _front_matter({**metadata, "artifact_type": "observations"}),
        f"# Observations — Round {trace.round_index}",
        "",
    ]
    if not combined:
        lines.extend(["No observations were recorded.", ""])
    else:
        for index, observation in enumerate(combined, 1):
            lines.extend([f"## Observation {index}", "", _value_markdown(redact_sensitive(observation)), ""])
    return _finish(lines)


def _render_strategy(trace: CampaignDecisionTrace, metadata: Mapping[str, Any]) -> str:
    plan = trace.decision_plan
    strategy = redact_sensitive(plan.strategy_trace)
    actions = _candidate_actions(strategy)
    lines = [
        _front_matter({**metadata, "artifact_type": "strategy"}),
        f"# Strategy — Round {trace.round_index}",
        "",
        "## Selected Route",
        "",
        f"- Campaign intent: {_safe_inline(plan.campaign_intent or 'Unspecified')}",
        f"- Optimization mode: {_safe_inline(plan.optimization_mode or 'Unspecified')}",
        f"- Backend: {_safe_inline(plan.candidate_generation_backend or 'Unspecified')}",
        f"- Confidence: {_format_scalar(plan.confidence)}",
        "",
        "## Ranked Actions",
        "",
        _action_table(actions, plan),
        "",
        "## Strategy Trace",
        "",
        _mapping_markdown(strategy),
        "",
    ]
    return _finish(lines)


def _render_evidence(trace: CampaignDecisionTrace, metadata: Mapping[str, Any]) -> str:
    return _finish(
        [
            _front_matter({**metadata, "artifact_type": "evidence"}),
            f"# Evidence — Round {trace.round_index}",
            "",
            _evidence_table(trace),
            "",
        ]
    )


def _render_failures(failures: Sequence[Any], metadata: Mapping[str, Any]) -> str:
    lines = [
        _front_matter({**metadata, "artifact_type": "failure"}),
        "# Failure Record",
        "",
    ]
    if not failures:
        lines.extend(["No failure events were recorded.", ""])
    else:
        for index, failure in enumerate(failures, 1):
            payload = redact_sensitive(failure)
            lines.extend(
                [
                    f"## Failure {index}",
                    "",
                    f"**Problem:** {_failure_label(payload)}",
                    "",
                    "### Evidence and Attribution",
                    "",
                    _value_markdown(payload),
                    "",
                ]
            )
    return _finish(lines)


def _render_recovery(recovery_events: Sequence[Any], metadata: Mapping[str, Any]) -> str:
    lines = [
        _front_matter({**metadata, "artifact_type": "recovery"}),
        "# Recovery Record",
        "",
    ]
    if not recovery_events:
        lines.extend(["No recovery action was recorded.", ""])
    else:
        for index, event in enumerate(recovery_events, 1):
            lines.extend(
                [
                    f"## Recovery {index}",
                    "",
                    _value_markdown(redact_sensitive(event)),
                    "",
                ]
            )
    return _finish(lines)


def _render_round_summary(
    trace: CampaignDecisionTrace,
    accounting: CampaignDecisionAccounting | None,
    failures: Sequence[Any],
    recovery_events: Sequence[Any],
    metadata: Mapping[str, Any],
) -> str:
    plan = trace.decision_plan
    lines = [
        _front_matter({**metadata, "artifact_type": "round_summary"}),
        f"# Round {trace.round_index} Summary",
        "",
        f"- Decision: {_safe_inline(plan.action_type.value)}",
        f"- Backend: {_safe_inline(plan.candidate_generation_backend or 'Unspecified')}",
        f"- Confidence: {_format_scalar(plan.confidence)}",
        f"- Failures: {len(failures)}",
        f"- Recoveries: {len(recovery_events)}",
    ]
    if accounting is None:
        lines.extend(["- Outcome: Pending", ""])
    else:
        outcome = accounting.outcome
        lines.extend(
            [
                f"- Outcome: {'Success' if outcome.execution_success else 'Failure' if outcome.execution_success is False else 'Unknown'}",
                f"- Objective delta: {_format_scalar(outcome.objective_delta)}",
                f"- Reward: {_format_scalar(accounting.reward.reward)}",
                "",
            ]
        )
    return _finish(lines)


def render_campaign_overview(
    *,
    campaign_id: str,
    decision_rows: Sequence[Mapping[str, Any]],
    campaign_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Render the campaign-level index and current scientific state."""
    safe_rows = [redact_sensitive(row) for row in decision_rows]
    metadata = redact_sensitive(campaign_metadata or {})
    lines = [
        _front_matter(
            {
                "artifact_type": "campaign",
                "campaign_id": campaign_id,
                "decision_count": len(safe_rows),
                "schema_version": DECISION_CARD_SCHEMA_VERSION,
            }
        ),
        f"# Campaign {_safe_inline(campaign_id)}",
        "",
        "## Objective and Metadata",
        "",
        _mapping_markdown(metadata),
        "",
        "## Decision Index",
        "",
        "| Round | Trace | Action | Backend | Confidence | Status | Reward |",
        "|---:|---|---|---|---:|---|---:|",
    ]
    for row in sorted(safe_rows, key=lambda item: (int(item.get("round_index", 0)), str(item.get("trace_id", "")))):
        lines.append(
            "| {round_index} | {trace_id} | {action} | {backend} | {confidence} | {status} | {reward} |".format(
                round_index=_safe_table(row.get("round_index")),
                trace_id=_safe_table(row.get("trace_id")),
                action=_safe_table(row.get("selected_action") or row.get("action")),
                backend=_safe_table(row.get("selected_backend") or row.get("backend")),
                confidence=_safe_table(_format_scalar(row.get("confidence"))),
                status=_safe_table(row.get("status")),
                reward=_safe_table(_format_scalar(row.get("reward"))),
            )
        )
    if not safe_rows:
        lines.append("| — | — | — | — | — | — | — |")
    lines.append("")
    return _finish(lines)


def render_policy_snapshot(
    *,
    campaign_id: str,
    policy: Mapping[str, Any],
    provenance: LedgerProvenance,
) -> str:
    safe_policy = redact_sensitive(policy)
    return _finish(
        [
            _front_matter(
                {
                    "artifact_type": "policy",
                    "campaign_id": campaign_id,
                    "policy_id": provenance.policy_id or "unversioned",
                    "policy_version": provenance.policy_version or "unversioned",
                    "source_sha256": _stable_hash(safe_policy),
                }
            ),
            "# Decision Policy",
            "",
            "## Version",
            "",
            f"- Policy ID: {_safe_inline(provenance.policy_id or 'unversioned')}",
            f"- Policy version: {_safe_inline(provenance.policy_version or 'unversioned')}",
            f"- Code commit: {_safe_inline(provenance.code_commit or 'unknown')}",
            "",
            "## Policy State",
            "",
            _mapping_markdown(safe_policy),
            "",
        ]
    )


def render_nexus_snapshot(
    *,
    campaign_id: str,
    diagnostics: Mapping[str, Any],
    provenance: LedgerProvenance,
) -> str:
    safe_diagnostics = redact_sensitive(diagnostics)
    return _finish(
        [
            _front_matter(
                {
                    "artifact_type": "nexus",
                    "campaign_id": campaign_id,
                    "contract_version": provenance.nexus_contract_version or "unknown",
                    "source_sha256": _stable_hash(safe_diagnostics),
                }
            ),
            "# Nexus Optimization Health",
            "",
            f"- Contract version: {_safe_inline(provenance.nexus_contract_version or 'unknown')}",
            "",
            "## Diagnostics",
            "",
            _mapping_markdown(safe_diagnostics),
            "",
        ]
    )


def render_trajectory_figure(
    *, campaign_id: str, decision_rows: Sequence[Mapping[str, Any]]
) -> str:
    """Render a Mermaid trajectory suitable for review and paper-figure export."""
    rows = sorted(
        (redact_sensitive(row) for row in decision_rows),
        key=lambda item: (int(item.get("round_index", 0)), str(item.get("trace_id", ""))),
    )
    lines = [
        _front_matter(
            {
                "artifact_type": "trajectory_figure",
                "campaign_id": campaign_id,
                "decision_count": len(rows),
            }
        ),
        "# Decision Trajectory",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    if not rows:
        lines.append('    empty["No decisions recorded"]')
    for index, row in enumerate(rows):
        node = f"d{index}"
        label = (
            f"R{row.get('round_index', '?')} · "
            f"{row.get('selected_action') or row.get('action', 'unknown')}"
            f" · {row.get('status', 'unknown')} · reward={_format_scalar(row.get('reward'))}"
        )
        lines.append(f'    {node}["{_mermaid_text(label)}"]')
        if index:
            lines.append(f"    d{index - 1} --> {node}")
    lines.extend(["```", ""])
    return _finish(lines)


def _card_metadata(
    trace: CampaignDecisionTrace,
    status: str,
    source_hash: str,
    provenance: LedgerProvenance,
) -> dict[str, Any]:
    return {
        "artifact_type": "decision_card",
        "campaign_id": trace.campaign_id,
        "code_commit": provenance.code_commit or "unknown",
        "created_at": trace.created_at.isoformat(),
        "nexus_contract_version": provenance.nexus_contract_version or "unknown",
        "policy_id": provenance.policy_id or "unversioned",
        "policy_version": provenance.policy_version or "unversioned",
        "renderer_version": provenance.renderer_version,
        "round_index": trace.round_index,
        "schema_version": DECISION_CARD_SCHEMA_VERSION,
        "source_sha256": source_hash,
        "status": status,
        "trace_id": trace.trace_id,
    }


def _provenance_dict(provenance: LedgerProvenance) -> dict[str, Any]:
    return redact_sensitive(
        {
            "code_commit": provenance.code_commit or "unknown",
            "policy_id": provenance.policy_id or "unversioned",
            "policy_version": provenance.policy_version or "unversioned",
            "nexus_contract_version": provenance.nexus_contract_version or "unknown",
            "rubric_version": provenance.rubric_version or "unknown",
            "renderer_version": provenance.renderer_version,
            **dict(provenance.extra),
        }
    )


def _candidate_actions(strategy: Any) -> list[Mapping[str, Any]]:
    if not isinstance(strategy, Mapping):
        return []
    for key in ("available_actions", "actions", "actions_considered"):
        value = strategy.get(key)
        if isinstance(value, list | tuple):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _action_table(actions: Sequence[Mapping[str, Any]], plan: Any) -> str:
    lines = [
        "| Rank | Action | Backend | Expected improvement | Information gain | Risk | Utility | Reason |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    ranked = sorted(
        actions,
        key=lambda action: float(action.get("utility", 0.0) or 0.0),
        reverse=True,
    )
    for index, action in enumerate(ranked, 1):
        lines.append(
            "| {rank} | {action} | {backend} | {improvement} | {info} | {risk} | {utility} | {reason} |".format(
                rank=index,
                action=_safe_table(action.get("name") or action.get("action") or "unknown"),
                backend=_safe_table(action.get("backend_name") or action.get("backend") or "—"),
                improvement=_safe_table(_format_scalar(action.get("expected_improvement"))),
                info=_safe_table(_format_scalar(action.get("expected_info_gain"))),
                risk=_safe_table(_format_scalar(action.get("risk"))),
                utility=_safe_table(_format_scalar(action.get("utility"))),
                reason=_safe_table(action.get("reason") or ""),
            )
        )
    if not ranked:
        lines.append(
            "| 1 | {action} | {backend} | — | — | — | — | {reason} |".format(
                action=_safe_table(plan.action_type.value),
                backend=_safe_table(plan.candidate_generation_backend or "—"),
                reason=_safe_table(plan.rationale),
            )
        )
    return "\n".join(lines)


def _evidence_table(trace: CampaignDecisionTrace) -> str:
    lines = [
        "| Source | Kind | Summary | Weight |",
        "|---|---|---|---:|",
    ]
    for evidence in trace.evidence:
        lines.append(
            f"| {_safe_table(evidence.source)} | {_safe_table(evidence.kind)} | "
            f"{_safe_table(evidence.summary)} | {_safe_table(_format_scalar(evidence.weight))} |"
        )
    if not trace.evidence:
        lines.append("| — | — | No structured evidence was recorded. | — |")
    return "\n".join(lines)


def _outcome_table(accounting: CampaignDecisionAccounting) -> str:
    outcome = accounting.outcome
    rows = [
        ("Observed action", outcome.observed_action),
        ("Observed backend", outcome.observed_backend),
        ("Candidate count", outcome.candidate_count),
        ("Execution success", outcome.execution_success),
        ("Failure count", outcome.failure_count),
        ("Safety incidents", outcome.safety_incident_count),
        ("Objective delta", outcome.objective_delta),
        ("Proxy-gap delta", outcome.proxy_gap_delta),
        ("Validation success", outcome.validation_success),
        ("Recovery attempted", outcome.recovery_attempted),
        ("Recovery success", outcome.recovery_success),
        ("Context fulfilled", outcome.context_request_fulfilled),
        ("Human override", outcome.human_override),
    ]
    return "\n".join(
        ["| Field | Value |", "|---|---|"]
        + [f"| {_safe_table(name)} | {_safe_table(_format_scalar(value))} |" for name, value in rows]
    )


def _reward_section(accounting: CampaignDecisionAccounting) -> str:
    reward = accounting.reward
    lines = [
        f"- Total reward: {_format_scalar(reward.reward)}",
        f"- Process reward: {_format_scalar(reward.process_reward)}",
        f"- Outcome reward: {_format_scalar(reward.outcome_reward)}",
        f"- Regret: {_format_scalar(reward.regret)}",
        f"- Rubric version: {_safe_inline(reward.rubric_version)}",
        f"- Rationale: {_safe_paragraph(reward.rationale)}",
        "",
        "| Verifier | Passed | Score | Evidence |",
        "|---|---|---:|---|",
    ]
    for verification in reward.verifications:
        payload = verification.model_dump(mode="json")
        lines.append(
            f"| {_safe_table(payload.get('name'))} | {_safe_table(payload.get('passed'))} | "
            f"{_safe_table(_format_scalar(payload.get('score')))} | "
            f"{_safe_table(payload.get('rationale') or payload.get('evidence') or '')} |"
        )
    return "\n".join(lines)


def _expected_gain(actions: Sequence[Mapping[str, Any]], plan: Any) -> str:
    backend = plan.candidate_generation_backend
    selected = next(
        (
            action
            for action in actions
            if action.get("backend_name") == backend or action.get("backend") == backend
        ),
        actions[0] if actions else None,
    )
    if not selected:
        return "not quantified"
    improvement = _format_scalar(selected.get("expected_improvement"))
    information = _format_scalar(selected.get("expected_info_gain"))
    utility = _format_scalar(selected.get("utility"))
    return f"improvement={improvement}, information_gain={information}, utility={utility}"


def _mapping_markdown(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "No structured data was recorded."
    lines = []
    for path, item in _flatten_mapping(value):
        lines.append(f"- **{_safe_inline(path)}:** {_safe_inline(_format_scalar(item))}")
    return "\n".join(lines) if lines else "No structured data was recorded."


def _value_markdown(value: Any) -> str:
    if isinstance(value, Mapping):
        return _mapping_markdown(value)
    if isinstance(value, list):
        if not value:
            return "No entries."
        return "\n".join(f"- {_safe_inline(_format_scalar(item))}" for item in value)
    return _safe_paragraph(_format_scalar(value))


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []
    for key in sorted(value, key=str):
        path = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping):
            flattened.extend(_flatten_mapping(item, path))
        elif isinstance(item, list) and item and all(isinstance(entry, Mapping) for entry in item):
            for index, entry in enumerate(item):
                flattened.extend(_flatten_mapping(entry, f"{path}[{index}]"))
        else:
            flattened.append((path, item))
    return flattened


def _front_matter(metadata: Mapping[str, Any]) -> str:
    safe = redact_sensitive(metadata)
    dumped = yaml.safe_dump(
        safe,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).strip()
    return f"---\n{dumped}\n---"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _redact_text(text: str) -> str:
    redacted = _BEARER_RE.sub(REDACTED, text)
    redacted = _OPENAI_KEY_RE.sub(REDACTED, redacted)
    return _JWT_RE.sub(REDACTED, redacted)


def _failure_label(payload: Any) -> str:
    if isinstance(payload, Mapping):
        for key in ("problem", "failure_type", "error_type", "error", "message", "reason"):
            value = payload.get(key)
            if value:
                return _safe_inline(_format_scalar(value))
    return _safe_inline(_format_scalar(payload))


def _format_scalar(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        return "; ".join(f"{key}={_format_scalar(item)}" for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return ", ".join(_format_scalar(item) for item in value) if value else "—"
    return _redact_text(str(value))


def _safe_inline(value: Any) -> str:
    return _redact_text(str(value)).replace("\n", " ").replace("\r", " ").replace("`", "'")


def _safe_paragraph(value: Any) -> str:
    return _redact_text(str(value)).replace("\r\n", "\n").replace("\r", "\n")


def _safe_table(value: Any) -> str:
    return _safe_inline(value).replace("|", "\\|")


def _mermaid_text(value: Any) -> str:
    return _safe_inline(value).replace('"', "'").replace("[", "(").replace("]", ")")


def _finish(lines: Sequence[str]) -> str:
    return "\n".join(str(line).rstrip() for line in lines).rstrip() + "\n"


__all__ = [
    "DECISION_CARD_SCHEMA_VERSION",
    "LEDGER_RENDERER_VERSION",
    "LedgerProvenance",
    "RenderedDecisionBundle",
    "redact_sensitive",
    "render_campaign_overview",
    "render_completed_decision",
    "render_nexus_snapshot",
    "render_pending_decision",
    "render_policy_snapshot",
    "render_trajectory_figure",
]
