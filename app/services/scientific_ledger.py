"""Git-trackable Markdown scientific memory for HELIOS campaigns.

This service projects typed decision accounting into an atomic Markdown tree,
maintains campaign indexes and policy/Nexus snapshots, supports exact-text
scientific-memory retrieval, and exposes deterministic RLVR training exports.
It is reporting-only: failures never change live campaign behavior.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from app.services.decision_markdown import (
    LedgerProvenance,
    RenderedDecisionBundle,
    redact_sensitive,
    render_campaign_overview,
    render_completed_decision,
    render_nexus_snapshot,
    render_pending_decision,
    render_policy_snapshot,
    render_trajectory_figure,
)
from app.services.decision_outcome import CampaignDecisionAccounting
from app.services.decision_trace import CampaignDecisionTrace
from app.services.hypothesis_experiment_planner import ExperimentPlan
from app.services.scientific_evidence import (
    ClaimAssessment,
    EvidenceSet,
    PromotionDecision,
    ScientificClaim,
)
from app.services.scientific_ledger_git import LedgerGitCommit, ScientificLedgerGit

try:  # Unix process lock; HELIOS production targets Linux/macOS.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback still has thread locking.
    fcntl = None  # type: ignore[assignment]


RLVR_EXPORT_SCHEMA_VERSION = "helios.rlvr/v1"
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class LedgerSearchHit:
    campaign_id: str
    path: str
    title: str
    line_number: int
    snippet: str


@dataclass(frozen=True)
class LedgerWriteResult:
    campaign_id: str
    campaign_directory: str
    status: str
    changed_paths: tuple[str, ...]
    unchanged_paths: tuple[str, ...]
    git_commit: LedgerGitCommit | None = None


class ScientificLedger:
    """Atomic campaign Markdown storage with optional campaign-local Git."""

    def __init__(
        self,
        root: str | Path,
        *,
        workspace_root: str | Path | None = None,
        git_enabled: bool = False,
        git_auto_init: bool = False,
        git_author_name: str = "HELIOS Scientific Ledger",
        git_author_email: str = "helios-ledger@localhost",
    ) -> None:
        self.root = Path(root).resolve()
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        self.git_enabled = git_enabled
        self.git_auto_init = git_auto_init
        self.git_author_name = git_author_name
        self.git_author_email = git_author_email

    def record_pending(
        self,
        trace: CampaignDecisionTrace,
        *,
        campaign_metadata: Mapping[str, Any] | None = None,
        policy_snapshot: Mapping[str, Any] | None = None,
        provenance: LedgerProvenance | None = None,
    ) -> LedgerWriteResult:
        """Write the decision before execution with Outcome=Pending."""
        effective_provenance = provenance or self.provenance_for(trace)
        bundle = render_pending_decision(
            trace,
            provenance=effective_provenance,
            campaign_metadata=campaign_metadata,
        )
        return self._record_bundle(
            bundle,
            trace=trace,
            provenance=effective_provenance,
            campaign_metadata=campaign_metadata or {},
            policy_snapshot=policy_snapshot or trace.decision_plan.strategy_trace,
            git_message=f"decision: record round {trace.round_index:03d} pending",
        )

    def record_completed(
        self,
        accounting: CampaignDecisionAccounting,
        *,
        campaign_metadata: Mapping[str, Any] | None = None,
        policy_snapshot: Mapping[str, Any] | None = None,
        observations: Sequence[Any] | None = None,
        failures: Sequence[Any] | None = None,
        recovery_events: Sequence[Any] | None = None,
        provenance: LedgerProvenance | None = None,
    ) -> LedgerWriteResult:
        """Finalize one card with its outcome, reward, failures, and recovery."""
        effective_provenance = provenance or self.provenance_for(
            accounting.trace, accounting=accounting
        )
        bundle = render_completed_decision(
            accounting,
            provenance=effective_provenance,
            campaign_metadata=campaign_metadata,
            observations=observations,
            failures=failures,
            recovery_events=recovery_events,
        )
        return self._record_bundle(
            bundle,
            trace=accounting.trace,
            provenance=effective_provenance,
            campaign_metadata=campaign_metadata or {},
            policy_snapshot=policy_snapshot or accounting.trace.decision_plan.strategy_trace,
            git_message=f"outcome: finalize round {accounting.trace.round_index:03d}",
        )

    def record_claim_evidence(
        self,
        *,
        campaign_id: str,
        claim: ScientificClaim,
        evidence: EvidenceSet,
        assessment: ClaimAssessment,
        promotion_decision: PromotionDecision | None = None,
    ) -> LedgerWriteResult:
        """Persist a typed claim posterior as a reviewable Markdown artifact."""

        ids = {claim.claim_id, evidence.claim_id, assessment.claim_id}
        if promotion_decision is not None:
            ids.add(promotion_decision.claim_id)
        if len(ids) != 1:
            raise ValueError("claim, evidence, assessment, and promotion decision must align")
        campaign_dir = self.campaign_directory(campaign_id)
        relative = f"evidence/claims/{safe_path_component(claim.claim_id)}.md"
        content = _render_claim_evidence(claim, evidence, assessment, promotion_decision)
        with self._campaign_lock(campaign_id):
            path = _validated_markdown_path(campaign_dir, relative)
            changed: list[Path] = []
            unchanged: list[Path] = []
            (changed if _atomic_write_markdown(path, content, campaign_dir) else unchanged).append(path)
            index_path = campaign_dir / "evidence" / "index.md"
            index_content = _render_scientific_evidence_index(campaign_id, campaign_dir)
            (changed if _atomic_write_markdown(
                index_path, index_content, campaign_dir
            ) else unchanged).append(index_path)
            git_commit = self._commit(
                campaign_dir,
                changed,
                f"evidence: assess claim {claim.claim_id}",
            )
        return LedgerWriteResult(
            campaign_id=campaign_id,
            campaign_directory=str(campaign_dir),
            status="claim_evidence",
            changed_paths=tuple(item.relative_to(campaign_dir).as_posix() for item in changed),
            unchanged_paths=tuple(item.relative_to(campaign_dir).as_posix() for item in unchanged),
            git_commit=git_commit,
        )

    def record_experiment_plan(
        self,
        *,
        campaign_id: str,
        plan: ExperimentPlan,
    ) -> LedgerWriteResult:
        """Persist a shadow hypothesis-discrimination plan for operator review."""

        campaign_dir = self.campaign_directory(campaign_id)
        relative = f"evidence/plans/{safe_path_component(plan.plan_id)}.md"
        content = _render_experiment_plan(plan)
        with self._campaign_lock(campaign_id):
            path = _validated_markdown_path(campaign_dir, relative)
            changed: list[Path] = []
            unchanged: list[Path] = []
            (changed if _atomic_write_markdown(path, content, campaign_dir) else unchanged).append(path)
            index_path = campaign_dir / "evidence" / "index.md"
            index_content = _render_scientific_evidence_index(campaign_id, campaign_dir)
            (changed if _atomic_write_markdown(
                index_path, index_content, campaign_dir
            ) else unchanged).append(index_path)
            git_commit = self._commit(
                campaign_dir,
                changed,
                f"evidence: record experiment plan {plan.plan_id}",
            )
        return LedgerWriteResult(
            campaign_id=campaign_id,
            campaign_directory=str(campaign_dir),
            status="experiment_plan",
            changed_paths=tuple(item.relative_to(campaign_dir).as_posix() for item in changed),
            unchanged_paths=tuple(item.relative_to(campaign_dir).as_posix() for item in unchanged),
            git_commit=git_commit,
        )

    def search(
        self,
        query: str,
        *,
        campaign_id: str | None = None,
        limit: int = 50,
    ) -> list[LedgerSearchHit]:
        """Case-insensitive exact-text retrieval over Markdown; no embeddings."""
        needle = query.strip().casefold()
        if not needle:
            raise ValueError("Scientific memory query must not be empty")
        if limit < 1 or limit > 1000:
            raise ValueError("Scientific memory result limit must be between 1 and 1000")
        roots = [self.campaign_directory(campaign_id)] if campaign_id else self._campaign_directories()
        hits: list[LedgerSearchHit] = []
        for campaign_dir in roots:
            if not campaign_dir.is_dir():
                continue
            for path in sorted(campaign_dir.rglob("*.md")):
                if ".git" in path.parts or path.is_symlink():
                    continue
                try:
                    path.resolve().relative_to(campaign_dir.resolve())
                except ValueError:
                    continue
                text = path.read_text(encoding="utf-8")
                title = _first_title(text) or path.stem
                for line_number, line in enumerate(text.splitlines(), 1):
                    if needle in line.casefold():
                        hits.append(
                            LedgerSearchHit(
                                campaign_id=_campaign_id_from_dir(campaign_dir),
                                path=path.relative_to(campaign_dir).as_posix(),
                                title=title,
                                line_number=line_number,
                                snippet=line.strip()[:500],
                            )
                        )
                        if len(hits) >= limit:
                            return hits
        return hits

    def export_rlvr_records(self, campaign_id: str | None = None) -> list[dict[str, Any]]:
        """Build deterministic machine records from the typed trajectory store."""
        from app.services.decision_trajectory import load_trajectories

        records: list[dict[str, Any]] = []
        for row in load_trajectories(campaign_id):
            if row.get("layer") != "campaign":
                continue
            trajectory = redact_sensitive(row.get("trajectory") or {})
            if not isinstance(trajectory, Mapping):
                continue
            trace = trajectory.get("trace") or {}
            outcome = trajectory.get("outcome") or {}
            reward = trajectory.get("reward") or {}
            if not isinstance(trace, Mapping):
                continue
            plan = trace.get("decision_plan") or {}
            context = trace.get("context") or {}
            strategy = plan.get("strategy_trace") or {} if isinstance(plan, Mapping) else {}
            candidates = _candidate_actions(strategy)
            record = {
                "schema_version": RLVR_EXPORT_SCHEMA_VERSION,
                "decision_id": row.get("trace_id"),
                "campaign_id": row.get("campaign_id"),
                "round_index": row.get("round_index"),
                "question": "What should happen next in this scientific campaign?",
                "context": context,
                "candidate_actions": candidates,
                "chosen_action": plan.get("action_type") if isinstance(plan, Mapping) else None,
                "chosen_backend": (
                    plan.get("candidate_generation_backend") if isinstance(plan, Mapping) else None
                ),
                "rationale": plan.get("rationale") if isinstance(plan, Mapping) else None,
                "confidence": plan.get("confidence") if isinstance(plan, Mapping) else None,
                "outcome": outcome,
                "reward": reward,
                "verifier_report": row.get("verifier_report") or [],
                "rubric_version": row.get("rubric_version"),
                "trajectory_schema_version": row.get("trajectory_schema_version"),
                "created_at": row.get("created_at"),
            }
            records.append(redact_sensitive(record))
        return sorted(
            records,
            key=lambda item: (
                str(item.get("campaign_id") or ""),
                int(item.get("round_index") or 0),
                str(item.get("decision_id") or ""),
            ),
        )

    def export_rlvr_jsonl(self, campaign_id: str | None = None) -> str:
        """Return one canonical RLVR JSON object per line for downstream training."""
        return "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in self.export_rlvr_records(campaign_id)
        )

    def write_training_dataset_markdown(self, campaign_id: str) -> Path:
        """Refresh the human-reviewable view of the RLVR training rows."""
        campaign_dir = self.campaign_directory(campaign_id)
        records = self.export_rlvr_records(campaign_id)
        content = _render_training_dataset(campaign_id, records)
        with self._campaign_lock(campaign_id):
            path = campaign_dir / "training_dataset.md"
            _atomic_write_markdown(path, content, campaign_dir)
        return path

    def provenance_for(
        self,
        trace: CampaignDecisionTrace,
        *,
        accounting: CampaignDecisionAccounting | None = None,
    ) -> LedgerProvenance:
        strategy = trace.decision_plan.strategy_trace
        policy_id, policy_version = _policy_identity(strategy)
        nexus_version = _first_nested_value(
            trace.context.nexus_diagnostics,
            ("contract_version", "schema_version", "version"),
        )
        code_commit, code_dirty = _code_identity(self.workspace_root)
        return LedgerProvenance(
            code_commit=code_commit,
            policy_id=policy_id,
            policy_version=policy_version,
            nexus_contract_version=nexus_version,
            rubric_version=accounting.reward.rubric_version if accounting else None,
            extra={"code_dirty": code_dirty},
        )

    def campaign_directory(self, campaign_id: str) -> Path:
        if not str(campaign_id).strip():
            raise ValueError("campaign_id must not be empty")
        campaigns_root = (self.root / "campaigns").resolve()
        campaign_dir = (campaigns_root / safe_path_component(campaign_id)).resolve()
        try:
            campaign_dir.relative_to(campaigns_root)
        except ValueError as exc:
            raise ValueError("Campaign ledger directory escapes the ledger root") from exc
        return campaign_dir

    def _record_bundle(
        self,
        bundle: RenderedDecisionBundle,
        *,
        trace: CampaignDecisionTrace,
        provenance: LedgerProvenance,
        campaign_metadata: Mapping[str, Any],
        policy_snapshot: Mapping[str, Any],
        git_message: str,
    ) -> LedgerWriteResult:
        campaign_dir = self.campaign_directory(bundle.campaign_id)
        changed: list[Path] = []
        unchanged: list[Path] = []
        with self._campaign_lock(bundle.campaign_id):
            for relative, content in sorted(bundle.files.items()):
                path = _validated_markdown_path(campaign_dir, relative)
                (changed if _atomic_write_markdown(path, content, campaign_dir) else unchanged).append(path)

            policy_content = render_policy_snapshot(
                campaign_id=bundle.campaign_id,
                policy=policy_snapshot,
                provenance=provenance,
            )
            policy_path = campaign_dir / "policy.md"
            (changed if _atomic_write_markdown(policy_path, policy_content, campaign_dir) else unchanged).append(
                policy_path
            )
            policy_version_path = (
                campaign_dir
                / "policy_versions"
                / f"{safe_path_component(provenance.policy_version or 'unversioned')}.md"
            )
            if not policy_version_path.exists():
                (changed if _atomic_write_markdown(
                    policy_version_path, policy_content, campaign_dir
                ) else unchanged).append(policy_version_path)

            nexus_content = render_nexus_snapshot(
                campaign_id=bundle.campaign_id,
                diagnostics=trace.context.nexus_diagnostics,
                provenance=provenance,
            )
            nexus_path = campaign_dir / "nexus.md"
            (changed if _atomic_write_markdown(nexus_path, nexus_content, campaign_dir) else unchanged).append(
                nexus_path
            )

            decision_rows = _read_decision_rows(campaign_dir)
            overview = render_campaign_overview(
                campaign_id=bundle.campaign_id,
                decision_rows=decision_rows,
                campaign_metadata=campaign_metadata,
            )
            for name, content in (
                ("campaign.md", overview),
                ("index.md", _render_navigation(bundle.campaign_id, decision_rows)),
                ("summary.md", _render_campaign_summary(bundle.campaign_id, decision_rows)),
                ("trajectory.md", render_trajectory_figure(
                    campaign_id=bundle.campaign_id, decision_rows=decision_rows
                )),
            ):
                path = campaign_dir / name
                (changed if _atomic_write_markdown(path, content, campaign_dir) else unchanged).append(path)

            training_records = self.export_rlvr_records(bundle.campaign_id)
            training_path = campaign_dir / "training_dataset.md"
            training_content = _render_training_dataset(bundle.campaign_id, training_records)
            (changed if _atomic_write_markdown(
                training_path, training_content, campaign_dir
            ) else unchanged).append(training_path)

            git_commit = self._commit(campaign_dir, changed, git_message)

        return LedgerWriteResult(
            campaign_id=bundle.campaign_id,
            campaign_directory=str(campaign_dir),
            status=bundle.status,
            changed_paths=tuple(path.relative_to(campaign_dir).as_posix() for path in changed),
            unchanged_paths=tuple(path.relative_to(campaign_dir).as_posix() for path in unchanged),
            git_commit=git_commit,
        )

    def _commit(
        self, campaign_dir: Path, changed: Sequence[Path], message: str
    ) -> LedgerGitCommit | None:
        if not self.git_enabled:
            return None
        return ScientificLedgerGit(
            campaign_dir,
            auto_init=self.git_auto_init,
            author_name=self.git_author_name,
            author_email=self.git_author_email,
        ).commit(changed, message)

    def _campaign_directories(self) -> list[Path]:
        campaigns_root = self.root / "campaigns"
        if not campaigns_root.is_dir():
            return []
        return sorted(path for path in campaigns_root.iterdir() if path.is_dir())

    @contextmanager
    def _campaign_lock(self, campaign_id: str) -> Iterator[None]:
        safe_id = safe_path_component(campaign_id)
        key = str((self.root / safe_id).resolve())
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
        with thread_lock:
            lock_dir = self.root / ".locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path = lock_dir / f"{safe_id}.lock"
            with lock_path.open("a+", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def safe_path_component(value: Any) -> str:
    """Return a traversal-safe, collision-resistant filesystem component."""
    original = str(value).strip()
    if not original:
        raise ValueError("Ledger path component must not be empty")
    normalized = _SAFE_COMPONENT_RE.sub("-", original).strip(" .-_")[:80]
    normalized = normalized or "item"
    if normalized != original or original in {".", ".."}:
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
        normalized = f"{normalized}-{digest}"
    return normalized


def get_scientific_ledger() -> ScientificLedger:
    """Build a ledger from current settings without module-level mutable state."""
    from app.core.config import get_settings

    settings = get_settings()
    return ScientificLedger(
        settings.scientific_ledger_root,
        workspace_root=settings.workspace_root,
        git_enabled=settings.scientific_ledger_git_enabled,
        git_auto_init=settings.scientific_ledger_git_auto_init,
        git_author_name=settings.scientific_ledger_git_author_name,
        git_author_email=settings.scientific_ledger_git_author_email,
    )


def _validated_markdown_path(campaign_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or ".git" in pure.parts
        or pure.suffix.casefold() != ".md"
    ):
        raise ValueError(f"Invalid scientific ledger Markdown path: {relative!r}")
    path = (campaign_dir / Path(*pure.parts)).resolve()
    try:
        path.relative_to(campaign_dir)
    except ValueError as exc:
        raise ValueError("Scientific ledger path escapes the campaign directory") from exc
    return path


def _atomic_write_markdown(path: Path, content: str, campaign_dir: Path) -> bool:
    path = path.resolve()
    try:
        path.relative_to(campaign_dir.resolve())
    except ValueError as exc:
        raise ValueError("Scientific ledger write escapes the campaign directory") from exc
    if path.suffix.casefold() != ".md":
        raise ValueError("Scientific ledger artifacts must be Markdown files")
    encoded = content.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
        return True
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _fsync_directory(directory: Path) -> None:
    """Best-effort durability for the rename that publishes an artifact."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform/filesystem dependent.
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - some filesystems reject directory fsync.
        pass
    finally:
        os.close(descriptor)


def _read_decision_rows(campaign_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((campaign_dir / "rounds").glob("*/decision_*.md")):
        metadata = _front_matter(path)
        if not metadata:
            continue
        metadata["path"] = path.relative_to(campaign_dir).as_posix()
        rows.append(metadata)
    return rows


def _front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        return {}
    decoded = yaml.safe_load("\n".join(lines[1:end])) or {}
    return decoded if isinstance(decoded, dict) else {}


def _render_navigation(campaign_id: str, rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "---",
        "artifact_type: index",
        f"campaign_id: {json.dumps(campaign_id, ensure_ascii=False)}",
        "---",
        f"# Campaign {campaign_id} Artifact Index",
        "",
        "- [Campaign overview](campaign.md)",
        "- [Campaign summary](summary.md)",
        "- [Decision trajectory](trajectory.md)",
        "- [Decision policy](policy.md)",
        "- [Nexus health](nexus.md)",
        "- [RLVR training dataset](training_dataset.md)",
        "",
        "## Rounds",
        "",
    ]
    for row in sorted(rows, key=lambda item: int(item.get("round_index", 0))):
        path = str(row.get("path", ""))
        lines.append(
            f"- [Round {row.get('round_index')} — {row.get('selected_action', 'decision')}]({path})"
        )
    if not rows:
        lines.append("- No rounds recorded.")
    return "\n".join(lines).rstrip() + "\n"


def _render_campaign_summary(campaign_id: str, rows: Sequence[Mapping[str, Any]]) -> str:
    completed = [row for row in rows if row.get("status") == "completed"]
    rewards = [float(row["reward"]) for row in completed if isinstance(row.get("reward"), int | float)]
    action_counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("selected_action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    lines = [
        "---",
        "artifact_type: campaign_summary",
        f"campaign_id: {json.dumps(campaign_id, ensure_ascii=False)}",
        f"decision_count: {len(rows)}",
        "---",
        "# Campaign Summary",
        "",
        f"- Decisions: {len(rows)}",
        f"- Completed decisions: {len(completed)}",
        f"- Pending decisions: {len(rows) - len(completed)}",
        f"- Mean reward: {sum(rewards) / len(rewards):.6g}" if rewards else "- Mean reward: —",
        "",
        "## Action Distribution",
        "",
    ]
    lines.extend(f"- {action}: {count}" for action, count in sorted(action_counts.items()))
    if not action_counts:
        lines.append("- No actions recorded.")
    return "\n".join(lines).rstrip() + "\n"


def _render_training_dataset(campaign_id: str, records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "---",
        "artifact_type: rlvr_training_dataset",
        f"campaign_id: {json.dumps(campaign_id, ensure_ascii=False)}",
        f"record_count: {len(records)}",
        f"schema_version: {RLVR_EXPORT_SCHEMA_VERSION}",
        "---",
        "# RLVR Training Dataset",
        "",
        "This is the human-reviewable projection. Machine consumers should use the deterministic JSONL export API.",
        "",
        "| Round | Decision | Chosen action | Backend | Reward | Rubric |",
        "|---:|---|---|---|---:|---|",
    ]
    for record in records:
        reward = record.get("reward") or {}
        total = reward.get("reward") if isinstance(reward, Mapping) else None
        lines.append(
            "| {round} | {decision} | {action} | {backend} | {reward} | {rubric} |".format(
                round=_table(record.get("round_index")),
                decision=_table(record.get("decision_id")),
                action=_table(record.get("chosen_action")),
                backend=_table(record.get("chosen_backend")),
                reward=_table("—" if total is None else f"{float(total):.6g}"),
                rubric=_table(record.get("rubric_version")),
            )
        )
    if not records:
        lines.append("| — | — | — | — | — | — |")
    return "\n".join(lines).rstrip() + "\n"


def _policy_identity(strategy: Any) -> tuple[str, str]:
    if isinstance(strategy, Mapping):
        policy_id = _first_nested_value(strategy, ("policy_id",))
        version = _first_nested_value(strategy, ("policy_version", "version"))
        return policy_id or "helios-strategy-selector", version or "unversioned"
    return "helios-strategy-selector", "unversioned"


def _first_nested_value(value: Any, keys: Sequence[str]) -> str | None:
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if candidate not in (None, "") and not isinstance(candidate, Mapping | list | tuple):
                return str(candidate)
        for item in value.values():
            nested = _first_nested_value(item, keys)
            if nested:
                return nested
    elif isinstance(value, list | tuple):
        for item in value:
            nested = _first_nested_value(item, keys)
            if nested:
                return nested
    return None


def _code_identity(workspace_root: Path | None) -> tuple[str | None, bool]:
    if workspace_root is None or not workspace_root.is_dir():
        return None, False
    try:
        sha = subprocess.run(
            ["git", "-C", str(workspace_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(workspace_root), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        return None, False


def _candidate_actions(strategy: Any) -> list[Any]:
    if not isinstance(strategy, Mapping):
        return []
    for key in ("available_actions", "actions", "actions_considered"):
        value = strategy.get(key)
        if isinstance(value, list | tuple):
            return list(value)
    return []


def _campaign_id_from_dir(campaign_dir: Path) -> str:
    metadata = _front_matter(campaign_dir / "campaign.md") if (campaign_dir / "campaign.md").exists() else {}
    return str(metadata.get("campaign_id") or campaign_dir.name)


def _first_title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _table(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def _render_claim_evidence(
    claim: ScientificClaim,
    evidence: EvidenceSet,
    assessment: ClaimAssessment,
    promotion: PromotionDecision | None,
) -> str:
    promotion_status = "not_evaluated" if promotion is None else str(promotion.promotion_allowed).lower()
    lines = [
        "---",
        "artifact_type: scientific_claim_evidence",
        f"claim_id: {json.dumps(claim.claim_id, ensure_ascii=False)}",
        f"claim_status: {assessment.status.value}",
        f"posterior_probability: {assessment.posterior_probability:.12g}",
        f"promotion_allowed: {promotion_status}",
        "shadow_only: true",
        "---",
        f"# Scientific Claim — {_table(redact_sensitive(claim.claim_id))}",
        "",
        "## Claim",
        "",
        _table(redact_sensitive(claim.statement)),
        "",
        f"- Scope: {_table(redact_sensitive(claim.scope))}",
        f"- Prior probability: {claim.prior_probability:.6g}",
        f"- Prior version: {_table(redact_sensitive(claim.prior_version))}",
        f"- Prior rationale: {_table(redact_sensitive(claim.prior_rationale or 'not recorded'))}",
        f"- Posterior probability: {assessment.posterior_probability:.6g}",
        f"- Status: {assessment.status.value}",
        f"- Cumulative log Bayes factor: {assessment.cumulative_log_bayes_factor:.6g}",
        f"- Method: {_table(assessment.method)}",
        "",
        "## Falsifiability",
        "",
    ]
    lines.extend(
        f"- {_table(redact_sensitive(item))}" for item in claim.falsifying_observations
    )
    if not claim.falsifying_observations:
        lines.append("- No falsifying observation has been declared; promotion should remain blocked.")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "| Evidence | Design | Log BF | Preregistered | Replicates | Blocks | Falsifier | Source |",
            "|---|---|---:|---|---:|---|---|---|",
        ]
    )
    for item in evidence.items:
        lines.append(
            "| {evidence_id} | {design} | {log_bf} | {registered} | {replicates} | "
            "{blocks} | {falsifier} | {source} |".format(
                evidence_id=_table(redact_sensitive(item.evidence_id)),
                design=item.design.value,
                log_bf="—" if item.log_bayes_factor is None else f"{item.log_bayes_factor:.6g}",
                registered="yes" if item.registered_before_observation else "no",
                replicates=item.replicate_count,
                blocks=_table(", ".join(item.block_ids) or "—"),
                falsifier="yes" if item.falsifier_triggered else "no",
                source=_table(redact_sensitive(item.source)),
            )
        )
    if not evidence.items:
        lines.append("| — | — | — | — | — | — | — | No evidence recorded |")
    lines.extend(
        [
            "",
            "## Evidence Quality",
            "",
            f"- Scored evidence: {assessment.scored_evidence_count}",
            f"- Descriptive/unscored evidence: {assessment.unscored_evidence_count}",
            f"- Prospective evidence: {assessment.prospective_evidence_count}",
            f"- Interventional evidence: {assessment.interventional_evidence_count}",
            f"- Preregistered evidence: {assessment.preregistered_evidence_count}",
            f"- Independent blocks: {assessment.independent_block_count}",
            f"- Safety incidents: {assessment.safety_incident_count}",
            f"- Falsifier triggered: {'yes' if assessment.falsifier_triggered else 'no'}",
            "",
            "## Unmet Requirements",
            "",
        ]
    )
    lines.extend(f"- {_table(item)}" for item in assessment.unmet_requirements)
    if not assessment.unmet_requirements:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {_table(item)}" for item in assessment.warnings)
    if not assessment.warnings:
        lines.append("- None")
    lines.extend(["", "## Promotion Gate", ""])
    if promotion is None:
        lines.append("- Not evaluated")
    else:
        lines.extend(
            [
                f"- Evidence criteria satisfied: {'yes' if promotion.evidence_criteria_satisfied else 'no'}",
                f"- Human approval required: {'yes' if promotion.human_approval_required else 'no'}",
                f"- Human approved: {'yes' if promotion.human_approved else 'no'}",
                f"- Promotion allowed: {'yes' if promotion.promotion_allowed else 'no'}",
                "- Auto-applied: no",
            ]
        )
        lines.extend(f"- Reason: {_table(item)}" for item in promotion.reasons)
    return "\n".join(lines).rstrip() + "\n"


def _render_experiment_plan(plan: ExperimentPlan) -> str:
    lines = [
        "---",
        "artifact_type: hypothesis_discrimination_plan",
        f"plan_id: {json.dumps(plan.plan_id, ensure_ascii=False)}",
        f"eligible_experiment_count: {len(plan.ranked_experiments)}",
        f"excluded_experiment_count: {len(plan.excluded_experiments)}",
        "operator_approval_required: true",
        "shadow_only: true",
        "---",
        f"# Hypothesis-Discrimination Plan — {_table(redact_sensitive(plan.plan_id))}",
        "",
        f"- Objective: {_table(plan.objective)}",
        f"- Prior scenarios: {_table(', '.join(plan.prior_scenario_ids))}",
        "- This artifact is advisory and cannot execute experiments.",
        "",
        "## Ranked Experiments",
        "",
        "| Rank | Experiment | Robust EIG | Mean EIG | Robust EIG / Cost |",
        "|---:|---|---:|---:|---:|",
    ]
    for score in plan.ranked_experiments:
        lines.append(
            f"| {score.rank} | {_table(redact_sensitive(score.experiment_id))} | "
            f"{score.robust_expected_information_gain:.6g} | "
            f"{score.mean_expected_information_gain:.6g} | "
            f"{score.information_gain_per_cost:.6g} |"
        )
    if not plan.ranked_experiments:
        lines.append("| — | — | — | — | — |")
    lines.extend(
        [
            "",
            "## Excluded Experiments",
            "",
            "| Experiment | Reason |",
            "|---|---|",
        ]
    )
    for score in plan.excluded_experiments:
        lines.append(
            f"| {_table(redact_sensitive(score.experiment_id))} | "
            f"{_table('; '.join(score.reasons))} |"
        )
    if not plan.excluded_experiments:
        lines.append("| — | None |")
    return "\n".join(lines).rstrip() + "\n"


def _render_scientific_evidence_index(campaign_id: str, campaign_dir: Path) -> str:
    lines = [
        "---",
        "artifact_type: scientific_evidence_index",
        f"campaign_id: {json.dumps(campaign_id, ensure_ascii=False)}",
        "---",
        "# Scientific Evidence",
        "",
        "## Claims",
        "",
    ]
    claim_paths = sorted((campaign_dir / "evidence" / "claims").glob("*.md"))
    for path in claim_paths:
        metadata = _front_matter(path)
        relative = path.relative_to(campaign_dir / "evidence").as_posix()
        label = metadata.get("claim_id") or path.stem
        status = metadata.get("claim_status") or "unknown"
        lines.append(f"- [{_table(label)}]({relative}) — {status}")
    if not claim_paths:
        lines.append("- No scientific claims recorded.")
    lines.extend(["", "## Experiment Plans", ""])
    plan_paths = sorted((campaign_dir / "evidence" / "plans").glob("*.md"))
    for path in plan_paths:
        metadata = _front_matter(path)
        relative = path.relative_to(campaign_dir / "evidence").as_posix()
        label = metadata.get("plan_id") or path.stem
        lines.append(f"- [{_table(label)}]({relative})")
    if not plan_paths:
        lines.append("- No hypothesis-discrimination plans recorded.")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "LedgerSearchHit",
    "LedgerWriteResult",
    "RLVR_EXPORT_SCHEMA_VERSION",
    "ScientificLedger",
    "get_scientific_ledger",
    "safe_path_component",
]
