from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.services.decision_trajectory import persist_campaign_trajectory
from app.services.scientific_ledger import ScientificLedger, safe_path_component
from tests.fixtures.scientific_ledger import decision_accounting, decision_trace


@pytest.fixture
def db_env(monkeypatch, request, tmp_path):
    from app.core.config import get_settings
    from app.core.db import init_db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "orchestrator.db"))
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "objects"))
    get_settings.cache_clear()
    request.addfinalizer(get_settings.cache_clear)
    init_db()


def test_pending_then_completed_lifecycle_builds_full_markdown_tree(tmp_path, db_env):
    ledger = ScientificLedger(tmp_path / "ledger", workspace_root=Path.cwd())
    trace = decision_trace()
    pending = ledger.record_pending(trace, campaign_metadata={"domain": "chemistry"})
    assert pending.status == "pending"
    campaign_dir = Path(pending.campaign_directory)
    assert (campaign_dir / "rounds/003/decision_003.md").exists()
    assert (campaign_dir / "campaign.md").exists()
    assert (campaign_dir / "policy.md").exists()
    assert (campaign_dir / "policy_versions/v4.md").exists()
    assert (campaign_dir / "nexus.md").exists()
    assert (campaign_dir / "trajectory.md").exists()
    assert (campaign_dir / "training_dataset.md").exists()

    accounting = decision_accounting()
    persist_campaign_trajectory(accounting)
    completed = ledger.record_completed(
        accounting,
        failures=[{"problem": "OT2 aspiration failed", "root_cause": "pipette offset"}],
        recovery_events=[{"fix": "increase z offset 0.5mm", "result": "pass"}],
    )
    assert completed.status == "completed"
    card = (campaign_dir / "rounds/003/decision_003.md").read_text()
    assert "status: completed" in card
    assert "record_count: 1" in (campaign_dir / "training_dataset.md").read_text()
    assert all(path.suffix == ".md" for path in campaign_dir.rglob("*.md"))


def test_scientific_memory_search_finds_failure_without_embeddings(tmp_path, db_env):
    ledger = ScientificLedger(tmp_path / "ledger")
    accounting = decision_accounting()
    persist_campaign_trajectory(accounting)
    ledger.record_completed(
        accounting,
        failures=[{"root_cause": "pipette offset", "error": "aspiration failed"}],
    )
    hits = ledger.search("pipette offset")
    assert hits
    assert hits[0].campaign_id == "campaign-32"
    assert hits[0].path == "rounds/003/failure.md"


def test_rlvr_export_uses_typed_trajectory_not_markdown_scraping(tmp_path, db_env):
    ledger = ScientificLedger(tmp_path / "ledger")
    accounting = decision_accounting()
    persist_campaign_trajectory(accounting)
    records = ledger.export_rlvr_records("campaign-32")
    assert len(records) == 1
    assert records[0]["schema_version"] == "helios.rlvr/v1"
    assert records[0]["chosen_action"] == "propose_candidates"
    assert records[0]["candidate_actions"][0]["name"] == "validation"
    assert records[0]["reward"]["reward"] == accounting.reward.reward
    assert ledger.export_rlvr_jsonl("campaign-32").count("\n") == 0


def test_campaign_id_is_path_safe_and_collision_resistant(tmp_path):
    ledger = ScientificLedger(tmp_path / "ledger")
    trace = decision_trace(campaign_id="../../danger")
    result = ledger.record_pending(trace)
    campaign_dir = Path(result.campaign_directory)
    assert campaign_dir.is_relative_to((tmp_path / "ledger").resolve())
    assert ".." not in campaign_dir.name
    assert safe_path_component("a/b") != safe_path_component("a-b")


def test_campaign_directory_rejects_symlink_escape(tmp_path):
    ledger = ScientificLedger(tmp_path / "ledger")
    campaigns = tmp_path / "ledger" / "campaigns"
    campaigns.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (campaigns / "escaped").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the ledger root"):
        ledger.campaign_directory("escaped")


def test_concurrent_idempotent_writes_leave_no_partial_files(tmp_path, db_env):
    ledger = ScientificLedger(tmp_path / "ledger")
    accounting = decision_accounting()
    persist_campaign_trajectory(accounting)

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda _: ledger.record_completed(accounting), range(12)))

    campaign_dir = Path(results[0].campaign_directory)
    assert (campaign_dir / "rounds/003/decision_003.md").read_text().endswith("\n")
    assert not list(campaign_dir.rglob("*.tmp"))
    assert any(result.changed_paths for result in results)
    assert any(not result.changed_paths for result in results)


def test_ledger_git_records_pending_and_completed_transitions(tmp_path, db_env):
    ledger = ScientificLedger(
        tmp_path / "ledger",
        git_enabled=True,
        git_auto_init=True,
    )
    trace = decision_trace()
    pending = ledger.record_pending(trace)
    assert pending.git_commit is not None
    assert pending.git_commit.committed is True

    accounting = decision_accounting()
    persist_campaign_trajectory(accounting)
    completed = ledger.record_completed(accounting)
    assert completed.git_commit is not None
    assert completed.git_commit.committed is True

    log = subprocess.run(
        ["git", "-C", completed.campaign_directory, "log", "--format=%s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert log == ["outcome: finalize round 003", "decision: record round 003 pending"]


def test_search_validates_query_and_limit(tmp_path):
    ledger = ScientificLedger(tmp_path / "ledger")
    with pytest.raises(ValueError, match="must not be empty"):
        ledger.search(" ")
    with pytest.raises(ValueError, match="between 1 and 1000"):
        ledger.search("offset", limit=0)


def test_search_does_not_follow_markdown_symlinks(tmp_path):
    ledger = ScientificLedger(tmp_path / "ledger")
    campaign = ledger.campaign_directory("campaign-1")
    campaign.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("# Secret\npipette offset", encoding="utf-8")
    (campaign / "linked.md").symlink_to(outside)

    assert ledger.search("pipette offset") == []
