from __future__ import annotations

import subprocess

import pytest

from app.services.scientific_ledger_git import ScientificLedgerGit


def test_campaign_local_git_initializes_and_commits_only_markdown(tmp_path):
    campaign = tmp_path / "campaign-32"
    campaign.mkdir()
    card = campaign / "decision.md"
    card.write_text("# Decision\n")
    ignored = campaign / "runtime.json"
    ignored.write_text('{"internal": true}')

    backend = ScientificLedgerGit(campaign, auto_init=True)
    first = backend.commit([card], "decision: record pending")
    assert first.committed is True
    assert first.commit_sha
    tracked = subprocess.run(
        ["git", "-C", str(campaign), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert tracked == ["decision.md"]
    remotes = subprocess.run(
        ["git", "-C", str(campaign), "remote"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert remotes == []

    second = backend.commit([card], "decision: duplicate")
    assert second.committed is False
    assert second.reason == "no_changes"


def test_git_backend_refuses_non_markdown_and_outside_paths(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    backend = ScientificLedgerGit(campaign, auto_init=True)
    json_path = campaign / "record.json"
    json_path.write_text("{}")
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside")

    with pytest.raises(ValueError, match="only stage Markdown"):
        backend.commit([json_path], "bad")
    with pytest.raises(ValueError, match="inside the campaign"):
        backend.commit([outside], "bad")

    git_metadata = campaign / ".git" / "metadata.md"
    git_metadata.parent.mkdir(exist_ok=True)
    git_metadata.write_text("# Internal")
    with pytest.raises(ValueError, match="may not stage Git metadata"):
        backend.commit([git_metadata], "bad")


def test_git_backend_does_not_use_parent_source_repository(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    subprocess.run(["git", "-C", str(parent), "init"], check=True, capture_output=True)
    campaign = parent / "ledger" / "campaign"
    campaign.mkdir(parents=True)
    card = campaign / "decision.md"
    card.write_text("# Decision")

    disabled = ScientificLedgerGit(campaign, auto_init=False).commit([card], "decision")
    assert disabled.committed is False
    assert disabled.reason == "git_repository_unavailable"
    assert not (campaign / ".git").exists()
