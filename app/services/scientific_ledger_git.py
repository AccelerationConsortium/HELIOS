"""Optional local Git history for one HELIOS campaign ledger.

Each campaign directory is its own repository. This prevents experiment
commits from dirtying or rewriting the HELIOS source repository and avoids
branch-switch races between concurrent campaigns. The backend never pushes.
"""
from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LedgerGitCommit:
    committed: bool
    commit_sha: str | None = None
    message: str = ""
    reason: str = ""


class ScientificLedgerGit:
    """Stage exact Markdown paths and commit them in a campaign-local repo."""

    def __init__(
        self,
        campaign_dir: str | Path,
        *,
        auto_init: bool = False,
        author_name: str = "HELIOS Scientific Ledger",
        author_email: str = "helios-ledger@localhost",
        git_executable: str = "git",
    ) -> None:
        self.campaign_dir = Path(campaign_dir).resolve()
        self.auto_init = auto_init
        self.author_name = (
            _single_line(author_name, limit=100) or "HELIOS Scientific Ledger"
        )
        self.author_email = _safe_email(author_email)
        self.git_executable = git_executable

    def commit(self, paths: Sequence[str | Path], message: str) -> LedgerGitCommit:
        """Commit exact paths if they changed; never add unrelated files."""
        if not paths:
            return LedgerGitCommit(committed=False, reason="no_paths")
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        if not self._ensure_repository():
            return LedgerGitCommit(committed=False, reason="git_repository_unavailable")

        relative_paths = sorted({self._relative_markdown_path(path) for path in paths})
        self._run(["add", "--", *relative_paths])
        staged = self._run(["diff", "--cached", "--quiet"], check=False)
        if staged.returncode == 0:
            return LedgerGitCommit(committed=False, reason="no_changes")
        if staged.returncode != 1:
            raise RuntimeError(staged.stderr.strip() or "Unable to inspect staged ledger changes")

        safe_message = _single_line(message, limit=160) or "Record HELIOS scientific decision"
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": self.author_name,
            "GIT_AUTHOR_EMAIL": self.author_email,
            "GIT_COMMITTER_NAME": self.author_name,
            "GIT_COMMITTER_EMAIL": self.author_email,
        }
        self._run(["commit", "-m", safe_message, "--no-gpg-sign"], env=env)
        sha = self._run(["rev-parse", "HEAD"]).stdout.strip()
        return LedgerGitCommit(committed=True, commit_sha=sha, message=safe_message)

    def _ensure_repository(self) -> bool:
        git_dir = self.campaign_dir / ".git"
        if git_dir.is_dir():
            top = Path(self._run(["rev-parse", "--show-toplevel"]).stdout.strip()).resolve()
            if top != self.campaign_dir:
                raise RuntimeError("Campaign Git repository resolves outside its ledger directory")
            return True
        if not self.auto_init:
            return False
        self._run(["init"])
        self._run(["config", "user.name", self.author_name])
        self._run(["config", "user.email", self.author_email])
        return True

    def _relative_markdown_path(self, path: str | Path) -> str:
        candidate = Path(path)
        absolute = candidate.resolve() if candidate.is_absolute() else (self.campaign_dir / candidate).resolve()
        try:
            relative = absolute.relative_to(self.campaign_dir)
        except ValueError as exc:
            raise ValueError("Ledger Git paths must remain inside the campaign directory") from exc
        if absolute.suffix.casefold() != ".md":
            raise ValueError("Scientific ledger Git commits may only stage Markdown artifacts")
        if ".git" in relative.parts:
            raise ValueError("Scientific ledger Git commits may not stage Git metadata")
        return relative.as_posix()

    def _run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.git_executable, "-C", str(self.campaign_dir), *args],
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )


def _single_line(value: str, *, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()[:limit]


def _safe_email(value: str) -> str:
    candidate = _single_line(value, limit=200)
    if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+", candidate):
        return "helios-ledger@localhost"
    return candidate


__all__ = ["LedgerGitCommit", "ScientificLedgerGit"]
