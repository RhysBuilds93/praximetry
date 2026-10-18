"""Best-effort git provenance for hosted evaluation captures."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    return result.stdout.strip() or None


def _github_default_branch() -> str | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    repository = event.get("repository") if isinstance(event, dict) else None
    default = repository.get("default_branch") if isinstance(repository, dict) else None
    return default.strip() or None if isinstance(default, str) else None


def detect_git_info(
    *, branch: str | None = None, commit: str | None = None, default_branch: str | None = None
) -> dict[str, str]:
    """Resolve each field independently: explicit option > GitHub Actions > git.

    Missing tools, unreadable CI payloads, and unknown refs never block an eval.
    A detached HEAD has a commit but no local branch; CI may still supply one.
    """
    if branch is None:
        branch = (
            os.environ.get("GITHUB_HEAD_REF")
            or os.environ.get("GITHUB_REF_NAME")
            or _git("rev-parse", "--abbrev-ref", "HEAD")
        )
        if branch == "HEAD":
            branch = None
    if commit is None:
        commit = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    if default_branch is None:
        default_branch = _github_default_branch()
        if default_branch is None:
            remote_head = _git("symbolic-ref", "refs/remotes/origin/HEAD")
            if remote_head and remote_head.startswith("refs/remotes/origin/"):
                default_branch = remote_head.removeprefix("refs/remotes/origin/")
    return {
        name: value
        for name, value in (
            ("git_branch", branch),
            ("git_commit", commit),
            ("default_branch", default_branch),
        )
        if value is not None
    }
