"""Optional evaluation provenance from CI metadata and a real local checkout."""

import json
import subprocess

import pytest

from praximetry.eval.git_info import detect_git_info


@pytest.fixture(autouse=True)
def clear_github_metadata(monkeypatch):
    for name in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME", "GITHUB_SHA", "GITHUB_EVENT_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--initial-branch=main")
    git("config", "user.email", "eval-tests@example.com")
    git("config", "user.name", "Eval Tests")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "initial")
    git("update-ref", "refs/remotes/origin/main", git("rev-parse", "HEAD"))
    git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    git("checkout", "-b", "feature/evals")
    return git


def test_detects_branch_commit_and_remote_default_in_plain_checkout(checkout):
    assert detect_git_info() == {
        "git_branch": "feature/evals",
        "git_commit": checkout("rev-parse", "HEAD"),
        "default_branch": "main",
    }


@pytest.mark.parametrize("head_ref,expected", [("feature/pr", "feature/pr"), ("", "release")])
def test_github_metadata_takes_precedence_over_checkout(
    checkout, monkeypatch, tmp_path, head_ref, expected
):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"repository": {"default_branch": "trunk"}}))
    monkeypatch.setenv("GITHUB_HEAD_REF", head_ref)
    monkeypatch.setenv("GITHUB_REF_NAME", "release")
    monkeypatch.setenv("GITHUB_SHA", "github-sha")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))

    assert detect_git_info() == {
        "git_branch": expected,
        "git_commit": "github-sha",
        "default_branch": "trunk",
    }


@pytest.mark.parametrize(
    "event_payload",
    [b"not JSON", b"null", b"[]", b'{"repository": null}', b'{"repository": {}}', b"\xff"],
)
def test_invalid_github_event_falls_back_to_git(checkout, monkeypatch, tmp_path, event_payload):
    event = tmp_path / "event.json"
    event.write_bytes(event_payload)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/pr")

    assert detect_git_info() == {
        "git_branch": "feature/pr",
        "git_commit": checkout("rev-parse", "HEAD"),
        "default_branch": "main",
    }


def test_missing_github_event_is_optional(checkout, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(tmp_path / "missing.json"))

    assert detect_git_info()["default_branch"] == "main"


def test_detached_head_omits_branch_but_preserves_known_commit(checkout):
    commit = checkout("rev-parse", "HEAD")
    checkout("checkout", "--detach", commit)

    assert detect_git_info() == {"git_commit": commit, "default_branch": "main"}


def test_unconfigured_remote_default_is_omitted(checkout):
    checkout("symbolic-ref", "--delete", "refs/remotes/origin/HEAD")

    assert detect_git_info() == {
        "git_branch": "feature/evals",
        "git_commit": checkout("rev-parse", "HEAD"),
    }


def test_not_a_checkout_is_optional(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert detect_git_info() == {}


def test_missing_git_keeps_any_github_metadata(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature/pr")
    monkeypatch.setenv("GITHUB_SHA", "github-sha")

    assert detect_git_info() == {"git_branch": "feature/pr", "git_commit": "github-sha"}


def test_explicit_overrides_win_over_github_and_git(checkout, monkeypatch):
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/pr")
    monkeypatch.setenv("GITHUB_SHA", "github-sha")

    assert detect_git_info(branch="manual", commit="manual-sha", default_branch="trunk") == {
        "git_branch": "manual",
        "git_commit": "manual-sha",
        "default_branch": "trunk",
    }
