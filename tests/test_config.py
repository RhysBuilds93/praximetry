"""Config.from_env(): the fresh_env fixture always calls set_config() directly,
so from_env()'s own env-var parsing has no other coverage."""

from pathlib import Path

from praximetry.config import Config


def test_from_env_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("PRAXIMETRY_PROJECT", raising=False)
    monkeypatch.delenv("PRAXIMETRY_DB", raising=False)
    monkeypatch.delenv("PRAXIMETRY_DISABLED", raising=False)

    cfg = Config.from_env()

    assert cfg.project == "default"
    assert cfg.db_path == Path(".praximetry") / "praximetry.db"
    assert cfg.enabled is True


def test_from_env_reads_project_db_and_disabled(monkeypatch):
    monkeypatch.setenv("PRAXIMETRY_PROJECT", "my-project")
    monkeypatch.setenv("PRAXIMETRY_DB", "/tmp/custom.db")
    monkeypatch.setenv("PRAXIMETRY_DISABLED", "1")

    cfg = Config.from_env()

    assert cfg.project == "my-project"
    assert cfg.db_path == Path("/tmp/custom.db")
    assert cfg.enabled is False
