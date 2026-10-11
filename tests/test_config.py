"""Config.from_env(): the fresh_env fixture always calls set_config() directly,
so from_env()'s own env-var parsing has no other coverage."""

from pathlib import Path

import pytest

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


@pytest.mark.parametrize("value", ["0", "false", "No", "OFF", ""])
def test_disabled_false_values_keep_recording_enabled(monkeypatch, value):
    monkeypatch.setenv("PRAXIMETRY_DISABLED", value)

    assert Config.from_env().enabled is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
def test_disabled_truthy_values_disable_recording(monkeypatch, value):
    monkeypatch.setenv("PRAXIMETRY_DISABLED", value)

    assert Config.from_env().enabled is False
