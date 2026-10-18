"""Config.from_env(): the fresh_env fixture always calls set_config() directly,
so from_env()'s own env-var parsing has no other coverage."""

from pathlib import Path
import sqlite3

import pytest

from praximetry import init
from praximetry.config import Config
from praximetry.models import Run
from praximetry.store import get_store


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


def test_init_changes_database_after_store_is_built(tmp_path):
    old_store = get_store()
    old_store.save_run(Run(id="old"))
    new_path = tmp_path / "new.db"

    init(db_path=new_path, auto_instrument_sdks=False, cloud=False)
    new_store = get_store()
    new_store.save_run(Run(id="new"))

    assert new_store.db_path == new_path
    assert [run.id for run in new_store.runs()] == ["new"]
    with sqlite3.connect(old_store.db_path) as conn:
        assert conn.execute("SELECT id FROM runs").fetchall() == [("old",)]
