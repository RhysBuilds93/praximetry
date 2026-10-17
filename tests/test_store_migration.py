"""Store opens against pre-existing on-disk schemas."""

import multiprocessing
import sqlite3
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from praximetry.models import Run

from praximetry.store import Store, _enable_wal

_OLD_CALLS = (
    "CREATE TABLE calls (id TEXT PRIMARY KEY, run_id TEXT, stage TEXT, provider TEXT, "
    "model TEXT, messages TEXT, response_text TEXT, input_tokens INTEGER, "
    "output_tokens INTEGER, cost_usd REAL, latency_ms REAL, ts REAL, error TEXT, metadata TEXT)"
)


def _open_when_ready(db_path: Path, start_at: float) -> str:
    time.sleep(max(0.0, start_at - time.time()))
    Store(db_path)
    return "ok"


def test_rows_written_before_migrated_columns_are_readable(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute(_OLD_CALLS)
    conn.execute(
        "INSERT INTO calls VALUES ('c1', 'r1', 's', 'openai', 'gpt-4o', '[]', 'legacy text', "
        "5, 7, 0.1, 12.0, 100.0, NULL, '{}')"
    )
    conn.execute("CREATE TABLE runs (id TEXT PRIMARY KEY, started_at REAL)")
    conn.execute("INSERT INTO runs (id, started_at) VALUES ('r1', 100.0)")
    conn.execute("CREATE TABLE experiments (id TEXT PRIMARY KEY, created_at REAL)")
    conn.execute("INSERT INTO experiments (id, created_at) VALUES ('e1', 100.0)")
    conn.execute("CREATE TABLE eval_results (id TEXT PRIMARY KEY, ts REAL)")
    conn.execute("INSERT INTO eval_results (id, ts) VALUES ('v1', 100.0)")
    conn.commit()
    conn.close()

    store = Store(db_path)

    (call,) = store.calls()
    assert call.id == "c1"
    assert call.input_tokens == 5
    assert call.output_text == ""
    assert call.reasoning_text == ""
    assert call.content_parts == []
    assert [r.id for r in store.runs()] == ["r1"]
    assert store.runs()[0].project == "default"
    assert [e.id for e in store.experiments()] == ["e1"]
    assert not store.experiments()[0].baseline
    assert [r.id for r in store.eval_results()] == ["v1"]
    assert not store.eval_results()[0].passed


def test_index_on_late_added_column_does_not_block_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE calls (id TEXT PRIMARY KEY, ts REAL)")
    conn.commit()
    conn.close()

    store = Store(db_path)

    cols = {row[1] for row in store._conn().execute("PRAGMA table_info(calls)").fetchall()}
    assert {"run_id", "stage"} <= cols


def test_concurrent_opens_of_an_unmigrated_db(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    with context.Pool(6) as pool:
        for trial in range(3):
            db_path = tmp_path / f"old{trial}.db"
            conn = sqlite3.connect(db_path)
            conn.execute(_OLD_CALLS)
            conn.commit()
            conn.close()

            start_at = time.time() + 1.5
            results = [
                pool.apply_async(_open_when_ready, (db_path, start_at)) for _ in range(6)
            ]
            assert [r.get(timeout=60) for r in results] == ["ok"] * 6


def test_in_memory_store_opens() -> None:
    assert Store(":memory:").totals()["n"] == 0


def test_relative_db_path_is_fixed_when_store_is_created(tmp_path: Path, monkeypatch) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(origin)
    store = Store(Path("data") / "traces.db")
    database = origin / "data" / "traces.db"

    monkeypatch.chdir(other)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(store.save_run, Run(id="from-worker")).result()

    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT id FROM runs").fetchall() == [("from-worker",)]
    assert not (other / "data" / "traces.db").exists()


def test_enable_wal_gives_up_while_another_connection_holds_the_lock(tmp_path: Path) -> None:
    db_path = tmp_path / "locked.db"
    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    contender = sqlite3.connect(db_path, timeout=0)

    started = time.monotonic()
    _enable_wal(contender, timeout=0.2)

    assert time.monotonic() - started < 5
    holder.execute("ROLLBACK")
    assert contender.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
