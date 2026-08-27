"""Core: pricing, store, decorators, recording."""

import asyncio
import sqlite3

import praximetry
from praximetry import config, pricing
from praximetry.runtime import STAGE_REGISTRY, record_call, run_context
from praximetry.store import Store, get_store


def test_generated_ids_use_full_uuid_entropy():
    from praximetry.models import Call, Run

    ids = {Call(run_id="r").id for _ in range(50)} | {Run().id for _ in range(50)}
    assert all(len(i) == 32 for i in ids)


def test_pricing_known_and_prefix():
    assert pricing.cost_usd("claude-sonnet-5", 1_000_000, 0) == 3.0
    assert pricing.cost_usd("claude-haiku-4-5-20251001", 0, 1_000_000) == 5.0
    assert pricing.cost_usd("mystery-model", 1000, 1000) == 0.0


def test_cheaper_models_ladder():
    assert "claude-haiku-4-5" in pricing.cheaper_models("claude-sonnet-5")
    assert pricing.cheaper_models("gpt-4o") == ["gpt-4o-mini"]
    assert pricing.cheaper_models("unknown") == []


def test_stage_decorator_registers_and_attributes():
    @praximetry.stage("greet")
    def greet(name):
        record_call(
            provider="fake",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            messages=[{"role": "user", "content": name}],
        )
        return f"hi {name}"

    assert "greet" in STAGE_REGISTRY
    assert greet("bob") == "hi bob"
    calls = get_store().calls()
    assert len(calls) == 1
    assert calls[0].stage == "greet"
    assert calls[0].input_tokens == 10


def test_bare_stage_decorator():
    @praximetry.stage
    def summarize(text):
        return text[:3]

    assert "summarize" in STAGE_REGISTRY
    assert summarize("hello") == "hel"


def test_store_totals_and_summary():
    for i in range(3):
        record_call(
            provider="fake",
            model="gpt-4o",
            stage="s1",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
    t = get_store().totals()
    assert t["n"] == 3 and t["tin"] == 300 and abs(t["cost"] - 0.03) < 1e-9
    summary = get_store().stage_summary()
    assert summary[0]["stage"] == "s1" and summary[0]["n"] == 3


def test_store_runs_reader():
    with run_context(name="triage") as run:
        record_call(provider="fake", model="gpt-4o", stage="classify")
    runs = get_store().runs(limit=1)
    assert runs[0].id == run.id
    assert runs[0].name == "triage"


def test_runs_and_calls_filter_by_project():
    config.get_config().project = "proj-a"
    with run_context(name="a-run"):
        record_call(provider="fake", model="gpt-4o", stage="s1")
    config.get_config().project = "proj-b"
    with run_context(name="b-run"):
        record_call(provider="fake", model="gpt-4o", stage="s1")

    store = get_store()
    runs_a = store.runs(project="proj-a")
    assert [r.name for r in runs_a] == ["a-run"]

    calls_a = store.calls(project="proj-a")
    assert len(calls_a) == 1
    assert calls_a[0].run_id == runs_a[0].id

    assert len(store.runs()) == 2
    assert len(store.calls()) == 2


def test_migrate_adds_missing_column_to_preexisting_table():
    """Simulates a `.praximetry.db` on disk from before `parent_call_id` existed:
    a `calls` table missing that column, created outside of `Store` so
    `CREATE TABLE IF NOT EXISTS` can't paper over it. `Store()` must ALTER it
    in on open, generically, from the `_TABLES` declaration alone."""
    db_path = config.get_config().db_path
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE calls (
            id TEXT PRIMARY KEY, run_id TEXT, stage TEXT, provider TEXT, model TEXT,
            messages TEXT, response_text TEXT,
            input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, latency_ms REAL,
            ts REAL, error TEXT, metadata TEXT
        )"""
    )
    conn.commit()
    conn.close()

    store = Store(db_path)
    cols = {row[1] for row in store._conn().execute("PRAGMA table_info(calls)").fetchall()}
    assert "parent_call_id" in cols

    with run_context(name="post-migration"):
        record_call(provider="fake", model="gpt-4o", stage="plan")


def test_save_run_survives_out_of_order_migrated_columns():
    """A `runs` table from before `name` existed: migration appends `name` last,
    so on-disk column order no longer matches a fresh CREATE TABLE. Named-column
    INSERTs must still land each field in the right column."""
    db_path = config.get_config().db_path
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, project TEXT, started_at REAL, "
        "ended_at REAL, experiment_id TEXT, metadata TEXT)"
    )
    conn.commit()
    conn.close()

    store = Store(db_path)
    with run_context(name="hello") as run:
        run.project = "proj"
    store.save_run(run)

    got = store.runs()[0]
    assert got.name == "hello"
    assert got.project == "proj"


def test_calls_query_logs_when_truncated_at_limit(caplog):
    import logging

    with run_context(name="many"):
        for _ in range(3):
            record_call(provider="fake", model="gpt-4o")

    with caplog.at_level(logging.WARNING, logger="praximetry.store"):
        get_store().calls(limit=2)
    assert any("limit=2" in r.message for r in caplog.records)


def test_parent_call_id_chains_sequential_calls():
    with run_context(name="seq"):
        a = record_call(provider="fake", model="gpt-4o", stage="plan")
        b = record_call(provider="fake", model="gpt-4o", stage="act")
    assert a.parent_call_id is None
    assert b.parent_call_id == a.id


def test_nested_stage_records_full_path():
    @praximetry.stage("extract")
    def extract():
        return record_call(provider="fake", model="gpt-4o")

    @praximetry.stage("summarize")
    def summarize():
        return extract()

    with run_context(name="nested"):
        call = summarize()
    assert call.stage == "summarize>extract"


def test_parent_call_id_auto_links_concurrent_fanout():
    async def sub_agent(name):
        return record_call(provider="fake", model="gpt-4o", stage=name)

    async def dispatch():
        lead = record_call(provider="fake", model="gpt-4o", stage="dispatch")
        a, b = await asyncio.gather(sub_agent("agent_a"), sub_agent("agent_b"))
        return lead, a, b

    with run_context(name="fanout"):
        lead, a, b = asyncio.run(dispatch())

    # asyncio.gather copies the context into each task, so both concurrent
    # sub-calls independently inherit "dispatch" as their parent.
    assert a.parent_call_id == lead.id
    assert b.parent_call_id == lead.id
