"""Core: pricing, store, decorators, recording."""

import asyncio
import inspect
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Barrier

import pytest
from pydantic import ValidationError

import praximetry
from praximetry import config, pricing
from praximetry.models import Call, Run
from praximetry.runtime import STAGE_REGISTRY, record_call, run_context
from praximetry.store import Store, get_store, reset_store


def test_generated_ids_use_full_uuid_entropy():
    from praximetry.models import Call, Run

    ids = {Call(run_id="r").id for _ in range(50)} | {Run().id for _ in range(50)}
    assert all(len(i) == 32 for i in ids)


def test_call_rejects_unexpected_field():
    import pytest
    from pydantic import ValidationError

    from praximetry.models import Call

    with pytest.raises(ValidationError):
        Call(run_id="r", response_text="typo for output_text")


def test_hook_bound_restores_previous_on_exit():
    from praximetry._hooks import Hook

    h = Hook()
    assert h.fn is None
    h.set(lambda: "outer")
    with h.bound(lambda: "inner"):
        assert h.fn() == "inner"
    assert h.fn() == "outer"


def test_hook_bound_is_isolated_between_overlapping_threads():
    from threading import Event, Thread

    from praximetry._hooks import Hook

    hook = Hook()

    def first_fn():
        return "first"

    def second_fn():
        return "second"

    first_entered = Event()
    second_entered = Event()
    first_exited = Event()
    observed = []

    def first():
        with hook.bound(first_fn):
            first_entered.set()
            assert second_entered.wait(timeout=2)
        first_exited.set()

    def second():
        assert first_entered.wait(timeout=2)
        with hook.bound(second_fn):
            second_entered.set()
            assert first_exited.wait(timeout=2)
            observed.append(hook.fn)

    threads = [Thread(target=first), Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert observed == [second_fn]
    assert hook.fn is None


def test_pricing_known_and_prefix():
    assert pricing.cost_usd("claude-sonnet-5", 1_000_000, 0) == 3.0
    assert pricing.cost_usd("claude-haiku-4-5-20251001", 0, 1_000_000) == 5.0
    assert pricing.cost_usd("mystery-model", 1000, 1000) == 0.0


def test_is_unpriced_distinguishes_known_from_unknown_models():
    assert pricing.is_unpriced("claude-sonnet-5") is False
    assert pricing.is_unpriced("claude-haiku-4-5-20251001") is False  # prefix match
    assert pricing.is_unpriced("mystery-model") is True


def test_is_unpriced_true_for_a_registered_zero_cost_model():
    pricing.register_pricing("internal-free-model", 0.0, 0.0)
    try:
        assert pricing.cost_usd("internal-free-model", 1000, 1000) == 0.0
        assert pricing.is_unpriced("internal-free-model") is False
    finally:
        del pricing.PRICING["internal-free-model"]


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


@pytest.mark.parametrize("kind", ["sync", "async", "generator", "async_generator"])
@pytest.mark.parametrize("different_attribute", ["__module__", "__qualname__"])
def test_stage_name_collisions_warn_but_keep_the_new_callable(caplog, kind, different_attribute):
    def original():
        return "original"

    def sync():
        return "replacement"

    async def async_():
        return "replacement"

    def generator():
        yield "replacement"

    async def async_generator():
        yield "replacement"

    replacement = {
        "sync": sync, "async": async_, "generator": generator, "async_generator": async_generator
    }[kind]
    original.__module__ = replacement.__module__ = "same_module"
    original.__qualname__ = replacement.__qualname__ = "run"
    setattr(replacement, different_attribute, "different_origin")
    praximetry.stage("collision")(original)
    with caplog.at_level(logging.WARNING, logger="praximetry.instrument.decorators"):
        praximetry.stage("collision")(replacement)

    assert len(caplog.records) == 1
    assert "collision" in caplog.text
    assert "different_origin" in caplog.text
    assert "same_module.run" in caplog.text
    result = STAGE_REGISTRY["collision"]()
    if inspect.isasyncgen(result):
        async def consume():
            return [item async for item in result]
        assert asyncio.run(consume()) == ["replacement"]
    elif inspect.isawaitable(result):
        assert asyncio.run(result) == "replacement"
    elif inspect.isgenerator(result):
        assert list(result) == ["replacement"]
    else:
        assert result == "replacement"


def test_stage_reregistration_from_the_same_module_and_qualname_does_not_warn(caplog):
    def make_run(value):
        def run():
            return value
        return run

    praximetry.stage("reloadable")(make_run("before reload"))
    with caplog.at_level(logging.WARNING, logger="praximetry.instrument.decorators"):
        praximetry.stage("reloadable")(make_run("after reload"))
    assert caplog.records == []
    assert STAGE_REGISTRY["reloadable"]() == "after reload"


@pytest.mark.parametrize("empty_name", [False, True])
@pytest.mark.parametrize("callable_kind", ["partial", "instance"])
def test_stage_uses_the_callable_type_name_without_a_function_name(empty_name, callable_kind):
    def execute():
        record_call(provider="fake")
        return "result"

    class CallableStage:
        def __call__(self):
            return execute()

    fn = partial(execute) if callable_kind == "partial" else CallableStage()
    expected = "partial" if callable_kind == "partial" else "CallableStage"
    wrapped = praximetry.stage("")(fn) if empty_name else praximetry.stage(fn)
    assert wrapped() == "result"
    assert STAGE_REGISTRY[expected]() == "result"
    assert {call.stage for call in get_store().calls()} == {expected}


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


@pytest.mark.parametrize("table", ["runs", "calls"])
@pytest.mark.parametrize("count", [0, 2, 3])
def test_truncation_warns_only_when_rows_are_omitted(table, count, caplog):
    store = get_store()
    for i in range(count):
        if table == "runs":
            store.save_run(Run(id=str(i), started_at=i))
        else:
            store.save_call(Call(id=str(i), run_id="r", ts=i))

    with caplog.at_level(logging.WARNING, logger="praximetry.store"):
        rows = getattr(store, table)(limit=2)

    assert [row.id for row in rows] == [str(i) for i in reversed(range(count))][:2]
    assert bool(caplog.records) == (count > 2)


@pytest.mark.parametrize("reset", [False, True])
def test_close_releases_connections_from_all_threads(reset):
    store = get_store()
    main_connection = store._conn()
    ready = Barrier(4)
    closed = Barrier(4)

    def worker():
        connection = store._conn()
        ready.wait(timeout=10)
        closed.wait(timeout=10)
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        with pytest.raises(sqlite3.ProgrammingError):
            store.totals()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker) for _ in range(3)]
        ready.wait(timeout=10)
        try:
            if reset:
                reset_store()
            else:
                store.close()
                store.close()
        finally:
            closed.wait(timeout=10)
        for future in futures:
            future.result(timeout=10)

    with pytest.raises(sqlite3.ProgrammingError):
        main_connection.execute("SELECT 1")
    if reset:
        assert get_store() is not store
        assert get_store().totals()["n"] == 0


@pytest.mark.parametrize("bad_row", [{"run_id": "r", "input_tokens": "bad"}, {}])
def test_ingest_rolls_back_every_row_on_invalid_call(bad_row):
    store = get_store()
    store.save_run(Run(id="existing", name="original"))
    payload = {
        "runs": [{"id": "existing", "name": "replacement"}, {"id": "new"}],
        "calls": [{"id": "valid", "run_id": "new"}, bad_row],
    }

    with pytest.raises(ValidationError):
        store.ingest(payload)

    assert [(run.id, run.name) for run in store.runs()] == [("existing", "original")]
    assert store.calls() == []


def test_calls_query_logs_when_truncated_at_limit(caplog):
    import logging

    with run_context(name="many"):
        for _ in range(3):
            record_call(provider="fake", model="gpt-4o")

    with caplog.at_level(logging.WARNING, logger="praximetry.store"):
        get_store().calls(limit=2)
    assert any("limit=2" in r.message for r in caplog.records)


def test_ingest_writes_runs_calls_experiments_eval_results():
    from praximetry.models import Call, EvalResult, Experiment, Run

    payload = {
        "runs": [Run(name="remote-run").model_dump()],
        "calls": [Call(run_id="r1", provider="fake", model="gpt-4o").model_dump()],
        "experiments": [Experiment(name="v1", stage="classify").model_dump()],
        "eval_results": [EvalResult(example_id="ex1", scorer="exact").model_dump()],
    }

    counts = get_store().ingest(payload)

    assert counts == {"runs": 1, "calls": 1, "experiments": 1, "eval_results": 1}
    assert get_store().runs()[0].name == "remote-run"
    assert get_store().calls()[0].model == "gpt-4o"
    assert get_store().experiments()[0].name == "v1"
    assert get_store().eval_results()[0].example_id == "ex1"


def test_ingest_empty_payload_returns_zero_counts():
    assert get_store().ingest({}) == {"runs": 0, "calls": 0, "experiments": 0, "eval_results": 0}


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
