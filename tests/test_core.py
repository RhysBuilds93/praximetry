"""Core: pricing, store, decorators, recording."""
import asyncio
import sqlite3

import pytest

import praximetry
from praximetry import config, pricing
from praximetry.models import Call, EvalResult, Experiment, Run
from praximetry.runtime import STAGE_REGISTRY, record_call, run_context
from praximetry.store import Store, get_store


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
        record_call(provider="fake", model="gpt-4o", input_tokens=10, output_tokens=5,
                    cost_usd=0.001, messages=[{"role": "user", "content": name}])
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
        record_call(provider="fake", model="gpt-4o", stage="s1",
                    input_tokens=100, output_tokens=50, cost_usd=0.01)
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


def test_parent_call_id_chains_sequential_calls():
    with run_context(name="seq"):
        a = record_call(provider="fake", model="gpt-4o", stage="plan")
        b = record_call(provider="fake", model="gpt-4o", stage="act")
    assert a.parent_call_id is None
    assert b.parent_call_id == a.id


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


def test_save_and_read_run_round_trips_all_fields():
    run = Run(
        id="run-1", project="proj", name="my-run", started_at=1.0, ended_at=2.0,
        experiment_id="exp-1", metadata={"k": "v", "n": 1},
    )
    store = get_store()
    store.save_run(run)
    (read_back,) = [r for r in store.runs(limit=10) if r.id == "run-1"]
    assert read_back == run


def test_save_and_read_call_round_trips_all_fields():
    call = Call(
        id="call-1", run_id="run-1", parent_call_id="call-0", stage="plan",
        provider="fake", model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        response_text="hello", input_tokens=10, output_tokens=5, cost_usd=0.01,
        latency_ms=123.4, ts=1.0, error="boom", metadata={"k": "v", "n": 1},
    )
    store = get_store()
    store.save_call(call)
    (read_back,) = [c for c in store.calls(run_id="run-1") if c.id == "call-1"]
    assert read_back == call


@pytest.mark.parametrize("baseline", [True, False])
def test_save_and_read_experiment_round_trips_all_fields(baseline):
    exp = Experiment(
        id=f"exp-{baseline}", name="my-exp", stage="plan",
        variant={"model": "gpt-4o-mini", "compact_prompts": True},
        created_at=1.0, quality=0.9, pass_rate=0.8, cost_usd=0.05,
        input_tokens=100, output_tokens=50, baseline=baseline,
    )
    store = get_store()
    store.save_experiment(exp)
    (read_back,) = [e for e in store.experiments(stage="plan") if e.id == exp.id]
    assert read_back == exp


@pytest.mark.parametrize("passed", [True, False])
def test_save_and_read_eval_result_round_trips_all_fields(passed):
    r = EvalResult(
        id=f"eval-{passed}", experiment_id="exp-1", run_id="run-1", stage="plan",
        example_id="ex-1", scorer="exact", score=0.75, passed=passed,
        detail="some detail", ts=1.0,
    )
    store = get_store()
    store.save_eval_result(r)
    (read_back,) = [x for x in store.eval_results(experiment_id="exp-1") if x.id == r.id]
    assert read_back == r
