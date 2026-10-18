"""runtime.py paths not already covered by test_core.py's decorator/store tests."""

import asyncio
import gc
import json
import logging
import sqlite3
import weakref
from contextvars import Context
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event

import pytest
from pydantic import ValidationError

from praximetry import config
from praximetry.runtime import (
    capture_context,
    current_run,
    get_overrides,
    override_context,
    policy_scope,
    record_call,
    reset_runtime,
    restore_context,
    run_context,
    set_policy_hook,
    stage_context,
)
from praximetry.store import Store, get_store, reset_store


def test_current_run_create_false_does_not_create_a_run():
    assert current_run(create=False) is None
    assert get_store().runs() == []


def test_record_call_noop_when_disabled():
    config.get_config().enabled = False
    call = record_call(provider="fake", model="gpt-4o")
    assert call.model == "gpt-4o"
    assert get_store().calls() == []


def test_record_call_accepts_a_prebuilt_call():
    from praximetry.models import Call

    with run_context(name="prebuilt") as run:
        call = Call(run_id=run.id, provider="fake", model="gpt-4o")
        recorded = record_call(call)
    assert recorded is call
    assert get_store().calls()[0].id == call.id


def test_capture_and_restore_context_round_trip():
    with run_context(name="outer") as run:
        with stage_context("plan"):
            record_call(provider="fake", model="gpt-4o")
            ctx = capture_context()

    assert ctx["run_id"] == run.id
    assert ctx["stage_stack"] == ("plan",)
    assert ctx["current_call_id"] is not None

    with restore_context(ctx):
        second = record_call(provider="fake", model="gpt-4o")
    assert second.run_id == run.id
    assert second.parent_call_id == ctx["current_call_id"]
    assert second.stage == "plan"


def test_restore_context_is_a_noop_without_a_run_id():
    with restore_context(None):
        assert current_run(create=False) is None
    with restore_context({}):
        assert current_run(create=False) is None


def test_policy_scope_wraps_the_stage_with_the_registered_hook():
    events = []

    @contextmanager
    def hook(stage):
        events.append(f"enter:{stage}")
        yield
        events.append(f"exit:{stage}")

    set_policy_hook(hook)
    try:
        with policy_scope("classify"):
            events.append("inside")
    finally:
        set_policy_hook(None)

    assert events == ["enter:classify", "inside", "exit:classify"]


def test_policy_scope_is_a_noop_without_a_hook():
    with policy_scope("classify"):
        pass


@pytest.mark.parametrize("explicit_run", [False, True])
def test_capture_restores_overrides_across_a_thread_hop(fake_llm, explicit_run):
    def transform(messages):
        return [{**message, "content": "transformed"} for message in messages]

    def worker(ctx):
        with override_context(model="worker-default"):
            with restore_context(ctx):
                fake_llm.chat("baseline", [{"role": "user", "content": "original"}])
                assert get_overrides()["prompt_transform"] is transform
            assert get_overrides()["model"] == "worker-default"

    @contextmanager
    def optional_run():
        if explicit_run:
            with run_context("trial"):
                yield
        else:
            yield

    with optional_run(), override_context(model="m-override", prompt_transform=transform):
        ctx = capture_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(worker, ctx).result()
        assert get_overrides()["model"] == "m-override"
    assert get_overrides() is None
    call = get_store().calls()[0]
    assert call.model == "m-override"
    assert call.messages == [{"role": "user", "content": "transformed"}]


def test_restored_worker_updates_the_shared_run_and_last_call_bridge():
    def worker(ctx):
        with restore_context(ctx):
            current_run().metadata["worker"] = "finished"
            return record_call(provider="fake", model="gpt-4o")

    with run_context("parent") as run:
        first = record_call(provider="fake", model="gpt-4o")
        with stage_context("dispatch"):
            ctx = capture_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                second = pool.submit(worker, ctx).result()
        assert second.run_id == run.id
        assert second.parent_call_id == first.id
        assert second.stage == "dispatch"
        assert run.last_call_id == second.id
        assert capture_context()["current_call_id"] == second.id
        assert run.metadata == {"worker": "finished"}
    saved_run = get_store().runs()[0]
    assert saved_run.ended_at == run.ended_at
    assert saved_run.metadata == {"worker": "finished"}

def test_restored_workers_keep_independent_capture_ancestry():
    both_restored = Barrier(2)
    recorded = Event()

    def one(ctx):
        with restore_context(ctx):
            both_restored.wait(timeout=5)
            child = record_call(provider="fake")
            recorded.set()
            return child

    def two(ctx):
        with restore_context(ctx):
            both_restored.wait(timeout=5)
            assert recorded.wait(timeout=5)
            snapshot = capture_context()
            return snapshot, record_call(provider="fake")

    with run_context("parent"):
        parent = record_call(provider="fake")
        ctx = capture_context()
        with ThreadPoolExecutor(max_workers=2) as pool:
            one_result = pool.submit(one, ctx)
            two_result = pool.submit(two, ctx)
            first = one_result.result()
            snapshot, second = two_result.result()
        assert snapshot["current_call_id"] == parent.id
        assert first.parent_call_id == second.parent_call_id == parent.id
        assert capture_context()["current_call_id"] == second.id


def test_capture_during_reconfiguration_does_not_roll_over_the_default_run():
    first = record_call(provider="fake")
    original_project = config.get_config().project
    config.get_config().project = "replacement"
    snapshot = capture_context()
    config.get_config().project = original_project

    second = record_call(provider="fake")
    assert snapshot["run_id"] == first.run_id
    assert snapshot["current_call_id"] == first.id
    assert second.run_id == first.run_id
    assert second.parent_call_id == first.id


@pytest.mark.parametrize("explicit_run", [False, True])
def test_interleaved_tasks_capture_their_own_branch_parent(explicit_run):
    async def exercise():
        first_recorded = asyncio.Event()
        second_recorded = asyncio.Event()

        async def one():
            first = record_call(provider="fake")
            first_recorded.set()
            await second_recorded.wait()
            return first, capture_context()

        async def two():
            await first_recorded.wait()
            second = record_call(provider="fake")
            second_recorded.set()
            return second, capture_context()

        return await asyncio.gather(one(), two())

    @contextmanager
    def optional_run():
        if explicit_run:
            with run_context("shared"):
                yield
        else:
            yield

    with optional_run():
        results = asyncio.run(exercise())
        for parent, ctx in results:
            with restore_context(ctx):
                child = record_call(provider="fake")
            assert ctx["current_call_id"] == parent.id
            assert child.parent_call_id == parent.id
            assert child.run_id == parent.run_id


def test_capture_bridges_awaited_children_without_crossing_sibling_branches():
    async def exercise():
        parent = record_call(provider="fake")
        child = await asyncio.create_task(async_record())
        assert child.parent_call_id == parent.id
        assert capture_context()["current_call_id"] == child.id

    async def async_record():
        return record_call(provider="fake")

    asyncio.run(exercise())



def test_restored_late_worker_does_not_mutate_the_finished_parent_run():
    def worker(ctx):
        with restore_context(ctx):
            current_run().metadata["late-worker"] = True
            return record_call(provider="fake", model="gpt-4o")

    with run_context("parent") as run:
        run.metadata["parent"] = "finished"
        first = record_call(provider="fake", model="gpt-4o")
        ctx = capture_context()
    ended_at = run.ended_at
    assert ended_at is not None

    with ThreadPoolExecutor(max_workers=1) as pool:
        late_call = pool.submit(worker, ctx).result()

    assert late_call.run_id == run.id
    assert late_call.parent_call_id == first.id
    assert run.last_call_id == first.id
    assert run.metadata == {"parent": "finished"}
    assert run.ended_at == ended_at
    saved_run = get_store().runs()[0]
    assert saved_run.metadata == {"parent": "finished"}
    assert saved_run.ended_at == ended_at


def test_restore_deserialized_context_with_an_unknown_run():
    ctx = json.loads(json.dumps({
        "run_id": "another-process",
        "run_project": "remote-project",
        "stage_stack": ["dispatch"],
        "current_call_id": "parent-call",
        "overrides": {"model": "m-override"},
    }))
    with restore_context(ctx):
        assert current_run().project == "remote-project"
        assert capture_context()["current_call_id"] == "parent-call"
        assert get_overrides() == {"model": "m-override"}
        call = record_call(provider="fake")
    assert call.run_id == "another-process"
    assert call.parent_call_id == "parent-call"
    assert call.stage == "dispatch"
    assert current_run(create=False) is None
    assert get_overrides() is None


def test_explicit_runs_remain_isolated_in_async_tasks():
    default = record_call(provider="fake")

    async def one(name):
        with run_context(name) as run:
            await asyncio.sleep(0)
            first = record_call(provider="fake")
            await asyncio.sleep(0)
            second = record_call(provider="fake")
            return run.id, first, second

    async def gather():
        return await asyncio.gather(one("one"), one("two"), one("three"))

    results = asyncio.run(gather())
    assert len({run_id for run_id, _, _ in results}) == 3
    for run_id, first, second in results:
        assert run_id != default.run_id
        assert first.run_id == second.run_id == run_id
        assert first.parent_call_id is None
        assert second.parent_call_id == first.id
    assert current_run().id == default.run_id


def test_explicit_runs_remain_isolated_in_thread_pool_workers():
    default = record_call(provider="fake")
    barrier = Barrier(3)

    def one(name):
        with run_context(name) as run:
            barrier.wait(timeout=5)
            call = record_call(provider="fake")
            assert current_run() is run
            return call

    with ThreadPoolExecutor(max_workers=3) as pool:
        calls = list(pool.map(one, ["one", "two", "three"]))
    assert len({call.run_id for call in calls}) == 3
    assert all(call.run_id != default.run_id for call in calls)
    assert all(call.parent_call_id is None for call in calls)
    assert current_run().id == default.run_id


def test_context_snapshot_and_live_lookup_do_not_keep_a_run_alive():
    with restore_context({"run_id": "temporary-run", "run_project": "test"}):
        run_ref = weakref.ref(current_run())
        ctx = capture_context()
        assert ctx["run_id"] == "temporary-run"
    gc.collect()
    assert run_ref() is None
    with restore_context(ctx):
        assert current_run().id == "temporary-run"


def test_reset_runtime_discards_the_process_run_for_a_new_database(tmp_path):
    first = record_call(provider="fake")
    reset_runtime()
    config.get_config().db_path = tmp_path / "another.db"
    config.get_config().project = "another-project"
    reset_store()
    assert current_run(create=False) is None
    second = record_call(provider="fake")
    assert second.run_id != first.run_id
    assert second.parent_call_id is None
    assert [run.project for run in get_store().runs()] == ["another-project"]
    assert [call.id for call in get_store().calls()] == [second.id]


@pytest.mark.parametrize("reconfigure", ["set_config", "init"])
def test_lazy_run_rolls_over_after_project_reconfiguration(reconfigure):
    first = record_call(provider="fake")
    db_path = config.get_config().db_path
    if reconfigure == "init":
        from praximetry import init

        init(project="another-project", db_path=db_path, auto_instrument_sdks=False, cloud=False)
    else:
        config.set_config(config.Config(project="another-project", db_path=db_path))

    second = record_call(provider="fake")
    assert second.run_id != first.run_id
    assert second.parent_call_id is None
    runs = {run.id: run for run in get_store().runs()}
    assert runs[first.run_id].project == "test"
    assert runs[second.run_id].project == "another-project"

@pytest.mark.parametrize("rollover", ["project", "store"])
def test_existing_contexts_clear_parents_after_default_run_rollover(tmp_path, rollover):
    contexts = [Context(), Context()]
    old_calls = [ctx.run(record_call, provider="fake") for ctx in contexts]
    assert old_calls[0].run_id == old_calls[1].run_id

    if rollover == "project":
        config.get_config().project = "replacement-project"
    else:
        config.get_config().db_path = tmp_path / "replacement.db"
        reset_store()

    new_calls = [ctx.run(record_call, provider="fake") for ctx in contexts]
    assert new_calls[0].run_id == new_calls[1].run_id
    assert new_calls[0].run_id != old_calls[0].run_id
    assert all(call.parent_call_id is None for call in new_calls)



def test_lazy_run_rolls_over_when_the_configured_store_is_reopened(tmp_path):
    first = record_call(provider="fake")
    first_store = get_store()
    config.set_config(config.Config(project="test", db_path=tmp_path / "replacement.db"))
    reset_store()

    second = record_call(provider="fake")
    assert second.run_id != first.run_id
    assert second.parent_call_id is None
    assert [run.id for run in get_store().runs()] == [second.run_id]
    assert [call.id for call in get_store().calls()] == [second.id]
    assert [run.id for run in first_store.runs()] == [first.run_id]
    assert [call.id for call in first_store.calls()] == [first.id]


@pytest.mark.parametrize("fail_on", [{1}, {2}, {1, 2}])
@pytest.mark.parametrize("body_raises", [False, True])
def test_run_context_store_failures_never_block_or_mask_the_body(
    monkeypatch, caplog, fail_on, body_raises
):
    save_run = Store.save_run
    attempts = 0

    def unreliable_save(self, run):
        nonlocal attempts
        attempts += 1
        if attempts in fail_on:
            raise sqlite3.OperationalError("run database is locked")
        save_run(self, run)

    with run_context("outer") as outer:
        monkeypatch.setattr(Store, "save_run", unreliable_save)
        with caplog.at_level(logging.WARNING, logger="praximetry.runtime"):
            if body_raises:
                with pytest.raises(ValueError, match="customer failure"):
                    with run_context("inner") as inner:
                        assert current_run() is inner
                        raise ValueError("customer failure")
            else:
                with run_context("inner") as inner:
                    assert current_run() is inner
        assert inner.ended_at is not None
        assert current_run() is outer
        assert attempts == 2
        failures = [record for record in caplog.records if "save run" in record.message]
        assert len(failures) == len(fail_on)
        assert all(record.exc_info for record in failures)
        monkeypatch.setattr(Store, "save_run", save_run)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("kwargs", [{"input_tokens": "invalid"}, {"response_text": "typo"}])
def test_record_call_validates_manual_input_even_when_disabled(enabled, kwargs):
    config.get_config().enabled = enabled
    with pytest.raises(ValidationError):
        record_call(**kwargs)
    assert get_store().calls() == []
