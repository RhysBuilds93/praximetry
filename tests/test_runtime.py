"""runtime.py paths not already covered by test_core.py's decorator/store tests."""

from contextlib import contextmanager

from praximetry import config
from praximetry.runtime import (
    capture_context,
    current_run,
    policy_scope,
    record_call,
    restore_context,
    run_context,
    set_policy_hook,
    stage_context,
)
from praximetry.store import get_store


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
