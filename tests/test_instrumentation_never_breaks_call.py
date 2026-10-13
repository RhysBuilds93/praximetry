"""Instrumentation failures must never change what the customer's call returns or raises."""

import asyncio
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from anthropic.types import Message, TextBlock, Usage

from praximetry.instrument import patch as P
from praximetry.instrument.adapters import ADAPTERS
from praximetry.models import Call
from praximetry.runtime import policy_scope, set_policy_hook
from praximetry.store import Store, get_store

USER = {"role": "user", "content": "hi"}


def _locked(self: Store, call: Call) -> None:
    raise sqlite3.OperationalError("database is locked")


@pytest.fixture
def locked_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Store, "save_call", _locked)


def _oai_resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _claude_msg() -> Message:
    return Message(
        id="m",
        content=[TextBlock(type="text", text="hello")],
        model="claude-sonnet-5",
        role="assistant",
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def test_store_failure_keeps_openai_response(locked_store: None) -> None:
    resp = _oai_resp("out")
    create = P._instrument(lambda self, **k: resp, "openai", ADAPTERS["openai"], False)
    assert create(None, model="gpt-4o", messages=[USER]) is resp


def test_store_failure_keeps_async_response(locked_store: None) -> None:
    resp = _oai_resp("out")

    async def original(self: object, **k: object) -> SimpleNamespace:
        return resp

    create = P._instrument(original, "openai", ADAPTERS["openai"], True)
    assert asyncio.run(create(None, model="gpt-4o", messages=[USER])) is resp


def test_store_failure_keeps_anthropic_response(locked_store: None) -> None:
    resp = _claude_msg()
    create = P._instrument(lambda self, **k: resp, "anthropic", ADAPTERS["anthropic"], False)
    assert create(None, model="claude-sonnet-5", messages=[USER]) is resp


def test_original_api_error_propagates_when_store_fails(locked_store: None) -> None:
    def original(self: object, **k: object) -> None:
        raise RuntimeError("api down")

    create = P._instrument(original, "openai", ADAPTERS["openai"], False)
    with pytest.raises(RuntimeError, match="api down"):
        create(None, model="gpt-4o", messages=[USER])


def test_unparseable_response_is_returned_untouched() -> None:
    resp = object()
    create = P._instrument(lambda self, **k: resp, "anthropic", ADAPTERS["anthropic"], False)
    assert create(None, model="claude-sonnet-5", messages=[USER]) is resp


def test_malformed_messages_kwarg_still_calls_original() -> None:
    create = P._instrument(
        lambda self, **k: "ok", "anthropic", ADAPTERS["anthropic"], False, messages_key="messages"
    )
    assert create(None, model="claude-sonnet-5", messages=5, system=[object()]) == "ok"


def test_multi_turn_anthropic_with_sdk_blocks_records() -> None:
    reply = _claude_msg()
    create = P._instrument(lambda self, **k: reply, "anthropic", ADAPTERS["anthropic"], False)
    messages = [
        USER,
        {"role": "assistant", "content": reply.content},
        {"role": "user", "content": "b"},
    ]
    assert create(None, model="claude-sonnet-5", messages=messages) is reply
    recorded = get_store().calls()[0]
    assert recorded.messages[1]["content"][0]["text"] == "hello"


def test_store_serialises_bytes_in_messages() -> None:
    get_store().save_call(Call(run_id="r", messages=[{"role": "user", "content": b"bytes"}]))
    assert get_store().calls()[0].messages[0]["content"] == "b'bytes'"


def _failing_done(state: dict[str, object]) -> None:
    raise TypeError("store broke")


def test_stream_on_done_failure_does_not_break_iteration() -> None:
    wrapper = P.SyncStreamWrapper(iter(["a", "b"]), lambda chunk, state: None, _failing_done)
    assert list(wrapper) == ["a", "b"]


def test_stream_on_done_failure_does_not_mask_customer_exception() -> None:
    wrapper = P.SyncStreamWrapper(iter(["a"]), lambda chunk, state: None, _failing_done)
    with pytest.raises(ValueError, match="customer bug"):
        with wrapper:
            raise ValueError("customer bug")


def test_async_stream_on_done_failure_does_not_break_iteration() -> None:
    async def chunks():
        yield "a"
        yield "b"

    wrapper = P.AsyncStreamWrapper(chunks(), lambda chunk, state: None, _failing_done)

    async def consume() -> list[str]:
        return [c async for c in wrapper]

    assert asyncio.run(consume()) == ["a", "b"]


def test_async_stream_on_done_failure_does_not_mask_customer_exception() -> None:
    async def chunks():
        yield "a"

    wrapper = P.AsyncStreamWrapper(chunks(), lambda chunk, state: None, _failing_done)

    async def body() -> None:
        async with wrapper:
            raise ValueError("customer bug")

    with pytest.raises(ValueError, match="customer bug"):
        asyncio.run(body())


def test_on_done_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    wrapper = P.SyncStreamWrapper(iter(["a"]), lambda chunk, state: None, _failing_done)
    with caplog.at_level(logging.WARNING, logger="praximetry.instrument.wrap"):
        list(wrapper)
    assert any("on_done failed" in r.message for r in caplog.records)


@pytest.fixture
def clear_policy_hook() -> Iterator[None]:
    yield
    set_policy_hook(None)


def test_policy_hook_raising_still_runs_stage(clear_policy_hook: None) -> None:
    def hook(stage: str) -> object:
        raise RuntimeError("corrupt overrides.json")

    set_policy_hook(hook)
    ran = False
    with policy_scope("classify"):
        ran = True
    assert ran


def test_policy_hook_enter_raising_still_runs_stage(clear_policy_hook: None) -> None:
    @contextmanager
    def hook(stage: str) -> Iterator[None]:
        raise RuntimeError("bad schema")
        yield

    set_policy_hook(hook)
    ran = False
    with policy_scope("classify"):
        ran = True
    assert ran


def test_stage_body_exception_still_propagates_through_policy_scope(
    clear_policy_hook: None,
) -> None:
    @contextmanager
    def hook(stage: str) -> Iterator[None]:
        yield

    set_policy_hook(hook)
    with pytest.raises(ValueError, match="body"):
        with policy_scope("classify"):
            raise ValueError("body")
