"""Instrumentation patchers: patch application and the sync/async +
buffered/streaming recording paths, driven through the OutputAdapter registry."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace as NS

import httpx
import pytest
from pydantic import BaseModel

from praximetry.instrument import patch as P
from praximetry.instrument.adapters import ADAPTERS
from praximetry.instrument.providers import PROVIDERS, PatchTarget, ProviderSpec
from praximetry.runtime import capture_context, current_run
from praximetry.store import get_store


def test_providers_cover_all_four():
    assert {spec.name for spec in PROVIDERS} == {"openai", "anthropic", "litellm", "gemini"}


# -- _apply_overrides ---------------------------------------------------------


def test_apply_overrides_is_noop_without_an_active_override():
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    assert P._apply_overrides(dict(kwargs), "messages") == kwargs


def test_apply_overrides_swaps_model():
    from praximetry.runtime import override_context

    with override_context(model="gpt-4o-mini"):
        out = P._apply_overrides({"model": "gpt-4o"}, "messages")
    assert out["model"] == "gpt-4o-mini"


def test_apply_overrides_transforms_message_list():
    from praximetry.runtime import override_context

    def transform(messages):
        return [{**m, "content": m["content"].upper()} for m in messages]

    with override_context(prompt_transform=transform):
        out = P._apply_overrides({"messages": [{"role": "user", "content": "hi"}]}, "messages")
    assert out["messages"] == [{"role": "user", "content": "HI"}]


def test_apply_overrides_also_transforms_anthropic_style_system_string():
    from praximetry.runtime import override_context

    def transform(messages):
        return [{**m, "content": m["content"] + "!"} for m in messages]

    with override_context(prompt_transform=transform):
        out = P._apply_overrides(
            {"messages": [{"role": "user", "content": "hi"}], "system": "be brief"}, "messages"
        )
    assert out["system"] == "be brief!"


def test_apply_overrides_leaves_non_message_list_untouched():
    from praximetry.runtime import override_context

    def transform(messages):
        raise AssertionError("should not be called")

    with override_context(prompt_transform=transform):
        out = P._apply_overrides({"messages": "not a list"}, "messages")
    assert out["messages"] == "not a list"


# -- patch application against REAL SDK classes -----------------------------


def test_patchers_apply_to_real_classes():
    assert P.auto_instrument()  # patches whatever is importable
    from anthropic.resources.messages import Messages
    from openai.resources.chat.completions import AsyncCompletions, Completions

    assert getattr(Completions.create, "_praximetry_patched", False)
    assert getattr(AsyncCompletions.create, "_praximetry_patched", False)
    assert getattr(Messages.create, "_praximetry_patched", False)


# -- recording paths via _instrument with a fake `original` -----------------


def _oai_resp(text, tin, tout):
    return NS(
        choices=[NS(message=NS(content=text))], usage=NS(prompt_tokens=tin, completion_tokens=tout)
    )


def _patched_litellm(monkeypatch):
    seen = []

    def completion(model, messages=None, **kwargs):
        seen.append((model, messages, kwargs))
        return _oai_resp("out", 5, 2)

    host = NS(completion=completion)
    spec = ProviderSpec(
        name="litellm",
        owner=lambda: (host, host),
        targets=[PatchTarget("completion", self_less=True)],
        adapter=ADAPTERS["litellm"],
    )
    monkeypatch.setattr(P, "_patched", set())
    assert P._patch(spec)
    return host, seen


def test_litellm_positional_call_is_recorded(monkeypatch):
    host, seen = _patched_litellm(monkeypatch)
    messages = [{"role": "user", "content": "hi"}]

    host.completion("gpt-4o", messages)

    assert seen == [("gpt-4o", messages, {})]
    call = get_store().calls()[0]
    assert call.model == "gpt-4o"
    assert call.messages == messages


def test_litellm_positional_call_accepts_model_override(monkeypatch):
    from praximetry.runtime import override_context

    host, seen = _patched_litellm(monkeypatch)
    messages = [{"role": "user", "content": "hi"}]

    with override_context(model="gpt-4o-mini"):
        host.completion("gpt-4o", messages)

    assert seen == [("gpt-4o-mini", messages, {})]

@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_litellm_preserves_third_positional_argument(monkeypatch, is_async, override):
    from praximetry.runtime import override_context

    seen = []

    def completion(model, messages=None, timeout=None, **kwargs):
        seen.append((model, messages, timeout, kwargs))
        return _oai_resp("out", 5, 2)

    async def acompletion(model, messages=None, timeout=None, **kwargs):
        return completion(model, messages, timeout, **kwargs)

    host = NS(completion=acompletion if is_async else completion)
    spec = ProviderSpec(
        name="litellm",
        owner=lambda: (host, host),
        targets=[PatchTarget("completion", is_async=is_async, self_less=True)],
        adapter=ADAPTERS["litellm"],
    )
    monkeypatch.setattr(P, "_patched", set())
    assert P._patch(spec)
    messages = [{"role": "user", "content": "hi"}]
    transformed = [{"role": "user", "content": "transformed"}]
    with override_context(
        model="gpt-4o-mini" if override else None,
        prompt_transform=(lambda _: transformed) if override else None,
    ):
        result = host.completion("gpt-4o", messages, 30)
        if is_async:
            asyncio.run(result)

    expected_model = "gpt-4o-mini" if override else "gpt-4o"
    expected_messages = transformed if override else messages
    assert seen == [(expected_model, expected_messages, 30, {})]
    call = get_store().calls()[0]
    assert call.model == expected_model
    assert call.messages == expected_messages



def test_sync_buffered_records_normalized_fields():
    create = P._instrument(
        lambda self, **k: _oai_resp("out", 5, 2), "openai", ADAPTERS["openai"], False
    )
    create(None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    c = get_store().calls()[0]
    assert c.provider == "openai" and c.output_tokens == 2 and c.output_text == "out"
    assert c.reasoning_text == ""
    assert c.cost_usd > 0


def test_sync_buffered_flags_unrecognized_model_as_unpriced():
    create = P._instrument(
        lambda self, **k: _oai_resp("out", 5, 2), "openai", ADAPTERS["openai"], False
    )
    create(None, model="mystery-model-9000", messages=[{"role": "user", "content": "hi"}])
    c = get_store().calls()[0]
    assert c.cost_usd == 0.0
    assert c.metadata["unpriced_model"] is True


def test_sync_streaming_records_after_consumption():
    chunks = [
        NS(choices=[NS(delta=NS(content="he"))], usage=None),
        NS(choices=[NS(delta=NS(content="llo"))], usage=None),
        NS(choices=[], usage=NS(prompt_tokens=4, completion_tokens=2)),
    ]
    create = P._instrument(lambda self, **k: iter(chunks), "openai", ADAPTERS["openai"], False)
    stream = create(None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    assert get_store().calls() == []  # nothing recorded until consumed
    text = "".join(c.choices[0].delta.content for c in stream if c.choices)
    assert text == "hello"
    c = get_store().calls()[0]
    assert c.output_text == "hello" and c.input_tokens == 4 and c.output_tokens == 2

def test_sync_stream_close_records_partial_output():

    class Stream:
        def __init__(self):
            self.chunks = iter(
                [
                    NS(choices=[NS(delta=NS(content="partial"))], usage=None),
                ]
            )

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self.chunks)
            except StopIteration:
                raise RuntimeError("iteration failed")

        def close(self):
            self.closed = True

    stream = P._instrument(lambda self, **k: Stream(), "openai", ADAPTERS["openai"], False)(
        None, model="gpt-4o", messages=[], stream=True
    )
    next(stream)
    stream.close()
    assert get_store().calls()[0].output_text == "partial"
    assert stream._stream.closed

def test_sync_stream_iteration_error_records_partial_output_and_error():
    def broken():
        yield NS(choices=[NS(delta=NS(content="partial"))], usage=None)
        raise RuntimeError("iteration failed")

    stream = P._instrument(lambda self, **k: broken(), "openai", ADAPTERS["openai"], False)(
        None, model="gpt-4o", messages=[], stream=True
    )
    next(stream)
    try:
        next(stream)
    except RuntimeError as exc:
        assert str(exc) == "iteration failed"
    else:
        raise AssertionError("iteration error was not re-raised")
    call = get_store().calls()[0]
    assert call.output_text == "partial" and call.error == "iteration failed"


def test_sync_stream_open_error_is_recorded_and_reraised():
    def broken(self, **kwargs):
        raise RuntimeError("open failed")

    create = P._instrument(broken, "openai", ADAPTERS["openai"], False)
    try:
        create(None, model="gpt-4o", messages=[], stream=True)
    except RuntimeError as exc:
        assert str(exc) == "open failed"
    else:
        raise AssertionError("open error was not re-raised")
    assert get_store().calls()[0].error == "open failed"


def test_async_stream_close_records_partial_output_and_delegates_aclose():
    closed = []

    async def agen():
        try:
            yield NS(choices=[NS(delta=NS(content="partial"))], usage=None)
        finally:
            closed.append(True)

    async def orig(self, **kwargs):
        return agen()

    create = P._instrument(orig, "openai", ADAPTERS["openai"], is_async=True)

    async def run():
        stream = await create(None, model="gpt-4o", messages=[], stream=True)
        await anext(stream)
        await stream.aclose()

    asyncio.run(run())
    assert get_store().calls()[0].output_text == "partial"
    assert closed == [True]



def test_async_stream_iteration_error_records_partial_output_and_error():
    async def agen():
        yield NS(choices=[NS(delta=NS(content="partial"))], usage=None)
        raise RuntimeError("async iteration failed")

    async def orig(self, **kwargs):
        return agen()

    create = P._instrument(orig, "openai", ADAPTERS["openai"], is_async=True)

    async def run():
        stream = await create(None, model="gpt-4o", messages=[], stream=True)
        await anext(stream)
        try:
            await anext(stream)
        except RuntimeError as exc:
            assert str(exc) == "async iteration failed"
        else:
            raise AssertionError("iteration error was not re-raised")

    asyncio.run(run())
    call = get_store().calls()[0]
    assert call.output_text == "partial" and call.error == "async iteration failed"


def test_async_stream_open_error_is_recorded_and_reraised():
    async def broken(self, **kwargs):
        raise RuntimeError("async open failed")

    create = P._instrument(broken, "openai", ADAPTERS["openai"], is_async=True)

    async def run():
        try:
            await create(None, model="gpt-4o", messages=[], stream=True)
        except RuntimeError as exc:
            assert str(exc) == "async open failed"
        else:
            raise AssertionError("open error was not re-raised")

    asyncio.run(run())
    assert get_store().calls()[0].error == "async open failed"





def test_openai_streaming_injects_stream_options_for_real_usage():
    # PRA-84: the openai spec is patched with inject_stream_usage=True, so a
    # caller streaming without passing stream_options still gets real usage
    # back instead of the (now-removed) chunk-count fallback.
    seen_kwargs = {}

    def original(self, **k):
        seen_kwargs.update(k)
        return iter(
            [
                NS(choices=[NS(delta=NS(content="hi"))], usage=None),
                NS(choices=[], usage=NS(prompt_tokens=4, completion_tokens=2)),
            ]
        )

    create = P._instrument(original, "openai", ADAPTERS["openai"], False, inject_stream_usage=True)
    stream = create(None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
    list(stream)
    assert seen_kwargs["stream_options"] == {"include_usage": True}
    c = get_store().calls()[0]
    assert c.output_tokens == 2


def test_openai_streaming_injection_does_not_clobber_caller_stream_options():
    seen_kwargs = {}

    def original(self, **k):
        seen_kwargs.update(k)
        return iter([NS(choices=[], usage=None)])

    create = P._instrument(original, "openai", ADAPTERS["openai"], False, inject_stream_usage=True)
    stream = create(
        None,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": False},
    )
    list(stream)
    assert seen_kwargs["stream_options"] == {"include_usage": False}


def test_async_buffered_records():
    async def orig(self, **k):
        return _oai_resp("aout", 6, 3)

    create = P._instrument(orig, "openai", ADAPTERS["openai"], is_async=True)
    asyncio.run(create(None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}]))
    c = get_store().calls()[0]
    assert c.output_text == "aout" and c.output_tokens == 3


def test_async_streaming_records():
    async def agen():
        for ch in [
            NS(choices=[NS(delta=NS(content="A"))], usage=None),
            NS(choices=[NS(delta=NS(content="B"))], usage=None),
            NS(choices=[], usage=NS(prompt_tokens=9, completion_tokens=2)),
        ]:
            yield ch

    async def orig(self, **k):
        return agen()

    create = P._instrument(orig, "openai", ADAPTERS["openai"], is_async=True)

    async def run():
        stream = await create(
            None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        return "".join([c.choices[0].delta.content async for c in stream if c.choices])

    assert asyncio.run(run()) == "AB"
    c = get_store().calls()[0]
    assert c.output_text == "AB" and c.input_tokens == 9 and c.output_tokens == 2


def test_anthropic_streaming_events():
    events = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=12))),
        NS(type="content_block_delta", delta=NS(type="text_delta", text="foo")),
        NS(type="content_block_delta", delta=NS(type="text_delta", text="bar")),
        NS(type="message_delta", usage=NS(output_tokens=5)),
        NS(type="message_stop"),
    ]
    create = P._instrument(
        lambda self, **k: iter(events), "anthropic", ADAPTERS["anthropic"], False
    )
    stream = create(
        None, model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    list(stream)  # consume
    c = get_store().calls()[0]
    assert c.output_text == "foobar" and c.input_tokens == 12 and c.output_tokens == 5


def test_error_path_records_error():
    def boom(self, **k):
        raise RuntimeError("api down")

    create = P._instrument(boom, "openai", ADAPTERS["openai"], False)
    try:
        create(None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    except RuntimeError:
        pass
    c = get_store().calls()[0]
    assert c.error == "api down"


def test_auto_instrument_logs_when_a_provider_patch_raises(monkeypatch, caplog):
    import logging

    spec = PROVIDERS[0]
    monkeypatch.setattr(P, "_patched", set())

    def broken(_spec):
        raise AttributeError("SDK internals moved")

    monkeypatch.setattr(P, "_patch", broken)

    with caplog.at_level(logging.DEBUG, logger="praximetry.instrument.patch"):
        assert P.auto_instrument() == []
    assert any(spec.name in r.message for r in caplog.records)


def test_stream_accumulate_failure_is_logged_not_swallowed_silently(caplog):
    import logging

    def boom_accumulate(chunk, state):
        raise ValueError("bad chunk")

    wrapper = P.SyncStreamWrapper(iter(["a"]), boom_accumulate, lambda state: None)

    with caplog.at_level(logging.DEBUG, logger="praximetry.instrument.wrap"):
        assert list(wrapper) == ["a"]
    assert any("accumulate failed" in r.message for r in caplog.records)


def test_lazy_run_is_shared_by_patched_calls_in_asyncio_gather():
    async def original(self, **kwargs):
        await asyncio.sleep(0)
        return _oai_resp("out", 5, 2)

    create = P._instrument(original, "openai", ADAPTERS["openai"], True)

    async def one(index):
        return await create(
            None, model="gpt-4o", messages=[{"role": "user", "content": str(index)}]
        )

    async def gather():
        return await asyncio.gather(one(0), one(1), one(2))

    asyncio.run(gather())
    calls = get_store().calls()
    assert len(calls) == 3
    assert {call.run_id for call in calls} == {current_run(create=False).id}
    assert all(call.parent_call_id is None for call in calls)
    assert len(get_store().runs()) == 1
    assert capture_context()["current_call_id"] in {call.id for call in calls}


def test_lazy_run_is_shared_by_patched_calls_in_thread_pool():
    barrier = Barrier(3)

    def original(self, **kwargs):
        barrier.wait(timeout=5)
        return _oai_resp("out", 5, 2)

    create = P._instrument(original, "openai", ADAPTERS["openai"], False)

    def one(index):
        return create(
            None, model="gpt-4o", messages=[{"role": "user", "content": str(index)}]
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(one, range(3)))
    assert len(responses) == 3
    calls = get_store().calls()
    assert len(calls) == 3
    assert {call.run_id for call in calls} == {current_run(create=False).id}
    assert all(call.parent_call_id is None for call in calls)
    assert len(get_store().runs()) == 1
    assert capture_context()["current_call_id"] in {call.id for call in calls}


def test_pydantic_response_parse_field_is_not_invoked_by_instrumentation():
    parsed = []

    def parse():
        parsed.append(True)
        return _oai_resp("not the response", 0, 0)

    class LegacyAPIResponse(BaseModel):
        choices: list
        usage: object
        parse: object

    response = LegacyAPIResponse(
        choices=[NS(message=NS(content="real response"))],
        usage=NS(prompt_tokens=5, completion_tokens=2),
        parse=parse,
    )
    create = P._instrument(lambda self, **kwargs: response, "openai", ADAPTERS["openai"], False)
    assert create(None, model="gpt-4o", messages=[]) is response
    assert parsed == []
    call = get_store().calls()[0]
    assert call.output_text == "real response"
    assert call.input_tokens == 5


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
def test_known_sdk_raw_responses_are_recorded_and_returned_intact(provider, legacy, is_async):
    if provider == "openai":
        from openai import OpenAI as Client
        from openai._legacy_response import LegacyAPIResponse
        from openai._models import FinalRequestOptions
        from openai._response import APIResponse
        from openai.types.chat import ChatCompletion as ResponseModel

        model = "gpt-4o"
        payload = {
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "raw reply"},
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    else:
        from anthropic import Anthropic as Client
        # Newer anthropic releases dropped the legacy wrapper module.
        legacy_module = pytest.importorskip("anthropic._legacy_response") if legacy else None
        LegacyAPIResponse = getattr(legacy_module, "LegacyAPIResponse", None)
        from anthropic._models import FinalRequestOptions
        from anthropic._response import APIResponse
        from anthropic.types import Message as ResponseModel

        model = "claude-sonnet-5"
        payload = {
            "id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "raw reply"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

    with Client(api_key="test") as client:
        response_type = LegacyAPIResponse if legacy else APIResponse
        response = response_type(
            raw=httpx.Response(
                200, json=payload, request=httpx.Request("POST", "https://example.test/messages")
            ),
            cast_to=ResponseModel, client=client, stream=False, stream_cls=None,
            options=FinalRequestOptions.construct(method="post", url="/messages"),
        )

        def original(self, **kwargs):
            return response

        async def async_original(self, **kwargs):
            return response

        create = P._instrument(
            async_original if is_async else original, provider, ADAPTERS[provider], is_async
        )
        result = create(None, model=model, messages=[])
        assert (asyncio.run(result) if is_async else result) is response
        parsed = response.parse()
        if provider == "openai":
            assert parsed.choices[0].message.content == "raw reply"
        else:
            assert parsed.content[0].text == "raw reply"
    call = get_store().calls()[0]
    assert call.output_text == "raw reply"
    assert call.input_tokens == 5
    assert call.output_tokens == 2
