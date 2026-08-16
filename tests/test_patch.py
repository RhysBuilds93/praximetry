"""Instrumentation patchers: patch application and the sync/async +
buffered/streaming recording paths, driven through the OutputAdapter registry."""

import asyncio
from types import SimpleNamespace as NS

from praximetry.instrument import patch as P
from praximetry.instrument.adapters import ADAPTERS
from praximetry.instrument.providers import PROVIDERS
from praximetry.store import get_store


def test_providers_cover_all_four():
    assert {spec.name for spec in PROVIDERS} == {"openai", "anthropic", "litellm", "gemini"}


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


def test_sync_buffered_records_normalized_fields():
    create = P._instrument(
        lambda self, **k: _oai_resp("out", 5, 2), "openai", ADAPTERS["openai"], False
    )
    create(None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    c = get_store().calls()[0]
    assert c.provider == "openai" and c.output_tokens == 2 and c.output_text == "out"
    assert c.reasoning_text == ""
    assert c.cost_usd > 0


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
