"""Full-stack verification against REAL SDK clients.

Uses httpx.MockTransport so the genuine OpenAI/Anthropic clients run their
complete request/response pipeline (serialization, parsing, typed models)
through praximetry's patches — no network needed.

The final test is a true live smoke test, opt-in via PRAXIMETRY_LIVE=1 + API keys.
"""

import json
import os

import httpx
import pytest

from praximetry.instrument.patch import auto_instrument
from praximetry.store import get_store

auto_instrument()


def _openai_json(text="mocked reply", tin=21, tout=7):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout},
    }


def test_openai_real_client_buffered():
    from openai import OpenAI

    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_openai_json()))
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
    resp = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.choices[0].message.content == "mocked reply"

    call = get_store().calls()[0]
    assert call.provider == "openai" and call.model == "gpt-4o"
    assert call.input_tokens == 21 and call.output_tokens == 7
    assert call.output_text == "mocked reply"
    assert call.cost_usd > 0 and call.latency_ms > 0


def test_openai_real_client_streaming():
    from openai import OpenAI

    def sse(data):
        return f"data: {json.dumps(data)}\n\n"

    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o"}
    body = (
        sse(
            {**chunk, "choices": [{"index": 0, "delta": {"content": "str"}, "finish_reason": None}]}
        )
        + sse(
            {
                **chunk,
                "choices": [{"index": 0, "delta": {"content": "eamed"}, "finish_reason": None}],
            }
        )
        + sse(
            {
                **chunk,
                "choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            }
        )
        + "data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=transport))

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text == "streamed"

    call = get_store().calls()[0]
    assert call.output_text == "streamed"
    assert call.input_tokens == 9 and call.output_tokens == 2


def _anthropic_json(text="claude says hi", tin=15, tout=5):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": tin, "output_tokens": tout},
    }


def test_anthropic_real_client_buffered():
    from anthropic import Anthropic

    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_anthropic_json()))
    client = Anthropic(api_key="test", http_client=httpx.Client(transport=transport))
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=64,
        system="be brief",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.content[0].text == "claude says hi"

    call = get_store().calls()[0]
    assert call.provider == "anthropic" and call.input_tokens == 15 and call.output_tokens == 5
    # system prompt captured as leading message
    assert call.messages[0] == {"role": "system", "content": "be brief"}
    assert call.cost_usd == pytest.approx(15 * 3 / 1e6 + 5 * 15 / 1e6)


def test_anthropic_real_client_async_buffered():
    import asyncio

    from anthropic import AsyncAnthropic

    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json=_anthropic_json("async hi"))
    )
    client = AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=transport))

    async def go():
        return await client.messages.create(
            model="claude-sonnet-5", max_tokens=64, messages=[{"role": "user", "content": "hi"}]
        )

    resp = asyncio.run(go())
    assert resp.content[0].text == "async hi"
    assert get_store().calls()[0].output_text == "async hi"


def test_override_rewrites_model_through_real_client():
    from openai import OpenAI

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["model"] = json.loads(req.content)["model"]
        return httpx.Response(200, json=_openai_json())

    client = OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    from praximetry.runtime import override_context

    with override_context(model="gpt-4o-mini"):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    # The override changed the model actually sent over the wire.
    assert seen["model"] == "gpt-4o-mini"
    assert get_store().calls()[0].model == "gpt-4o-mini"


@pytest.mark.skipif(
    not (os.environ.get("PRAXIMETRY_LIVE") and os.environ.get("ANTHROPIC_API_KEY")),
    reason="live test: set PRAXIMETRY_LIVE=1 and ANTHROPIC_API_KEY",
)
def test_live_anthropic_smoke():
    from anthropic import Anthropic

    Anthropic().messages.create(
        model="claude-haiku-4-5", max_tokens=16, messages=[{"role": "user", "content": "Say OK"}]
    )
    call = get_store().calls()[0]
    assert call.input_tokens > 0 and call.output_tokens > 0 and call.cost_usd > 0
    assert call.output_text
