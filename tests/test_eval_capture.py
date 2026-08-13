"""capture_request(): run a stage against a golden example's input and stop
right before the outbound LLM call, capturing its request shape.

No real network contact anywhere in this file — every stage under test uses
`fake_llm` / `record_call` directly, which record before any network call
would happen (there isn't one here), so interception is clean.
"""
import asyncio

import httpx
import pytest

import praximetry
from praximetry import runtime
from praximetry.eval import CaptureError, CapturedRequest, capture_request
from praximetry.eval.dataset import Example
from praximetry.instrument.patch import auto_instrument


def test_captures_provider_model_and_messages(fake_llm):
    @praximetry.stage("classify")
    def classify(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    ex = Example(id="e1", stage="classify", input="double charge")

    captured = capture_request(ex)

    assert isinstance(captured, CapturedRequest)
    assert captured.example_id == "e1"
    assert captured.stage == "classify"
    assert captured.provider == "fake"
    assert captured.model == "gpt-4o"
    assert captured.messages == [{"role": "user", "content": "double charge"}]


def test_captures_tool_defs_when_present():
    @praximetry.stage("plan_action")
    def plan_action(text):
        runtime.record_call(
            provider="fake", model="gpt-4o",
            messages=[{"role": "user", "content": text}],
            tools=[{"name": "get_order", "args": {"order_id": "str"}}],
        )
        raise AssertionError("should never get here — record_call should have halted execution")

    ex = Example(id="e1", stage="plan_action", input="track my order")

    captured = capture_request(ex)

    assert captured.tools == [{"name": "get_order", "args": {"order_id": "str"}}]


def test_never_persists_a_call(fake_llm):
    @praximetry.stage("classify")
    def classify(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    ex = Example(id="e1", stage="classify", input="double charge")
    capture_request(ex)

    from praximetry.store import get_store
    assert get_store().calls() == []


def test_execution_stops_before_post_call_code(fake_llm):
    ran_after = []

    @praximetry.stage("classify")
    def classify(text):
        fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)
        ran_after.append(True)  # must never run
        return "done"

    capture_request(Example(id="e1", stage="classify", input="x"))

    assert ran_after == []


def test_only_the_first_call_of_multiple_is_captured(fake_llm):
    @praximetry.stage("multi")
    def multi(text):
        fake_llm.chat("model-a", [{"role": "user", "content": "first"}], expected_key="first")
        fake_llm.chat("model-b", [{"role": "user", "content": "second"}], expected_key="second")
        return "done"

    captured = capture_request(Example(id="e1", stage="multi", input="x"))

    assert captured.model == "model-a"
    assert captured.messages == [{"role": "user", "content": "first"}]


def test_async_stage_support(fake_llm):
    @praximetry.stage("async_classify")
    async def async_classify(text):
        await asyncio.sleep(0)
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    captured = capture_request(Example(id="e1", stage="async_classify", input="x"))

    assert captured.model == "gpt-4o"


def test_unregistered_stage_error_lists_known_stages(fake_llm):
    @praximetry.stage("classify")
    def classify(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    with pytest.raises(CaptureError, match="ghost"):
        capture_request(Example(id="e1", stage="ghost", input="x"))

    with pytest.raises(CaptureError, match="classify"):
        capture_request(Example(id="e1", stage="ghost", input="x"))


def test_no_llm_call_raises_capture_error():
    @praximetry.stage("no_call")
    def no_call(text):
        return text.upper()

    with pytest.raises(CaptureError, match="no_call"):
        capture_request(Example(id="e1", stage="no_call", input="x"))


def test_captures_via_instrumented_real_sdk_client():
    """A stage that calls through an auto_instrument()-patched real SDK client
    (not runtime.record_call directly) must still be captured cleanly, with
    no real network contact — see PRA-66."""
    from openai import OpenAI

    auto_instrument()

    def _unreachable(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network call must not happen during capture")

    client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(_unreachable)))

    @praximetry.stage("real_sdk_classify")
    def real_sdk_classify(text):
        resp = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": text}])
        return resp.choices[0].message.content

    ex = Example(id="e1", stage="real_sdk_classify", input="double charge")

    captured = capture_request(ex)

    assert isinstance(captured, CapturedRequest)
    assert captured.provider == "openai"
    assert captured.model == "gpt-4o"
    assert captured.messages == [{"role": "user", "content": "double charge"}]


def test_record_call_restored_after_success_and_error(fake_llm):
    @praximetry.stage("classify")
    def classify(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    @praximetry.stage("no_call")
    def no_call(text):
        return text.upper()

    original = runtime.record_call

    capture_request(Example(id="e1", stage="classify", input="x"))
    assert runtime.record_call is original

    with pytest.raises(CaptureError):
        capture_request(Example(id="e1", stage="no_call", input="x"))
    assert runtime.record_call is original
