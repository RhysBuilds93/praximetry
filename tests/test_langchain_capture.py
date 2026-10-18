import gc
import weakref
from uuid import uuid4

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.outputs import Generation, LLMResult

from praximetry.instrument.langchain import LangChainCallbackCapture
from praximetry.store import get_store


def test_langchain_capture_records_call():
    capture = LangChainCallbackCapture()
    llm = FakeListLLM(responses=["the answer is 4"], callbacks=[capture])
    llm.invoke("what is 2+2?")

    calls = get_store().calls()
    assert calls
    c = calls[0]
    assert c.provider == "langchain"
    assert c.output_text == "the answer is 4"
    assert c.reasoning_text == ""


def test_langchain_capture_records_real_text_for_non_openai_model():
    from uuid import uuid4

    capture = LangChainCallbackCapture()
    run_id = uuid4()
    capture.on_llm_start({}, ["hi"], run_id=run_id)
    result = LLMResult(
        generations=[[Generation(text="the answer is 4")]],
        llm_output={"model_name": "claude-sonnet-5"},
    )
    capture.on_llm_end(result, run_id=run_id)

    calls = get_store().calls()
    assert calls[0].output_text == "the answer is 4"

    run_id = uuid4()
    capture.on_llm_start({}, ["hi"], run_id=run_id)
    result = LLMResult(
        generations=[[Generation(text="the answer is 4")]],
        llm_output={"model_name": "gemini-2.0-flash"},
    )
    capture.on_llm_end(result, run_id=run_id)

    calls = get_store().calls()
    assert calls[-1].output_text == "the answer is 4"


def test_langchain_capture_records_token_usage_when_present():
    from uuid import uuid4

    capture = LangChainCallbackCapture()
    run_id = uuid4()
    capture.on_llm_start({}, ["hi"], run_id=run_id)
    result = LLMResult(
        generations=[[Generation(text="42")]],
        llm_output={
            "model_name": "gpt-4o",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )
    capture.on_llm_end(result, run_id=run_id)

    call = get_store().calls()[-1]
    assert call.input_tokens == 12
    assert call.output_tokens == 3


def test_langchain_capture_records_error():
    class BoomLLM(FakeListLLM):
        def _call(self, *a, **k):
            raise RuntimeError("boom")

    capture = LangChainCallbackCapture()
    llm = BoomLLM(responses=["unused"], callbacks=[capture])
    try:
        llm.invoke("trigger")
    except RuntimeError:
        pass
    calls = get_store().calls()
    assert calls[0].error and "boom" in calls[0].error


@pytest.mark.parametrize(
    ("serialized", "kwargs", "expected"),
    [
        ({"kwargs": {"model_name": "gpt-4o"}}, {}, "gpt-4o"),
        ({"kwargs": {"model": "claude-sonnet-5"}}, {}, "claude-sonnet-5"),
        ({}, {"invocation_params": {"model_name": "gpt-4o-mini"}}, "gpt-4o-mini"),
        ({}, {"invocation_params": {"model": "claude-sonnet-5"}}, "claude-sonnet-5"),
        ({}, {"model": "gemini-2.0-flash"}, "gemini-2.0-flash"),
        (
            {"kwargs": {"model_name": "baseline"}},
            {"invocation_params": {"model": "trial-model"}},
            "trial-model",
        ),
    ],
)
def test_langchain_errors_keep_the_model_and_prompt_from_start(serialized, kwargs, expected):
    capture = LangChainCallbackCapture()
    run_id = uuid4()
    capture.on_llm_start(serialized, ["original prompt"], run_id=run_id, **kwargs)
    capture.on_llm_error(RuntimeError("provider failed"), run_id=run_id)
    call = get_store().calls()[0]
    assert call.model == expected
    assert call.error == "provider failed"
    assert call.messages == [{"role": "user", "content": "original prompt"}]
    assert call.latency_ms >= 0


def test_langchain_success_uses_the_start_model_when_output_omits_it():
    capture = LangChainCallbackCapture()
    run_id = uuid4()
    capture.on_llm_start(
        {}, ["prompt"], run_id=run_id, invocation_params={"model_name": "gpt-4o"}
    )
    capture.on_llm_end(
        LLMResult(
            generations=[[Generation(text="answer")]],
            llm_output={"token_usage": {"prompt_tokens": 12, "completion_tokens": 3}},
        ),
        run_id=run_id,
    )
    call = get_store().calls()[0]
    assert call.model == "gpt-4o"
    assert call.cost_usd > 0
    assert call.output_text == "answer"


def test_langchain_abandoned_starts_are_bounded_and_new_calls_still_record():
    class Prompt(str):
        pass

    capture = LangChainCallbackCapture()
    abandoned_id = uuid4()
    prompt = Prompt("abandoned prompt")
    prompt_ref = weakref.ref(prompt)
    capture.on_llm_start({}, [prompt], run_id=abandoned_id)
    del prompt
    assert prompt_ref() is not None
    for _ in range(1024):
        newest_id = uuid4()
        capture.on_llm_start(
            {}, ["newest prompt"], run_id=newest_id, invocation_params={"model": "gpt-4o"}
        )
    gc.collect()
    assert prompt_ref() is None

    capture.on_llm_error(RuntimeError("late error"), run_id=abandoned_id)
    capture.on_llm_end(
        LLMResult(generations=[[Generation(text="answer")]]), run_id=newest_id
    )
    calls = get_store().calls()
    late = next(call for call in calls if call.error == "late error")
    newest = next(call for call in calls if call.output_text == "answer")
    assert late.messages == []
    assert newest.messages == [{"role": "user", "content": "newest prompt"}]
    assert newest.model == "gpt-4o"
