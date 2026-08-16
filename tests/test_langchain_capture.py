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
