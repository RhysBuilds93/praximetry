from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage

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
