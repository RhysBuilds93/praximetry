import pytest

from langchain_core.messages import AIMessage

from praximetry.examples.workflows import (
    _real,
    human_approval,
    incident_response,
    rag_retrieval,
    research_supervisor,
    supervisor_delegation,
)


class FakeChatModel:
    def __init__(self, content):
        self.content = content

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content=self.content)


@pytest.fixture
def fake_model():
    requested = []

    def factory(name):
        requested.append(name)
        return FakeChatModel('{"domains": ["billing"]}')

    _real.set_chat_model_factory(factory)
    yield requested
    _real.set_chat_model_factory(None)


@pytest.mark.parametrize(
    ("handle", "request_text"),
    [
        (human_approval.handle, "Restart the checkout service"),
        (incident_response.handle, "Checkout latency spiked after the deploy"),
        (rag_retrieval.handle, "How long do refunds take?"),
        (research_supervisor.handle, "How does our pricing compare to competitors?"),
        (supervisor_delegation.handle, "I was charged twice and the app keeps crashing"),
    ],
)
def test_workflow_runs_on_injected_model(fake_model, handle, request_text):
    handle(request_text)
    assert fake_model


def test_default_factory_builds_real_client(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "test")
    assert type(_real.chat_model("gpt-4o-mini")).__name__ == "ChatOpenAI"
