import pytest

from praximetry import config, pricing, runtime
from praximetry.store import reset_store


@pytest.fixture(autouse=True)
def fresh_env(tmp_path, monkeypatch):
    """Isolated DB + clean runtime state per test.

    PRAXIMETRY_DB is also set via env so example modules that call praximetry.init()
    at import time (rebuilding config from env) stay inside the tmp sandbox.
    """
    monkeypatch.setenv("PRAXIMETRY_DB", str(tmp_path / "test.db"))
    cfg = config.Config(project="test", db_path=tmp_path / "test.db")
    config.set_config(cfg)
    reset_store()
    runtime.STAGE_REGISTRY.clear()
    yield
    reset_store()


class FakeLLM:
    """Simulates an instrumented SDK: honors overrides, records calls.

    Cheaper models respond with configurable quality degradation.
    """

    def __init__(self, responses: dict[str, str] | None = None, degrade: dict[str, str] | None = None):
        self.responses = responses or {}
        self.degrade = degrade or {}  # model -> forced (wrong) answer

    def chat(self, model: str, messages: list[dict], expected_key: str = "") -> str:
        ov = runtime.get_overrides() or {}
        model = ov.get("model") or model
        if ov.get("prompt_transform"):
            messages = ov["prompt_transform"](messages)
        text = self.degrade.get(model, self.responses.get(expected_key, "ok"))
        tin = sum(len(str(m.get("content", ""))) // 4 for m in messages)
        runtime.record_call(
            provider="fake", model=model, messages=messages, response_text=text,
            input_tokens=tin, output_tokens=len(text) // 4 + 1,
            cost_usd=pricing.cost_usd(model, tin, len(text) // 4 + 1),
            latency_ms=1.0,
        )
        return text


@pytest.fixture
def fake_llm():
    return FakeLLM()
