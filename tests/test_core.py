"""Core: pricing, store, decorators, recording."""
import praximetry
from praximetry import pricing
from praximetry.runtime import STAGE_REGISTRY, record_call
from praximetry.store import get_store


def test_pricing_known_and_prefix():
    assert pricing.cost_usd("claude-sonnet-5", 1_000_000, 0) == 3.0
    assert pricing.cost_usd("claude-haiku-4-5-20251001", 0, 1_000_000) == 5.0
    assert pricing.cost_usd("mystery-model", 1000, 1000) == 0.0


def test_cheaper_models_ladder():
    assert "claude-haiku-4-5" in pricing.cheaper_models("claude-sonnet-5")
    assert pricing.cheaper_models("gpt-4o") == ["gpt-4o-mini"]
    assert pricing.cheaper_models("unknown") == []


def test_stage_decorator_registers_and_attributes():
    @praximetry.stage("greet")
    def greet(name):
        record_call(provider="fake", model="gpt-4o", input_tokens=10, output_tokens=5,
                    cost_usd=0.001, messages=[{"role": "user", "content": name}])
        return f"hi {name}"

    assert "greet" in STAGE_REGISTRY
    assert greet("bob") == "hi bob"
    calls = get_store().calls()
    assert len(calls) == 1
    assert calls[0].stage == "greet"
    assert calls[0].input_tokens == 10


def test_bare_stage_decorator():
    @praximetry.stage
    def summarize(text):
        return text[:3]

    assert "summarize" in STAGE_REGISTRY
    assert summarize("hello") == "hel"


def test_store_totals_and_summary():
    for i in range(3):
        record_call(provider="fake", model="gpt-4o", stage="s1",
                    input_tokens=100, output_tokens=50, cost_usd=0.01)
    t = get_store().totals()
    assert t["n"] == 3 and t["tin"] == 300 and abs(t["cost"] - 0.03) < 1e-9
    summary = get_store().stage_summary()
    assert summary[0]["stage"] == "s1" and summary[0]["n"] == 3
