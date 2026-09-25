"""Model pricing lookup and cost computation, including prefix matching."""

import pytest

from praximetry import pricing


def test_cost_usd_exact_match():
    cost = pricing.cost_usd("gpt-4o", 1_000_000, 1_000_000)
    assert cost == pytest.approx(2.5 + 10.0)


def test_cost_usd_unknown_model_is_zero():
    assert pricing.cost_usd("some-model-nobody-priced", 1000, 1000) == 0.0


def test_is_unpriced():
    assert pricing.is_unpriced("some-model-nobody-priced") is True
    assert pricing.is_unpriced("gpt-4o") is False


def test_dated_snapshot_matches_by_prefix():
    assert pricing.cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == pricing.cost_usd(
        "claude-haiku-4-5", 1_000_000, 0
    )
    assert pricing.is_unpriced("claude-haiku-4-5-20251001") is False


def test_longer_prefix_wins_over_shorter():
    # "gpt-4o-mini-2024-07-18" starts with both "gpt-4o" and "gpt-4o-mini".
    assert pricing.cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pricing.cost_usd(
        "gpt-4o-mini", 1_000_000, 0
    )


def test_register_pricing_adds_new_model():
    pricing.register_pricing("test-only-model-xyz", 1.0, 2.0)
    assert pricing.is_unpriced("test-only-model-xyz") is False
    assert pricing.cost_usd("test-only-model-xyz", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_register_pricing_overwrites_existing_model():
    original = pricing.PRICING["gpt-4o-mini"]
    try:
        pricing.register_pricing("gpt-4o-mini", 99.0, 99.0)
        assert pricing.cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(99.0)
    finally:
        pricing.PRICING["gpt-4o-mini"] = original


def test_cheaper_models_for_known_model():
    assert pricing.cheaper_models("gpt-4o") == ["gpt-4o-mini"]


def test_cheaper_models_for_unknown_model_is_empty():
    assert pricing.cheaper_models("some-model-nobody-priced") == []
