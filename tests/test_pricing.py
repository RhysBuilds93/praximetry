"""Pricing edge cases not already covered by test_core.py's pricing tests."""

import pytest

from praximetry import pricing


def test_longer_prefix_wins_over_shorter():
    # "gpt-4o-mini-2024-07-18" starts with both "gpt-4o" and "gpt-4o-mini".
    assert pricing.cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pricing.cost_usd(
        "gpt-4o-mini", 1_000_000, 0
    )


def test_register_pricing_overwrites_existing_model():
    original = pricing.PRICING["gpt-4o-mini"]
    try:
        pricing.register_pricing("gpt-4o-mini", 99.0, 99.0)
        assert pricing.cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(99.0)
    finally:
        pricing.PRICING["gpt-4o-mini"] = original
