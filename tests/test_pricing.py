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


@pytest.mark.parametrize(
    "model", ["o3-mini", "o3-pro", "claude-sonnet-5-fast", "gemini-2.5-flash-lite-mini"]
)
def test_distinct_models_sharing_a_prefix_are_unpriced(model: str):
    assert pricing.is_unpriced(model)
    assert pricing.cost_usd(model, 1_000_000, 1_000_000) == 0.0


@pytest.mark.parametrize(
    ("snapshot", "base"),
    [
        ("o3-2025-04-16", "o3"),
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ("gpt-4.1-nano-2025-04-14", "gpt-4.1-nano"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
        ("gemini-2.5-flash-lite@001", "gemini-2.5-flash-lite"),
        ("gemini-2.5-flash-preview", "gemini-2.5-flash"),
        ("gemini-2.5-flash-preview-05-20", "gemini-2.5-flash"),
        ("gemini-2.5-flash-preview-2025-05-20", "gemini-2.5-flash"),
        ("gemini-2.5-flash-latest", "gemini-2.5-flash"),
        ("gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-lite"),
    ],
)
def test_snapshot_resolves_to_most_specific_base(snapshot: str, base: str):
    assert pricing._lookup(snapshot) == pricing.PRICING[base]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o-mini", []),
        ("gpt-4.1-mini", ["gpt-4.1-nano"]),
        ("gpt-4.1-nano-2025-04-14", []),
        ("gemini-2.5-flash-lite", []),
        ("gemini-2.5-flash-2025-06-01", ["gemini-2.5-flash-lite"]),
        ("gpt-4.1", ["gpt-4.1-mini", "gpt-4.1-nano"]),
        ("o3-mini", []),
    ],
)
def test_cheaper_models_are_strictly_cheaper(model: str, expected: list[str]):
    assert pricing.cheaper_models(model) == expected


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "expected"),
    [
        (None, None, 0.0),
        (None, 1_000_000, 10.0),
        (1_000_000, None, 2.5),
        (-1, -1, 0.0),
        (-1, 1_000_000, 10.0),
        (1_000_000, -1, 2.5),
    ],
)
def test_missing_and_negative_token_counts_are_zero(input_tokens, output_tokens, expected):
    cost = pricing.cost_usd("gpt-4o", input_tokens, output_tokens)
    assert isinstance(cost, float)
    assert cost == expected
