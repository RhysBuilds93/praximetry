"""Model pricing (USD per million tokens) and cost computation.

Prices as of mid-2026. Update via PRICING or register_pricing().
"""

from __future__ import annotations

# (input $/MTok, output $/MTok)
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # OpenAI (representative)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "o3": (2.0, 8.0),
    "o4-mini": (1.1, 4.4),
    # Google Gemini (representative)
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.5),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
}

# Cheaper-alternative ladder used by the optimizer, per provider tier.
DOWNGRADE_LADDER: dict[str, list[str]] = {
    "claude-fable-5": ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"],
    "claude-opus-4-8": ["claude-sonnet-5", "claude-haiku-4-5"],
    "claude-sonnet-5": ["claude-haiku-4-5"],
    "claude-sonnet-4-6": ["claude-haiku-4-5"],
    "gpt-4o": ["gpt-4o-mini"],
    "gpt-4.1": ["gpt-4.1-mini", "gpt-4.1-nano"],
    "o3": ["o4-mini"],
    "gemini-2.5-pro": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "gemini-2.5-flash": ["gemini-2.5-flash-lite"],
    "gemini-1.5-pro": ["gemini-1.5-flash"],
}


def register_pricing(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    PRICING[model] = (input_per_mtok, output_per_mtok)


def _lookup(model: str) -> tuple[float, float] | None:
    if model in PRICING:
        return PRICING[model]
    # Dated snapshots like claude-haiku-4-5-20251001 -> longest prefix match.
    best = None
    for known in PRICING:
        if model.startswith(known) and (best is None or len(known) > len(best)):
            best = known
    return PRICING[best] if best else None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _lookup(model)
    if not p:
        return 0.0
    return input_tokens * p[0] / 1e6 + output_tokens * p[1] / 1e6


def cheaper_models(model: str) -> list[str]:
    for known, ladder in DOWNGRADE_LADDER.items():
        if model.startswith(known):
            return ladder
    return []
