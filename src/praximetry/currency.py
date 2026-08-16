"""Display currency.

Provider prices are quoted in USD, so USD stays the canonical unit stored on
every Call (`cost_usd`). Everything shown to a human — CLI, dashboard, detector
messages — is converted here. Default is GBP; override with env vars:

    PRAXIMETRY_CURRENCY=GBP|USD|EUR
    PRAXIMETRY_FX_RATE=0.79            # display units per 1 USD
"""

from __future__ import annotations

import os

_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€"}
_DEFAULT_RATES = {"GBP": 0.79, "USD": 1.0, "EUR": 0.92}

CODE = os.environ.get("PRAXIMETRY_CURRENCY", "GBP").upper()
SYMBOL = _SYMBOLS.get(CODE, "£")
RATE = float(os.environ.get("PRAXIMETRY_FX_RATE", _DEFAULT_RATES.get(CODE, 0.79)))


def to_display(usd: float) -> float:
    return (usd or 0.0) * RATE


def fmt(usd: float, dp: int | None = None) -> str:
    """Format a USD amount in the display currency, e.g. 0.0193 USD -> '£0.0152'."""
    v = to_display(usd)
    if dp is None:
        dp = 4 if abs(v) < 1 else 2
    return f"{SYMBOL}{v:,.{dp}f}"
