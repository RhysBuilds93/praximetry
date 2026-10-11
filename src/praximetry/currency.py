"""Display currency.

Provider prices are quoted in USD, so USD stays the canonical unit stored on
every Call (`cost_usd`). Everything shown to a human — CLI, dashboard, detector
messages — is converted here. Default is GBP; override with env vars:

    PRAXIMETRY_CURRENCY=GBP|USD|EUR  # other codes need FX_RATE
    PRAXIMETRY_FX_RATE=0.79            # display units per 1 USD
"""

from __future__ import annotations

import os
import warnings

_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€"}
_DEFAULT_RATES = {"GBP": 0.79, "USD": 1.0, "EUR": 0.92}

def _load() -> tuple[str, str, float]:
    code = os.environ.get("PRAXIMETRY_CURRENCY", "GBP").upper()
    raw_rate = os.environ.get("PRAXIMETRY_FX_RATE")
    rate = None
    if raw_rate is not None:
        try:
            rate = float(raw_rate)
        except ValueError:
            warnings.warn(f"Ignoring invalid PRAXIMETRY_FX_RATE={raw_rate!r}", stacklevel=2)
    if code not in _SYMBOLS and rate is None:
        warnings.warn(
            f"PRAXIMETRY_CURRENCY={code} needs a valid PRAXIMETRY_FX_RATE; using GBP",
            stacklevel=2,
        )
        code = "GBP"
    if rate is None:
        rate = _DEFAULT_RATES[code]
    return code, _SYMBOLS.get(code, f"{code} "), rate


CODE, SYMBOL, RATE = _load()


def to_display(usd: float) -> float:
    return (usd or 0.0) * RATE


def fmt(usd: float, dp: int | None = None) -> str:
    """Format a USD amount in the display currency, e.g. 0.0193 USD -> '£0.0152'."""
    v = to_display(usd)
    if dp is None:
        dp = 4 if abs(v) < 1 else 2
    return f"{SYMBOL}{v:,.{dp}f}"
