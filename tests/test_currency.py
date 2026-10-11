import importlib

import pytest

from praximetry import currency


@pytest.fixture(autouse=True)
def restore_currency(monkeypatch: pytest.MonkeyPatch):
    yield
    importlib.reload(currency)


def reload_with(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    monkeypatch.delenv("PRAXIMETRY_CURRENCY", raising=False)
    monkeypatch.delenv("PRAXIMETRY_FX_RATE", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(f"PRAXIMETRY_{name}", value)
    importlib.reload(currency)


def test_unsupported_code_with_rate_uses_code_not_gbp_symbol(monkeypatch):
    reload_with(monkeypatch, CURRENCY="JPY", FX_RATE="150")

    assert currency.CODE == "JPY"
    assert currency.fmt(1.0) == "JPY 150.00"


def test_unsupported_code_without_rate_warns_and_falls_back_to_gbp(monkeypatch):
    with pytest.warns(UserWarning, match="JPY"):
        reload_with(monkeypatch, CURRENCY="JPY")

    assert (currency.CODE, currency.SYMBOL, currency.RATE) == ("GBP", "£", 0.79)


def test_invalid_rate_warns_and_uses_default_instead_of_crashing(monkeypatch):
    with pytest.warns(UserWarning, match="PRAXIMETRY_FX_RATE"):
        reload_with(monkeypatch, CURRENCY="EUR", FX_RATE="abc")

    assert currency.RATE == 0.92
