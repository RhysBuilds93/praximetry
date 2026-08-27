"""Tests for src/praximetry/console.py."""

from praximetry.console import console


def test_console_theme_has_brand_styles():
    for name in ("success", "danger", "warn", "accent"):
        assert console.get_style(name).color is not None
