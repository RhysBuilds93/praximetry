"""Brand console: shared Rich Console + theme for CLI output.

Colors are hex conversions of praximetry-cloud's canonical OKLCH tokens in
site/theme.css (the source of truth for praximetry.io + the dashboard) — kept
in sync by eye, not by importing that repo (closed-source, one-way dependency
direction: praximetry-cloud -> praximetry).

The startup wordmark lives in `_banner.py`, not here.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

ACCENT = "#009e77"
SUCCESS = "#43b966"
WARN = "#dfa11a"
DANGER = "#e85854"

_THEME = Theme(
    {
        "success": f"bold {SUCCESS}",
        "danger": f"bold {DANGER}",
        "warn": f"bold {WARN}",
        "accent": ACCENT,
    }
)

console = Console(theme=_THEME)
