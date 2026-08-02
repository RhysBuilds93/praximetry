"""praximetry — drop-in observability for LLM agents.

Quickstart:
    import praximetry
    praximetry.init(project="my-agent")   # auto-instruments openai/anthropic SDKs

    @praximetry.stage("summarize")
    def summarize(text): ...

    # then:  praximetry summary
"""
from __future__ import annotations

from pathlib import Path

from .config import Config, get_config, set_config
from .instrument import auto_instrument, stage
from .models import Call
from .otel import instrument_otel, record_spans
from .pricing import register_pricing
from .runtime import current_run, record_call, run_context

__version__ = "0.1.0"
__all__ = [
    "init", "stage", "record_call", "run_context", "current_run",
    "register_pricing", "instrument_otel", "record_spans",
    "Call", "get_config",
]


def init(
    project: str = "default",
    db_path: str | Path | None = None,
    auto_instrument_sdks: bool = True,
) -> list[str]:
    """Initialize praximetry. Returns list of auto-instrumented providers."""
    cfg = Config.from_env()
    cfg.project = project
    if db_path:
        cfg.db_path = Path(db_path)
    cfg.auto_instrument = auto_instrument_sdks
    set_config(cfg)
    return auto_instrument() if auto_instrument_sdks else []
