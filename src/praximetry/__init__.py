"""praximetry — drop-in observability for LLM agents.

Quickstart:
    import praximetry as px
    px.init(project="my-agent")   # auto-instruments openai/anthropic SDKs

    @px.stage("summarize")
    def summarize(text): ...

    # then:  praximetry summary
"""
from __future__ import annotations

from pathlib import Path

import os

from . import cloud_sync
from .config import Config, get_config, set_config
from .eval.hosted import CloudError, client_from_env
from .instrument import auto_instrument, stage
from .models import Call
from .otel import instrument_otel, record_spans
from .pricing import register_pricing
from .runtime import current_run, record_call, run_context

__version__ = "0.1.0"
__all__ = [
    "init", "stage", "record_call", "run_context", "current_run",
    "register_pricing", "instrument_otel", "record_spans",
    "Call", "get_config", "CloudError",
]


def init(
    project: str = "default",
    db_path: str | Path | None = None,
    auto_instrument_sdks: bool = True,
    cloud: bool | None = None,
) -> list[str]:
    """Initialize praximetry. Returns list of auto-instrumented providers.

    `cloud`: whether to also push recorded calls to the hosted /api/traces
    route in the background (see `praximetry.cloud_sync`).
      - None (default): auto-enable only if PRAXIMETRY_API_KEY is set in the
        environment. Unset -> zero behavior change from today.
      - True: force-enable; raises CloudError if PRAXIMETRY_API_KEY is unset
        (an explicit opt-in with missing config should fail loudly).
      - False: force local-only regardless of whether a key is present.
    """
    cfg = Config.from_env()
    cfg.project = project
    if db_path:
        cfg.db_path = Path(db_path)
    cfg.auto_instrument = auto_instrument_sdks
    set_config(cfg)

    want_cloud = cloud if cloud is not None else bool(os.environ.get("PRAXIMETRY_API_KEY"))
    if want_cloud:
        cloud_sync.start(client_from_env())

    return auto_instrument() if auto_instrument_sdks else []
