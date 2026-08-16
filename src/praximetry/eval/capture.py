"""Capture the request shape a stage would send to an LLM, without sending it.

`capture_request()` runs a stage's real registered code against one golden
example's business-level input, then instead of letting the call complete, it
intercepts the first `praximetry.runtime.record_call` invocation, grabs its
kwargs (provider, model, messages, tool defs, ...), and halts execution right
there. No real network contact, and no `Call` is ever persisted.

**How the interception works:** two complementary paths, both active for the
duration of the stage call:

- A stage that calls `runtime.record_call(...)` directly does so via a
  *module attribute* lookup at call time — so reassigning
  `praximetry.runtime.record_call` is enough to intercept it, with zero
  coordination from the stage's own code, *provided the stage calls
  `record_call` before it calls out to a model*. The test-only `FakeLLM` does
  this (records first, no network involved at all).
- A stage that instead calls through an `auto_instrument()`-patched real SDK
  client (openai/anthropic/etc) would otherwise make the real network call
  first and only call `record_call` afterward with the real token counts —
  too late to intercept. `praximetry.instrument.patch.capturing()` (PRA-66)
  installs a pre-flight hook inside the patched create()/acreate() wrappers
  that fires *before* the real SDK call, so this path is intercepted too,
  with no network contact.

Both paths raise `_CaptureSignal` to unwind before any real call completes,
and converge on the same `CapturedRequest` shape.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import praximetry.runtime as runtime
from pydantic import BaseModel, Field

from ..instrument import patch as instrument_patch
from .dataset import Example


class CapturedRequest(BaseModel):
    example_id: str
    stage: str
    provider: str = "unknown"
    model: str = "unknown"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)


class CaptureError(RuntimeError):
    """Capture could not produce a request shape for this example."""


class _CaptureSignal(Exception):
    """Internal control-flow signal carrying the intercepted call kwargs."""

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = kwargs


def call_stage(fn: Any, example: Example) -> Any:
    inp = example.input
    if isinstance(inp, dict):
        out = fn(**inp)
    elif isinstance(inp, list):
        out = fn(*inp)
    else:
        out = fn(inp)
    if inspect.iscoroutine(out):  # async stages run to completion
        out = asyncio.run(out)
    return out


def capture_request(example: Example) -> CapturedRequest:
    """Run `example`'s stage until its first outbound-LLM-call-shaped invocation
    and return that call's request shape, without ever completing it."""
    fn = runtime.STAGE_REGISTRY.get(example.stage)
    if fn is None:
        raise CaptureError(
            f"Stage '{example.stage}' not registered. Decorate it with @praximetry.stage "
            f"and import the module before capturing. Known: {sorted(runtime.STAGE_REGISTRY)}"
        )

    def _intercept(call: Any = None, **kwargs: Any) -> Any:
        if call is not None:
            kwargs = {
                "provider": call.provider,
                "model": call.model,
                "messages": call.messages,
                **kwargs,
            }
        raise _CaptureSignal(kwargs)

    def _hook(kwargs: dict[str, Any]) -> Any:
        raise _CaptureSignal(kwargs)

    original = runtime.record_call
    runtime.record_call = _intercept
    try:
        with instrument_patch.capturing(_hook):
            call_stage(fn, example)
    except _CaptureSignal as signal:
        kwargs = signal.kwargs
        return CapturedRequest(
            example_id=example.id,
            stage=example.stage,
            provider=kwargs.get("provider", "unknown"),
            model=kwargs.get("model", "unknown"),
            messages=kwargs.get("messages", []),
            tools=kwargs.get("tools", []),
        )
    else:
        raise CaptureError(
            f"Stage '{example.stage}' example '{example.id}' finished without making an "
            "LLM-shaped call (record_call was never invoked) — nothing to capture."
        )
    finally:
        runtime.record_call = original
