"""Capture the request shape a stage would send to an LLM, without sending it.

`capture_request()` runs a stage's real registered code against one golden
example's business-level input, then instead of letting the call complete, it
intercepts the first `praximetry.runtime.record_call` invocation, grabs its
kwargs (provider, model, messages, tool defs, ...), and halts execution right
there. No real network contact, and no `Call` is ever persisted.

**How the interception works, and its limit:** a stage calls
`runtime.record_call(...)` via a *module attribute* lookup at call time — so
reassigning `praximetry.runtime.record_call` for the duration of the call is
enough to intercept it, with zero coordination from the stage's own code,
*provided the stage calls `record_call` before it calls out to a model*. The
test-only `FakeLLM` does this (records first, no network involved at all). It
does **not** intercept `praximetry.instrument.auto_instrument()`'s patched real
SDK clients (openai/anthropic/etc) — those make the real network call first
and only call `record_call` afterward with the real token counts, so by the
time `capture_request()` intercepts anything, the request has already gone
out. Customer stages that call `runtime.record_call` directly with their own
pre-flight request shape are intercepted cleanly; stages that go through an
instrumented real SDK client are not — a capture-mode addition to
`praximetry.instrument.patch` would be needed for those (see PRA-66).
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import praximetry.runtime as runtime
from pydantic import BaseModel, Field

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
    """Internal control-flow signal — carries the intercepted record_call() kwargs."""

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = kwargs


def call_stage(fn: Any, example: Example) -> Any:
    """Invoke `fn` with `example.input` shaped to match how the stage is called."""
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

    original = runtime.record_call
    runtime.record_call = _intercept
    try:
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
