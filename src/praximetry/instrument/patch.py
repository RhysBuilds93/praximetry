"""Auto-instrumentation: monkey-patch OpenAI, Anthropic, LiteLLM, and Gemini SDKs.

Covers sync + async and buffered + streaming calls. Experiment overrides
(model swap, prompt transform) are applied in-flight so the optimizer can trial
variants without touching user code.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any
from collections.abc import Callable

from .. import pricing
from ..runtime import get_overrides, record_call
from .output import NormalizedOutput
from .providers import PROVIDERS, ProviderSpec
from .wrap import AsyncStreamWrapper, SyncStreamWrapper

_patched: set[str] = set()

# Pre-flight capture hook (PRA-66): when set, patched create()/acreate()
# wrappers call this INSTEAD of the real `original(...)`, so
# `praximetry.eval.capture` can intercept requests that go through an
# `auto_instrument()`-patched SDK client, not just direct `record_call` use.
# The hook is expected to raise to unwind the call stack before any network
# contact happens.
_capture_hook: Callable[[dict], Any] | None = None


def _unwrap_for_parsing(resp: Any) -> Any:
    """LangChain (and any caller using `with_raw_response`) gets back a
    LegacyAPIResponse wrapper with no `.usage`/`.choices` -- only its parsed
    `.parse()` result has those. `.parse()` is cached, so calling it here
    doesn't disturb the caller's own later `.parse()` call on the same resp.
    """
    parse = getattr(resp, "parse", None)
    return parse() if callable(parse) else resp


@contextmanager
def capturing(hook: Callable[[dict], Any]):
    """Route the next patched SDK call(s) to `hook` instead of the real API.

    `hook` receives {"provider", "model", "messages", "tools"} and is
    expected to raise to halt execution before `original(...)` is invoked.
    """
    global _capture_hook
    prev, _capture_hook = _capture_hook, hook
    try:
        yield
    finally:
        _capture_hook = prev


def auto_instrument() -> list[str]:
    """Patch whichever supported SDKs are importable. Returns providers patched."""
    done = []
    for spec in PROVIDERS:
        try:
            if _patch(spec):
                done.append(spec.name)
        except Exception:  # a broken/renamed SDK internal must not crash init()
            pass
    return done


def _apply_overrides(kwargs: dict[str, Any], messages_key: str) -> dict[str, Any]:
    ov = get_overrides()
    if not ov:
        return kwargs
    if ov.get("model"):
        kwargs["model"] = ov["model"]
    transform = ov.get("prompt_transform")
    val = kwargs.get(messages_key)
    is_message_list = isinstance(val, list) and val and isinstance(val[0], dict)
    if transform and is_message_list:
        kwargs[messages_key] = transform(kwargs[messages_key])
        if isinstance(kwargs.get("system"), str):
            kwargs["system"] = transform([{"role": "system", "content": kwargs["system"]}])[0][
                "content"
            ]
    return kwargs


def _record(
    provider: str, model: str, messages: list, out: NormalizedOutput, t0: float, error: str | None
) -> None:
    record_call(
        provider=provider,
        model=model,
        messages=messages,
        output_text=out.output_text,
        reasoning_text=out.reasoning_text,
        tool_calls=[tc.model_dump() for tc in out.tool_calls],
        structured_output=out.structured_output,
        content_parts=[cp.model_dump() for cp in out.content_parts],
        input_tokens=out.tokens_in,
        output_tokens=out.tokens_out,
        cost_usd=pricing.cost_usd(model, out.tokens_in, out.tokens_out),
        latency_ms=(time.perf_counter() - t0) * 1000,
        error=error,
    )


def _make_stream_done(provider: str, model: str, messages: list, adapter, t0: float):
    def on_done(state: dict[str, Any]) -> None:
        state["model"] = model
        out = adapter.finalize_stream(state)
        _record(provider, model, messages, out, t0, None)

    return on_done


def _instrument(
    original: Callable,
    provider: str,
    adapter,
    is_async: bool,
    messages_key: str = "messages",
    force_stream: bool = False,
) -> Callable:
    """Build a patched create() wrapping `original` for one provider/sync-ness."""
    if is_async:

        async def acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs = _apply_overrides(kwargs, messages_key)
            model, messages = kwargs.get("model", "unknown"), adapter.get_messages(kwargs)
            if _capture_hook is not None:
                return _capture_hook(
                    {
                        "provider": provider,
                        "model": model,
                        "messages": messages,
                        "tools": kwargs.get("tools", []),
                    }
                )
            t0 = time.perf_counter()
            if force_stream or kwargs.get("stream"):
                resp = await original(self, *args, **kwargs)
                return AsyncStreamWrapper(
                    resp,
                    adapter.accumulate,
                    _make_stream_done(provider, model, messages, adapter, t0),
                )
            try:
                resp = await original(self, *args, **kwargs)
            except Exception as e:  # noqa: BLE001
                _record(provider, model, messages, NormalizedOutput(), t0, str(e))
                raise
            out = adapter.parse_response(_unwrap_for_parsing(resp), model)
            _record(provider, model, messages, out, t0, None)
            return resp

        acreate._praximetry_patched = True  # type: ignore[attr-defined]
        return acreate

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = _apply_overrides(kwargs, messages_key)
        model, messages = kwargs.get("model", "unknown"), adapter.get_messages(kwargs)
        if _capture_hook is not None:
            return _capture_hook(
                {
                    "provider": provider,
                    "model": model,
                    "messages": messages,
                    "tools": kwargs.get("tools", []),
                }
            )
        t0 = time.perf_counter()
        if force_stream or kwargs.get("stream"):
            resp = original(self, *args, **kwargs)
            return SyncStreamWrapper(
                resp, adapter.accumulate, _make_stream_done(provider, model, messages, adapter, t0)
            )
        try:
            resp = original(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            _record(provider, model, messages, NormalizedOutput(), t0, str(e))
            raise
        out = adapter.parse_response(_unwrap_for_parsing(resp), model)
        _record(provider, model, messages, out, t0, None)
        return resp

    create._praximetry_patched = True  # type: ignore[attr-defined]
    return create


def _self_less(orig: Callable) -> Callable:
    """Wrap a module-level function (no `self`) so `_instrument()` can treat
    it like a class method. Matches the technique litellm.completion/
    acompletion need since they have no bound receiver."""

    def with_self(_self: Any, *a: Any, **k: Any) -> Any:
        return orig(*a, **k)

    return with_self


def _self_less_caller(inst: Callable, is_async: bool) -> Callable:
    """Strip the `self` arg back off before calling an `_instrument()`-built
    wrapper, so callers of the module-level function see its original
    (self-less) signature again."""
    if is_async:

        async def acall(*a: Any, **k: Any) -> Any:
            return await inst(None, *a, **k)

        return acall

    def call(*a: Any, **k: Any) -> Any:
        return inst(None, *a, **k)

    return call


def _patch(spec: ProviderSpec) -> bool:
    """Apply one provider's PatchTarget table, generically."""
    if spec.name in _patched:
        return True
    try:
        sync_host, async_host = spec.owner()
    except ImportError:
        return False

    for target in spec.targets:
        host = async_host if target.is_async else sync_host
        if target.optional and not hasattr(host, target.attr):
            continue
        original = getattr(host, target.attr)
        if target.self_less:
            inst = _instrument(
                _self_less(original),
                spec.name,
                spec.adapter,
                is_async=target.is_async,
                messages_key=spec.messages_key,
                force_stream=target.force_stream,
            )
            setattr(host, target.attr, _self_less_caller(inst, target.is_async))
        else:
            new = _instrument(
                original,
                spec.name,
                spec.adapter,
                is_async=target.is_async,
                messages_key=spec.messages_key,
                force_stream=target.force_stream,
            )
            setattr(host, target.attr, new)  # type: ignore[method-assign]

    _patched.add(spec.name)
    return True
