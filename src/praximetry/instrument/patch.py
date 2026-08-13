"""Auto-instrumentation: monkey-patch OpenAI, Anthropic, and LiteLLM SDKs.

Covers sync + async and buffered + streaming calls. Experiment overrides
(model swap, prompt transform) are applied in-flight so the optimizer can trial
variants without touching user code.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .. import pricing
from ..runtime import get_overrides, record_call
from .providers import PROVIDERS, ProviderSpec
from .wrap import AsyncStreamWrapper, SyncStreamWrapper

_patched: set[str] = set()


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


def _record(provider: str, model: str, messages: list, text: str,
            tin: int, tout: int, t0: float, error: str | None) -> None:
    record_call(
        provider=provider, model=model, messages=messages, response_text=text,
        input_tokens=tin, output_tokens=tout, cost_usd=pricing.cost_usd(model, tin, tout),
        latency_ms=(time.perf_counter() - t0) * 1000, error=error,
    )


def _make_stream_done(provider: str, model: str, messages: list, t0: float):
    def on_done(state: dict[str, Any]) -> None:
        _record(provider, model, messages, state["text"], state["tin"], state["tout"], t0, None)
    return on_done


def _instrument(original: Callable, provider: str, get_messages, response_extract,
                accumulate, is_async: bool, messages_key: str = "messages",
                force_stream: bool = False) -> Callable:
    """Build a patched create() wrapping `original` for one provider/sync-ness."""
    if is_async:
        async def acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs = _apply_overrides(kwargs, messages_key)
            model, messages = kwargs.get("model", "unknown"), get_messages(kwargs)
            t0 = time.perf_counter()
            if force_stream or kwargs.get("stream"):
                resp = await original(self, *args, **kwargs)
                return AsyncStreamWrapper(resp, accumulate,
                                          _make_stream_done(provider, model, messages, t0))
            try:
                resp = await original(self, *args, **kwargs)
            except Exception as e:  # noqa: BLE001
                _record(provider, model, messages, "", 0, 0, t0, str(e))
                raise
            text, tin, tout = response_extract(resp)
            _record(provider, model, messages, text, tin, tout, t0, None)
            return resp
        acreate._praximetry_patched = True  # type: ignore[attr-defined]
        return acreate

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = _apply_overrides(kwargs, messages_key)
        model, messages = kwargs.get("model", "unknown"), get_messages(kwargs)
        t0 = time.perf_counter()
        if force_stream or kwargs.get("stream"):
            resp = original(self, *args, **kwargs)
            return SyncStreamWrapper(resp, accumulate,
                                     _make_stream_done(provider, model, messages, t0))
        try:
            resp = original(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            _record(provider, model, messages, "", 0, 0, t0, str(e))
            raise
        text, tin, tout = response_extract(resp)
        _record(provider, model, messages, text, tin, tout, t0, None)
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
                _self_less(original), spec.name, spec.get_messages, spec.response_extract,
                spec.accumulate, is_async=target.is_async, messages_key=spec.messages_key,
                force_stream=target.force_stream)
            setattr(host, target.attr, _self_less_caller(inst, target.is_async))
        else:
            new = _instrument(
                original, spec.name, spec.get_messages, spec.response_extract,
                spec.accumulate, is_async=target.is_async, messages_key=spec.messages_key,
                force_stream=target.force_stream)
            setattr(host, target.attr, new)  # type: ignore[method-assign]

    _patched.add(spec.name)
    return True
