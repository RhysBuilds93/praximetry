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
from . import extractors as ex
from .wrap import AsyncStreamWrapper, SyncStreamWrapper

_patched: set[str] = set()


def auto_instrument() -> list[str]:
    """Patch whichever supported SDKs are importable. Returns providers patched."""
    done = []
    for name, fn in (("openai", _patch_openai), ("anthropic", _patch_anthropic),
                     ("litellm", _patch_litellm), ("gemini", _patch_gemini)):
        try:
            if fn():
                done.append(name)
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


def _patch_openai() -> bool:
    if "openai" in _patched:
        return True
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except ImportError:
        return False
    Completions.create = _instrument(  # type: ignore[method-assign]
        Completions.create, "openai", ex.openai_messages, ex.openai_response,
        ex.openai_accumulate, is_async=False)
    AsyncCompletions.create = _instrument(  # type: ignore[method-assign]
        AsyncCompletions.create, "openai", ex.openai_messages, ex.openai_response,
        ex.openai_accumulate, is_async=True)
    _patched.add("openai")
    return True


def _patch_anthropic() -> bool:
    if "anthropic" in _patched:
        return True
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except ImportError:
        return False
    Messages.create = _instrument(  # type: ignore[method-assign]
        Messages.create, "anthropic", ex.anthropic_messages, ex.anthropic_response,
        ex.anthropic_accumulate, is_async=False)
    AsyncMessages.create = _instrument(  # type: ignore[method-assign]
        AsyncMessages.create, "anthropic", ex.anthropic_messages, ex.anthropic_response,
        ex.anthropic_accumulate, is_async=True)
    _patched.add("anthropic")
    return True


def _patch_litellm() -> bool:
    """LiteLLM exposes module-level completion()/acompletion() with OpenAI shapes."""
    if "litellm" in _patched:
        return True
    try:
        import litellm
    except ImportError:
        return False

    def wrap(orig: Callable, is_async: bool) -> Callable:
        inst = _instrument(lambda _self, *a, **k: orig(*a, **k), "litellm",
                           ex.openai_messages, ex.openai_response, ex.openai_accumulate,
                           is_async=is_async)
        if is_async:
            async def acall(*a: Any, **k: Any) -> Any:
                return await inst(None, *a, **k)
            return acall

        def call(*a: Any, **k: Any) -> Any:
            return inst(None, *a, **k)
        return call

    litellm.completion = wrap(litellm.completion, is_async=False)
    if hasattr(litellm, "acompletion"):
        litellm.acompletion = wrap(litellm.acompletion, is_async=True)
    _patched.add("litellm")
    return True


def _patch_gemini() -> bool:
    """google-genai: Models.generate_content(+_stream) and async equivalents."""
    if "gemini" in _patched:
        return True
    try:
        from google.genai.models import AsyncModels, Models
    except ImportError:
        return False

    def inst(orig, is_async, force_stream=False):
        return _instrument(orig, "gemini", ex.gemini_messages, ex.gemini_response,
                           ex.gemini_accumulate, is_async=is_async,
                           messages_key="contents", force_stream=force_stream)

    Models.generate_content = inst(Models.generate_content, False)  # type: ignore[method-assign]
    AsyncModels.generate_content = inst(AsyncModels.generate_content, True)  # type: ignore[method-assign]
    if hasattr(Models, "generate_content_stream"):
        Models.generate_content_stream = inst(  # type: ignore[method-assign]
            Models.generate_content_stream, False, force_stream=True)
    _patched.add("gemini")
    return True
