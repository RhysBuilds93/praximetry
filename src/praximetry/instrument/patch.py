"""Auto-instrumentation: monkey-patch OpenAI, Anthropic, LiteLLM, and Gemini SDKs.

Covers sync + async and buffered + streaming calls. Experiment overrides
(model swap, prompt transform) are applied in-flight so the optimizer can trial
variants without touching user code.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any
from collections.abc import Callable

from .. import pricing
from .._hooks import Hook
from ..runtime import current_stage, effective_stage, get_overrides, policy_hook, record_call
from ..labeling import derive_label
from ..overrides import apply_prompt_override, load_override, validate_override
from .output import NormalizedOutput
from .providers import PROVIDERS, ProviderSpec
from .wrap import AsyncStreamWrapper, SyncStreamWrapper

logger = logging.getLogger(__name__)

_patched: set[str] = set()
_RAW_RESPONSE_TYPES = {
    ("openai._legacy_response", "LegacyAPIResponse"),
    ("anthropic._legacy_response", "LegacyAPIResponse"),
    ("openai._response", "APIResponse"),
    ("anthropic._response", "APIResponse"),
}

# Pre-flight capture hook (PRA-66): when set, patched create()/acreate()
# wrappers call this INSTEAD of the real `original(...)`, so
# `praximetry.eval.capture` can intercept requests that go through an
# `auto_instrument()`-patched SDK client, not just direct `record_call` use.
# The hook is expected to raise to unwind the call stack before any network
# contact happens.
capture_hook: Hook[Callable[[dict], Any]] = Hook()


def _unwrap_for_parsing(resp: Any) -> Any:
    """Unwrap known SDK raw responses via their cached synchronous parse().

    Match the defining class in the MRO without importing optional SDKs. A
    response model's own `parse` method/field is not an SDK wrapper.
    """
    for cls in type(resp).__mro__:
        if (cls.__module__, cls.__name__) in _RAW_RESPONSE_TYPES:
            return resp.parse()
    return resp


@contextmanager
def capturing(hook: Callable[[dict], Any]):
    """Route the next patched SDK call(s) to `hook` instead of the real API.

    `hook` receives {"provider", "model", "messages", "tools"} and is
    expected to raise to halt execution before `original(...)` is invoked.
    """
    with capture_hook.bound(hook):
        yield


def auto_instrument() -> list[str]:
    """Patch whichever supported SDKs are importable. Returns providers patched."""
    done = []
    for spec in PROVIDERS:
        try:
            if _patch(spec):
                done.append(spec.name)
        except Exception:  # a broken/renamed SDK internal must not crash init()
            logger.debug("auto_instrument: failed to patch %s", spec.name, exc_info=True)
    return done


def _apply_overrides(kwargs: dict[str, Any], messages_key: str, target: str | None = None) -> dict[str, Any]:
    ov = get_overrides()
    if ov is None and policy_hook.fn is None and target is not None:
        try:
            ov = load_override(target)
            if ov is not None:
                validate_override(ov)
                updated = dict(kwargs)
                apply_prompt_override(updated, messages_key, ov)
                kwargs = updated
        except Exception:
            logger.warning("praximetry: could not apply stored override for %s", target, exc_info=True)
            return kwargs
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
    provider: str,
    model: str,
    messages: list,
    out: NormalizedOutput,
    t0: float,
    error: str | None,
    attribution: dict[str, Any] | None = None,
) -> None:
    metadata = {"unpriced_model": True} if pricing.is_unpriced(model) else {}
    attribution = dict(attribution or {})
    metadata.update(attribution.pop("metadata", {}))
    try:
        record_call(
            provider=provider,
            **(attribution or {}),
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
            metadata=metadata,
        )
    except Exception:  # recording must never break or mask the customer's call
        logger.warning("praximetry: failed to record %s call", provider, exc_info=True)



def _make_stream_done(
    provider: str, model: str, messages: list, adapter, t0: float, attribution=None
):
    def on_done(state: dict[str, Any]) -> None:
        state["model"] = model
        out = adapter.finalize_stream(state)
        _record(provider, model, messages, out, t0, state.get("error"), attribution)

    return on_done




def _prepare_call(
    args: tuple[Any, ...], kwargs: dict[str, Any], messages_key: str,
    provider: str, adapter, self_less: bool,
):
    """Apply overrides/capture while preserving module APIs' positional binding."""
    positional = {
        index: key
        for index, key in enumerate(("model", messages_key))
        if self_less and index < len(args) and key not in kwargs
    }
    if positional:
        kwargs = dict(kwargs)
        for index, key in positional.items():
            kwargs[key] = args[index]
    value = kwargs.get(messages_key)
    if value is not None and hasattr(value, "__next__"):
        kwargs[messages_key] = list(value)
    config = kwargs.get("config") if provider == "gemini" else None
    if config is not None and callable(getattr(config, "model_dump", None)):
        config = config.model_dump(exclude_none=True)
    config_supported = config is None or isinstance(config, dict)
    raw_tools = config.get("tools", []) if isinstance(config, dict) else kwargs.get("tools", [])
    tools = []
    tools_complete = config_supported
    if raw_tools is not None:
        if not isinstance(raw_tools, (list, tuple)):
            tools_complete = False
        else:
            for tool in raw_tools:
                if callable(getattr(tool, "model_dump", None)):
                    tool = tool.model_dump(exclude_none=True)
                if isinstance(tool, dict):
                    tools.append(tool)
                else:
                    tools_complete = False
    try:
        original_messages = adapter.get_messages(kwargs)
        request_complete = tools_complete
    except Exception:
        original_messages = []
        request_complete = False
    if provider == "gemini":
        request_complete = request_complete and isinstance(kwargs.get("contents"), str)
        if isinstance(config, dict) and isinstance(config.get("system_instruction"), str):
            if not any(m.get("role") == "system" for m in original_messages):
                original_messages.insert(0, {"role": "system", "content": config["system_instruction"]})
        elif isinstance(config, dict) and config.get("system_instruction") is not None:
            request_complete = False
    identity_messages = original_messages
    if isinstance(kwargs.get("system"), str):
        identity_messages = [
            {"role": "system", "content": kwargs["system"]},
            *original_messages,
        ]
    derived = derive_label(
        identity_messages, tools, kwargs.get("model", "unknown")
    )
    stage = current_stage() or derived
    target = effective_stage() or derived
    kwargs = _apply_overrides(kwargs, messages_key, target)
    model = kwargs.get("model", "unknown")
    try:
        messages = adapter.get_messages(kwargs)
        if provider == "gemini":
            outbound_config = kwargs.get("config")
            if isinstance(outbound_config, dict):
                instruction = outbound_config.get("system_instruction")
                if isinstance(instruction, str) and not any(m.get("role") == "system" for m in messages):
                    messages.insert(0, {"role": "system", "content": instruction})
    except Exception:  # unrecognised kwargs shape: still make the call, just unrecorded
        logger.warning("praximetry: could not read %s messages", provider, exc_info=True)
        messages = []
        request_complete = False
    attribution = {
        "derived_label": derived,
        "stage": stage,
        "target": target,
        "tools": tools,
        "metadata": {"request_complete": request_complete},
    }
    capture_payload = (
        {
            "provider": provider,
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        if capture_hook.fn is not None
        else None
    )
    if positional:
        args = tuple(
            kwargs.pop(positional[index]) if index in positional else value
            for index, value in enumerate(args)
        )
    return args, kwargs, model, messages, capture_payload, attribution


def _finish_call(
    provider: str, model: str, messages: list, adapter, resp: Any, t0: float, attribution=None
) -> Any:
    try:
        out = adapter.parse_response(_unwrap_for_parsing(resp), model)
    except Exception:  # the response is already paid for; hand it back untouched
        logger.warning("praximetry: could not parse %s response", provider, exc_info=True)
        return resp
    _record(provider, model, messages, out, t0, None, attribution)
    return resp


def _instrument(
    original: Callable,
    provider: str,
    adapter,
    is_async: bool,
    messages_key: str = "messages",
    force_stream: bool = False,
    inject_stream_usage: bool = False,
    self_less: bool = False,
) -> Callable:
    """Build a patched create() wrapping `original` for one provider/sync-ness."""
    if is_async:

        async def acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
            args, kwargs, model, messages, capture_payload, attribution = _prepare_call(
                args, kwargs, messages_key, provider, adapter, self_less
            )
            if capture_payload is not None:
                return capture_hook.fn(capture_payload)
            t0 = time.perf_counter()
            if force_stream or kwargs.get("stream"):
                if inject_stream_usage:
                    kwargs.setdefault("stream_options", {"include_usage": True})
                try:
                    resp = await original(self, *args, **kwargs)
                except Exception as e:
                    _record(provider, model, messages, NormalizedOutput(), t0, str(e), attribution)
                    raise
                return AsyncStreamWrapper(
                    resp,
                    adapter.accumulate,
                    _make_stream_done(provider, model, messages, adapter, t0, attribution),
                )


            try:
                resp = await original(self, *args, **kwargs)
            except Exception as e:
                _record(provider, model, messages, NormalizedOutput(), t0, str(e), attribution)
                raise
            return _finish_call(provider, model, messages, adapter, resp, t0, attribution)

        acreate._praximetry_patched = True  # type: ignore[attr-defined]
        return acreate

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        args, kwargs, model, messages, capture_payload, attribution = _prepare_call(
            args, kwargs, messages_key, provider, adapter, self_less
        )
        if capture_payload is not None:
            return capture_hook.fn(capture_payload)
        t0 = time.perf_counter()
        if force_stream or kwargs.get("stream"):
            if inject_stream_usage:
                kwargs.setdefault("stream_options", {"include_usage": True})
            try:
                resp = original(self, *args, **kwargs)
            except Exception as e:
                _record(provider, model, messages, NormalizedOutput(), t0, str(e), attribution)
                raise
            return SyncStreamWrapper(
                resp, adapter.accumulate, _make_stream_done(provider, model, messages, adapter, t0, attribution)
            )


        try:
            resp = original(self, *args, **kwargs)
        except Exception as e:
            _record(provider, model, messages, NormalizedOutput(), t0, str(e), attribution)
            raise
        return _finish_call(provider, model, messages, adapter, resp, t0, attribution)

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

        acall._praximetry_patched = True  # type: ignore[attr-defined]
        return acall

    def call(*a: Any, **k: Any) -> Any:
        return inst(None, *a, **k)

    call._praximetry_patched = True  # type: ignore[attr-defined]
    return call


def _patch(spec: ProviderSpec) -> bool:
    """Apply one provider's PatchTarget table, generically."""
    if spec.name in _patched:
        return True
    try:
        sync_host, async_host = spec.owner()
    except ImportError:
        return False

    replacements = []
    for target in spec.targets:
        host = async_host if target.is_async else sync_host
        if target.optional and not hasattr(host, target.attr):
            continue
        original = getattr(host, target.attr)
        if getattr(original, "_praximetry_patched", False):
            continue
        wrapped = _instrument(
            _self_less(original) if target.self_less else original,
            spec.name,
            spec.adapter,
            is_async=target.is_async,
            messages_key=spec.messages_key,
            force_stream=target.force_stream,
            inject_stream_usage=spec.inject_stream_usage,
            self_less=target.self_less,
        )
        if target.self_less:
            wrapped = _self_less_caller(wrapped, target.is_async)
        replacements.append((host, target.attr, original, wrapped))

    applied = []
    try:
        for host, attr, original, wrapped in replacements:
            applied.append((host, attr, original))
            setattr(host, attr, wrapped)
    except BaseException:
        for host, attr, original in reversed(applied):
            setattr(host, attr, original)
        raise

    _patched.add(spec.name)
    return True
