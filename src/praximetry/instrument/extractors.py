"""Provider-specific extraction of messages, text, and token usage.

Pure functions over SDK request kwargs / response / stream-chunk objects, using
attribute access so they work against real SDK objects and test doubles alike.
Kept separate from patching so they're unit-testable without monkeypatching.
"""
from __future__ import annotations

from typing import Any


def _g(obj: Any, *path: str, default: Any = None) -> Any:
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return default
    return obj


# -- OpenAI (also LiteLLM, which returns OpenAI-shaped objects) --------------

def openai_messages(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    return list(kwargs.get("messages", []))


def openai_response(resp: Any) -> tuple[str, int, int]:
    text = ""
    choices = getattr(resp, "choices", None)
    if choices:
        text = _g(choices[0], "message", "content", default="") or ""
    tin = _g(resp, "usage", "prompt_tokens", default=0) or 0
    tout = _g(resp, "usage", "completion_tokens", default=0) or 0
    return text, tin, tout


def openai_accumulate(chunk: Any, state: dict[str, Any]) -> None:
    choices = getattr(chunk, "choices", None)
    if choices:
        delta = _g(choices[0], "delta", "content", default="") or ""
        state["text"] += delta
        state["tout"] += 1 if delta else 0  # fallback count; overwritten if usage present
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        state["tin"] = getattr(usage, "prompt_tokens", state.get("tin", 0)) or state.get("tin", 0)
        ct = getattr(usage, "completion_tokens", None)
        if ct is not None:
            state["tout"] = ct


# -- Anthropic --------------------------------------------------------------

def anthropic_messages(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = list(kwargs.get("messages", []))
    system = kwargs.get("system")
    if system:
        text = system if isinstance(system, str) else str(system)
        msgs = [{"role": "system", "content": text}] + msgs
    return msgs


def anthropic_response(resp: Any) -> tuple[str, int, int]:
    text = ""
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "") or ""
    tin = _g(resp, "usage", "input_tokens", default=0) or 0
    tout = _g(resp, "usage", "output_tokens", default=0) or 0
    return text, tin, tout


def anthropic_accumulate(event: Any, state: dict[str, Any]) -> None:
    etype = getattr(event, "type", "")
    if etype == "message_start":
        state["tin"] = _g(event, "message", "usage", "input_tokens", default=state.get("tin", 0))
    elif etype == "content_block_delta":
        state["text"] += _g(event, "delta", "text", default="") or ""
    elif etype == "message_delta":
        out = _g(event, "usage", "output_tokens", default=None)
        if out is not None:
            state["tout"] = out


# -- Google Gemini (google-genai) -------------------------------------------

def gemini_messages(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    contents = kwargs.get("contents")
    system = kwargs.get("config")
    msgs = [{"role": "user", "content": contents if isinstance(contents, str) else str(contents)}]
    sys_instr = _g(system, "system_instruction") if system is not None else None
    if isinstance(sys_instr, str):
        msgs = [{"role": "system", "content": sys_instr}] + msgs
    return msgs


def gemini_response(resp: Any) -> tuple[str, int, int]:
    text = getattr(resp, "text", "") or ""
    tin = _g(resp, "usage_metadata", "prompt_token_count", default=0) or 0
    tout = _g(resp, "usage_metadata", "candidates_token_count", default=0) or 0
    return text, tin, tout


def gemini_accumulate(chunk: Any, state: dict[str, Any]) -> None:
    state["text"] += getattr(chunk, "text", "") or ""
    tin = _g(chunk, "usage_metadata", "prompt_token_count", default=None)
    tout = _g(chunk, "usage_metadata", "candidates_token_count", default=None)
    if tin:
        state["tin"] = tin
    if tout:
        state["tout"] = tout
