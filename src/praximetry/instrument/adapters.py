"""OutputAdapter: pure, per-provider SDK object -> NormalizedOutput parsing.

Kept separate from capture mechanism (patch.py / capture.py) so adapters stay
unit-testable against real SDK response objects without any monkeypatching.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any

from .output import NormalizedOutput, ToolCall
from .reasoning_patterns import split_embedded_reasoning


def _g(obj: Any, *path: str, default: Any = None) -> Any:
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return default
    return obj


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    """Some reasoning models (e.g. gpt-oss) occasionally emit malformed JSON
    in tool_call.function.arguments. This is instrumentation code -- a parse
    failure here must not crash the caller's actual LLM call.
    """
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_unparsed": raw}


class OutputAdapter(ABC):
    """Pure per-provider parsing: SDK request/response objects -> NormalizedOutput."""

    name: str

    @abstractmethod
    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract the request's messages from the SDK call's kwargs."""

    @abstractmethod
    def parse_response(self, resp: Any, model: str) -> NormalizedOutput:
        """Parse one complete, non-streaming SDK response object."""

    @abstractmethod
    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None:
        """Fold one streaming chunk into `state`, mutated in place across the stream."""

    @abstractmethod
    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput:
        """Build the NormalizedOutput once a stream accumulated via `accumulate` is exhausted."""


class OpenAIAdapter(OutputAdapter):
    name = "openai"

    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        return list(kwargs.get("messages", []))

    def parse_response(self, resp: Any, model: str) -> NormalizedOutput:
        tin = _g(resp, "usage", "prompt_tokens", default=0) or 0
        tout = _g(resp, "usage", "completion_tokens", default=0) or 0
        choices = getattr(resp, "choices", None)
        if not choices:
            return NormalizedOutput(tokens_in=tin, tokens_out=tout)
        message = _g(choices[0], "message")
        text = _g(message, "content", default="") or ""
        tool_calls = [
            ToolCall(
                id=tc.id, name=tc.function.name, arguments=_parse_tool_args(tc.function.arguments)
            )
            for tc in (_g(message, "tool_calls", default=[]) or [])
        ]
        output_text, reasoning_text = split_embedded_reasoning(text, model)
        return NormalizedOutput(
            output_text=output_text,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
            tokens_in=tin,
            tokens_out=tout,
        )

    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None:
        state.setdefault("text", "")
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = _g(choices[0], "delta", "content", default="") or ""
            state["text"] += delta
            state["tout"] = state.get("tout", 0) + (1 if delta else 0)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            state["tin"] = getattr(usage, "prompt_tokens", state.get("tin", 0)) or state.get(
                "tin", 0
            )
            ct = getattr(usage, "completion_tokens", None)
            if ct is not None:
                state["tout"] = ct

    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput:
        model = state.get("model", "")
        output_text, reasoning_text = split_embedded_reasoning(state.get("text", ""), model)
        return NormalizedOutput(
            output_text=output_text,
            reasoning_text=reasoning_text,
            tokens_in=state.get("tin", 0),
            tokens_out=state.get("tout", 0),
        )


class AnthropicAdapter(OutputAdapter):
    name = "anthropic"

    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        msgs = list(kwargs.get("messages", []))
        system = kwargs.get("system")
        if system:
            text = system if isinstance(system, str) else str(system)
            msgs = [{"role": "system", "content": text}] + msgs
        return msgs

    def parse_response(self, resp: Any, model: str) -> NormalizedOutput:
        output_text = ""
        reasoning_text = ""
        tool_calls: list[ToolCall] = []
        for block in getattr(resp, "content", None) or []:
            btype = getattr(block, "type", "")
            if btype == "text":
                output_text += getattr(block, "text", "") or ""
            elif btype == "thinking":
                reasoning_text += getattr(block, "thinking", "") or ""
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        tin = _g(resp, "usage", "input_tokens", default=0) or 0
        tout = _g(resp, "usage", "output_tokens", default=0) or 0
        return NormalizedOutput(
            output_text=output_text,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
            tokens_in=tin,
            tokens_out=tout,
        )

    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None:
        state.setdefault("text", "")
        state.setdefault("reasoning", "")
        etype = getattr(chunk, "type", "")
        if etype == "message_start":
            state["tin"] = _g(
                chunk, "message", "usage", "input_tokens", default=state.get("tin", 0)
            )
        elif etype == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            dtype = getattr(delta, "type", "")
            if dtype == "text_delta":
                state["text"] += getattr(delta, "text", "") or ""
            elif dtype == "thinking_delta":
                state["reasoning"] += getattr(delta, "thinking", "") or ""
        elif etype == "message_delta":
            out = _g(chunk, "usage", "output_tokens", default=None)
            if out is not None:
                state["tout"] = out

    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput:
        return NormalizedOutput(
            output_text=state.get("text", ""),
            reasoning_text=state.get("reasoning", ""),
            tokens_in=state.get("tin", 0),
            tokens_out=state.get("tout", 0),
        )


ADAPTERS: dict[str, OutputAdapter] = {
    "openai": OpenAIAdapter(),
    "litellm": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
}


class GeminiAdapter(OutputAdapter):
    name = "gemini"

    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        contents = kwargs.get("contents")
        config = kwargs.get("config")
        msgs = [
            {"role": "user", "content": contents if isinstance(contents, str) else str(contents)}
        ]
        sys_instr = _g(config, "system_instruction") if config is not None else None
        if isinstance(sys_instr, str):
            msgs = [{"role": "system", "content": sys_instr}] + msgs
        return msgs

    def parse_response(self, resp: Any, model: str) -> NormalizedOutput:
        text = ""
        reasoning_text = ""
        tool_calls: list[ToolCall] = []
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            for part in _g(candidates[0], "content", "parts", default=[]) or []:
                if getattr(part, "text", None):
                    if _g(part, "thought", default=False):
                        reasoning_text += part.text
                    else:
                        text += part.text
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append(
                        ToolCall(
                            id=uuid.uuid4().hex[:16], name=fc.name, arguments=dict(fc.args or {})
                        )
                    )
        tin = _g(resp, "usage_metadata", "prompt_token_count", default=0) or 0
        tout = _g(resp, "usage_metadata", "candidates_token_count", default=0) or 0
        return NormalizedOutput(
            output_text=text,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
            tokens_in=tin,
            tokens_out=tout,
        )

    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None:
        state.setdefault("text", "")
        state["text"] += getattr(chunk, "text", "") or ""
        tin = _g(chunk, "usage_metadata", "prompt_token_count", default=None)
        tout = _g(chunk, "usage_metadata", "candidates_token_count", default=None)
        if tin:
            state["tin"] = tin
        if tout:
            state["tout"] = tout

    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput:
        return NormalizedOutput(
            output_text=state.get("text", ""),
            tokens_in=state.get("tin", 0),
            tokens_out=state.get("tout", 0),
        )


ADAPTERS["gemini"] = GeminiAdapter()
