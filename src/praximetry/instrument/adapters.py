"""OutputAdapter: pure, per-provider SDK object -> NormalizedOutput parsing.

Kept separate from capture mechanism (patch.py / capture.py) so adapters stay
unit-testable against real SDK response objects without any monkeypatching,
same convention the old extractors.py used.
"""
from __future__ import annotations

import json
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


class OutputAdapter(ABC):
    name: str

    @abstractmethod
    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]: ...

    @abstractmethod
    def parse_response(self, resp: Any, model: str) -> NormalizedOutput: ...

    @abstractmethod
    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput: ...


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
            ToolCall(id=tc.id, name=tc.function.name,
                     arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (_g(message, "tool_calls", default=[]) or [])
        ]
        output_text, reasoning_text = split_embedded_reasoning(text, model)
        return NormalizedOutput(output_text=output_text, reasoning_text=reasoning_text,
                                 tool_calls=tool_calls, tokens_in=tin, tokens_out=tout)

    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None:
        state.setdefault("text", "")
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = _g(choices[0], "delta", "content", default="") or ""
            state["text"] += delta
            state["tout"] = state.get("tout", 0) + (1 if delta else 0)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            state["tin"] = getattr(usage, "prompt_tokens", state.get("tin", 0)) or state.get("tin", 0)
            ct = getattr(usage, "completion_tokens", None)
            if ct is not None:
                state["tout"] = ct

    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput:
        model = state.get("model", "")
        output_text, reasoning_text = split_embedded_reasoning(state.get("text", ""), model)
        return NormalizedOutput(output_text=output_text, reasoning_text=reasoning_text,
                                 tokens_in=state.get("tin", 0), tokens_out=state.get("tout", 0))


ADAPTERS: dict[str, OutputAdapter] = {
    "openai": OpenAIAdapter(),
    "litellm": OpenAIAdapter(),
}
