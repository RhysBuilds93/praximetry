"""LangChain integration via its BaseCallbackHandler event stream, not
monkeypatching — LangChain exposes no client.create() method to patch. This
is the proof that CaptureMechanism generalizes beyond SDK-method-patching.
"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ..runtime import record_call
from .. import pricing
from .capture import CaptureMechanism
from .reasoning_patterns import split_embedded_reasoning


class LangChainCallbackCapture(BaseCallbackHandler, CaptureMechanism):
    name = "langchain"

    def __init__(self) -> None:
        self._starts: dict[UUID, tuple[list[dict], float]] = {}

    def install(self, adapter: Any) -> bool:
        """No-op: this class IS the handler, attached via `callbacks=[...]`.

        Returns:
            Always True.
        """
        return True

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        messages = [{"role": "user", "content": p} for p in prompts]
        self._starts[run_id] = (messages, time.perf_counter())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        messages, t0 = self._starts.pop(run_id, ([], time.perf_counter()))
        llm_output = response.llm_output or {}
        model = llm_output.get("model_name", "unknown")
        text = response.generations[0][0].text if response.generations and response.generations[0] else ""
        output_text, reasoning_text = split_embedded_reasoning(text, model)
        usage = llm_output.get("token_usage", {}) or {}
        tin = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        tout = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        record_call(
            provider="langchain", model=model, messages=messages,
            output_text=output_text, reasoning_text=reasoning_text,
            tool_calls=[], structured_output=None, content_parts=[],
            input_tokens=tin, output_tokens=tout,
            cost_usd=pricing.cost_usd(model, tin, tout),
            latency_ms=(time.perf_counter() - t0) * 1000, error=None,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        messages, t0 = self._starts.pop(run_id, ([], time.perf_counter()))
        record_call(
            provider="langchain", model="unknown", messages=messages,
            output_text="", reasoning_text="", tool_calls=[], structured_output=None,
            content_parts=[], input_tokens=0, output_tokens=0, cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000, error=str(error),
        )


def install_langchain_capture() -> bool:
    """LangChain has no client to monkeypatch, so there's nothing to install
    globally — attach `LangChainCallbackCapture()` via `callbacks=[...]` on
    the LLM/chain you want captured.

    Returns:
        Always True.
    """
    return True
