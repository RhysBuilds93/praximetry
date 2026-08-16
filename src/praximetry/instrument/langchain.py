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
from .adapters import ADAPTERS
from .capture import CaptureMechanism
from .output import NormalizedOutput


class LangChainCallbackCapture(BaseCallbackHandler, CaptureMechanism):
    name = "langchain"

    def __init__(self) -> None:
        self._starts: dict[UUID, tuple[list[dict], float]] = {}

    def install(self, adapter: Any) -> bool:
        return True  # this class IS the handler; caller attaches it via `callbacks=[...]`

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        messages = [{"role": "user", "content": p} for p in prompts]
        self._starts[run_id] = (messages, time.perf_counter())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        messages, t0 = self._starts.pop(run_id, ([], time.perf_counter()))
        model = (response.llm_output or {}).get("model_name", "unknown")
        text = response.generations[0][0].text if response.generations and response.generations[0] else ""
        adapter = ADAPTERS.get(_provider_for_model(model))
        if adapter is not None:
            out = adapter.parse_response(_fake_openai_response(text), model)
        else:
            out = NormalizedOutput(output_text=text)
        record_call(
            provider="langchain", model=model, messages=messages,
            output_text=out.output_text, reasoning_text=out.reasoning_text,
            tool_calls=[tc.model_dump() for tc in out.tool_calls],
            structured_output=out.structured_output,
            content_parts=[cp.model_dump() for cp in out.content_parts],
            input_tokens=out.tokens_in, output_tokens=out.tokens_out,
            cost_usd=pricing.cost_usd(model, out.tokens_in, out.tokens_out),
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


def _provider_for_model(model: str) -> str:
    if model.startswith("gpt") or model.startswith("openai"):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    return "openai"


def _fake_openai_response(text: str) -> Any:
    from types import SimpleNamespace as NS
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))],
              usage=NS(prompt_tokens=0, completion_tokens=0))
