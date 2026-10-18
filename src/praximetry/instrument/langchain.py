"""LangChain integration via its BaseCallbackHandler event stream, not
monkeypatching — LangChain exposes no client.create() method to patch. This
is the proof that CaptureMechanism generalizes beyond SDK-method-patching.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ..runtime import record_call
from .. import pricing
from .capture import CaptureMechanism
from .reasoning_patterns import split_embedded_reasoning

_MAX_PENDING_RUNS = 1024


def _model_name(serialized: dict, kwargs: dict) -> str:
    # Invocation parameters reflect overrides; serialized kwargs may hold defaults.
    for source in (
        kwargs.get("invocation_params") or {}, kwargs, serialized.get("kwargs") or {}, serialized
    ):
        model = source.get("model_name") or source.get("model")
        if isinstance(model, str) and model:
            return model
    return "unknown"


class LangChainCallbackCapture(BaseCallbackHandler, CaptureMechanism):
    """Capture LLM callbacks, retaining at most 1024 unfinished starts per handler.

    Oldest unfinished starts are evicted; late completions still record, but
    without the evicted prompt, model, or original start time.
    """

    name = "langchain"

    def __init__(self) -> None:
        self._starts: dict[UUID, tuple[list[dict], float, str]] = {}
        self._starts_lock = Lock()

    def install(self, adapter: Any) -> bool:
        """No-op: this class IS the handler, attached via `callbacks=[...]`.

        Returns:
            Always True.
        """
        return True

    def on_llm_start(
        self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        messages = [{"role": "user", "content": p} for p in prompts]
        start = (messages, time.perf_counter(), _model_name(serialized or {}, kwargs))
        with self._starts_lock:
            self._starts.pop(run_id, None)
            if len(self._starts) >= _MAX_PENDING_RUNS:
                self._starts.pop(next(iter(self._starts)))
            self._starts[run_id] = start

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._starts_lock:
            start = self._starts.pop(run_id, None)
        messages, t0, start_model = start or ([], time.perf_counter(), "unknown")
        llm_output = response.llm_output or {}
        model = llm_output.get("model_name") or llm_output.get("model") or start_model
        text = (
            response.generations[0][0].text
            if response.generations and response.generations[0]
            else ""
        )
        output_text, reasoning_text = split_embedded_reasoning(text, model)
        usage = llm_output.get("token_usage", {}) or {}
        tin = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        tout = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        record_call(
            provider="langchain",
            model=model,
            messages=messages,
            output_text=output_text,
            reasoning_text=reasoning_text,
            tool_calls=[],
            structured_output=None,
            content_parts=[],
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=pricing.cost_usd(model, tin, tout),
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=None,
            metadata={"unpriced_model": True} if pricing.is_unpriced(model) else {},
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._starts_lock:
            start = self._starts.pop(run_id, None)
        messages, t0, model = start or ([], time.perf_counter(), "unknown")
        if model == "unknown":
            model = _model_name({}, kwargs)
        record_call(
            provider="langchain",
            model=model,
            messages=messages,
            output_text="",
            reasoning_text="",
            tool_calls=[],
            structured_output=None,
            content_parts=[],
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=str(error),
        )


def install_langchain_capture() -> bool:
    """LangChain has no client to monkeypatch, so there's nothing to install
    globally — attach `LangChainCallbackCapture()` via `callbacks=[...]` on
    the LLM/chain you want captured.

    Returns:
        Always True.
    """
    return True
