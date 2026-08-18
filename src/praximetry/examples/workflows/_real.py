"""Shared LLM setup for the example workflows -- ordinary customer code, not a
special harness. Every workflow calls `px.init(auto_instrument_sdks=True)`
(the default), which patches the openai SDK client that `ChatOpenAI` uses
internally, so calls made through `chat_model()` below are recorded, priced,
and timed the same way they would be in your own agent code.

Setup: point these two env vars at your own provider before running a
workflow.

    AI_ENDPOINT   base URL of any OpenAI-compatible chat completions API,
                  e.g. https://api.openai.com/v1 (OpenAI), an
                  OpenAI-compatible endpoint your provider exposes, or a
                  local server (Ollama, vLLM, ...) at http://localhost:PORT/v1
    AI_API_KEY    that provider's API key

Then either set PRAXIMETRY_EXAMPLE_MODEL (and PRAXIMETRY_EXAMPLE_PREMIUM_MODEL,
where used) or edit the fallbacks below to a model id your AI_ENDPOINT
actually serves -- there's no cross-provider default that works everywhere.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from praximetry.instrument.reasoning_patterns import split_embedded_reasoning

_DEFAULT_MODEL_FALLBACK = "gpt-4o-mini"

# A handful of workflows (support_triage's `classify`, tau_retail's
# `plan_action`) deliberately call an oversized/expensive model for a cheap
# task, on purpose, so the traffic detector has a real oversized_model
# anti-pattern in the recorded traffic to catch. Point this at a pricier
# model your AI_ENDPOINT serves.
_PREMIUM_MODEL_FALLBACK = "gpt-4o"


def default_model() -> str:
    return os.environ.get("PRAXIMETRY_EXAMPLE_MODEL", _DEFAULT_MODEL_FALLBACK)


def premium_model() -> str:
    return os.environ.get("PRAXIMETRY_EXAMPLE_PREMIUM_MODEL", _PREMIUM_MODEL_FALLBACK)


def chat_model(name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=name,
        base_url=os.environ.get("AI_ENDPOINT", "https://api.openai.com/v1"),
        api_key=os.environ["AI_API_KEY"],
    )


def clean_content(message: AIMessage, model: str) -> str:
    """Strip a leaked <reasoning>...</reasoning> block some providers (e.g.
    gpt-oss models via Bedrock's OpenAI-compatible endpoint) prepend to content."""
    text, _reasoning = split_embedded_reasoning(message.content, model)
    return text
