"""Shared LLM call for the example workflows -- ordinary customer code, not a
special harness.

Every workflow calls `px.init(auto_instrument_sdks=True)` (the
default), which monkey-patches `openai.resources.chat.completions.Completions
.create` for the whole process. So the `client.chat.completions.create(...)`
call below is recorded, priced, and timed the exact same way it would be in
your own agent code -- there's no separate recording path here, and any
active experiment override (model swap, prompt transform) is applied to it
the same way it would be to yours.

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

from openai import OpenAI

from praximetry.instrument.reasoning_patterns import split_embedded_reasoning

# Change the fallback below to match whatever your AI_ENDPOINT serves, e.g.
# "gpt-4o-mini" for OpenAI, "claude-haiku-4-5-20251001" for an
# Anthropic-compatible endpoint, or your own model name for a self-hosted
# server. Read live (not cached at import) so PRAXIMETRY_EXAMPLE_MODEL can be
# set per-run without reimporting.
_DEFAULT_MODEL_FALLBACK = "gpt-4o-mini"

# A handful of workflows (support_triage's `classify`, tau_retail's
# `plan_action`) deliberately call an oversized/expensive model for a cheap
# task, on purpose, so the traffic detector has a real oversized_model
# anti-pattern in the recorded traffic to catch. Point this at a pricier
# model your AI_ENDPOINT serves.
_PREMIUM_MODEL_FALLBACK = "gpt-4o"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("AI_ENDPOINT", "https://api.openai.com/v1"),
            api_key=os.environ["AI_API_KEY"],
        )
    return _client


def default_model() -> str:
    return os.environ.get("PRAXIMETRY_EXAMPLE_MODEL", _DEFAULT_MODEL_FALLBACK)


def premium_model() -> str:
    return os.environ.get("PRAXIMETRY_EXAMPLE_PREMIUM_MODEL", _PREMIUM_MODEL_FALLBACK)


def real_chat(model: str, messages: list[dict]) -> str:
    """Call your configured provider. Same call a stage in your own agent makes."""
    completion = _get_client().chat.completions.create(model=model, messages=messages)
    content = completion.choices[0].message.content
    output_text, _reasoning_text = split_embedded_reasoning(content, model)
    return output_text
