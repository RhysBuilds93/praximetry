"""Declarative table of provider SDK patch targets.

Each `ProviderSpec` describes one SDK's shape (class-level vs module-level
methods, sync/async hosts, extra force_stream wiring) so `patch._patch()` can
apply the same wrap/record/mark-patched skeleton to all of them generically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import extractors as ex


@dataclass
class PatchTarget:
    """One attribute to monkeypatch on a spec's sync or async host.

    attr: the attribute name to replace (e.g. "create", "completion").
    is_async: whether to resolve against `owner()`'s async host and build
        the `_instrument()` async wrapper.
    force_stream: always treat the call as a stream, regardless of the
        `stream` kwarg (used for gemini's `generate_content_stream`).
    optional: skip via hasattr() if the host doesn't have this attribute,
        rather than erroring (litellm's `acompletion` isn't always present).
    self_less: the underlying callable takes no `self` (module-level
        functions like `litellm.completion`), so it needs the self-less
        wrapping technique instead of class-method wrapping.
    """
    attr: str
    is_async: bool = False
    force_stream: bool = False
    optional: bool = False
    self_less: bool = False


@dataclass
class ProviderSpec:
    name: str
    owner: Callable[[], tuple[Any, Any]]  # () -> (sync_host, async_host); raises ImportError
    targets: list[PatchTarget]
    get_messages: Callable
    response_extract: Callable
    accumulate: Callable
    messages_key: str = "messages"


def _openai_owner() -> tuple[Any, Any]:
    from openai.resources.chat.completions import AsyncCompletions, Completions
    return Completions, AsyncCompletions


def _anthropic_owner() -> tuple[Any, Any]:
    from anthropic.resources.messages import AsyncMessages, Messages
    return Messages, AsyncMessages


def _litellm_owner() -> tuple[Any, Any]:
    import litellm
    return litellm, litellm


def _gemini_owner() -> tuple[Any, Any]:
    from google.genai.models import AsyncModels, Models
    return Models, AsyncModels


PROVIDERS: list[ProviderSpec] = [
    ProviderSpec(
        name="openai",
        owner=_openai_owner,
        targets=[
            PatchTarget(attr="create", is_async=False),
            PatchTarget(attr="create", is_async=True),
        ],
        get_messages=ex.openai_messages,
        response_extract=ex.openai_response,
        accumulate=ex.openai_accumulate,
    ),
    ProviderSpec(
        name="anthropic",
        owner=_anthropic_owner,
        targets=[
            PatchTarget(attr="create", is_async=False),
            PatchTarget(attr="create", is_async=True),
        ],
        get_messages=ex.anthropic_messages,
        response_extract=ex.anthropic_response,
        accumulate=ex.anthropic_accumulate,
    ),
    ProviderSpec(
        name="litellm",
        owner=_litellm_owner,
        targets=[
            PatchTarget(attr="completion", is_async=False, self_less=True),
            PatchTarget(attr="acompletion", is_async=True, self_less=True, optional=True),
        ],
        get_messages=ex.openai_messages,
        response_extract=ex.openai_response,
        accumulate=ex.openai_accumulate,
    ),
    ProviderSpec(
        name="gemini",
        owner=_gemini_owner,
        targets=[
            PatchTarget(attr="generate_content", is_async=False),
            PatchTarget(attr="generate_content", is_async=True),
            PatchTarget(attr="generate_content_stream", is_async=False,
                        force_stream=True, optional=True),
        ],
        get_messages=ex.gemini_messages,
        response_extract=ex.gemini_response,
        accumulate=ex.gemini_accumulate,
        messages_key="contents",
    ),
]
