"""Declarative table of provider SDK patch targets.

Each `ProviderSpec` describes one SDK's shape (class-level vs module-level
methods, sync/async hosts, extra force_stream wiring) so `patch._patch()` can
apply the same wrap/record/mark-patched skeleton to all of them generically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from .adapters import ADAPTERS, OutputAdapter


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
    adapter: OutputAdapter
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
        adapter=ADAPTERS["openai"],
    ),
    ProviderSpec(
        name="anthropic",
        owner=_anthropic_owner,
        targets=[
            PatchTarget(attr="create", is_async=False),
            PatchTarget(attr="create", is_async=True),
        ],
        adapter=ADAPTERS["anthropic"],
    ),
    ProviderSpec(
        name="litellm",
        owner=_litellm_owner,
        targets=[
            PatchTarget(attr="completion", is_async=False, self_less=True),
            PatchTarget(attr="acompletion", is_async=True, self_less=True, optional=True),
        ],
        adapter=ADAPTERS["litellm"],
    ),
    ProviderSpec(
        name="gemini",
        owner=_gemini_owner,
        targets=[
            PatchTarget(attr="generate_content", is_async=False),
            PatchTarget(attr="generate_content", is_async=True),
            PatchTarget(
                attr="generate_content_stream", is_async=False, force_stream=True, optional=True
            ),
        ],
        adapter=ADAPTERS["gemini"],
        messages_key="contents",
    ),
]
