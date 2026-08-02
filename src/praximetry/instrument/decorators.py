"""Developer-facing decorators. Minimal input: one decorator per agent stage."""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar, overload

from ..runtime import STAGE_REGISTRY, stage_context

F = TypeVar("F", bound=Callable[..., Any])


@overload
def stage(fn: F) -> F: ...
@overload
def stage(name: str | None = None) -> Callable[[F], F]: ...


def stage(name: Any = None) -> Any:
    """Mark a function as an agent stage.

    Usage:
        @praximetry.stage                # stage name = function name
        @praximetry.stage("summarize")   # explicit name

    All LLM calls made inside are attributed to this stage, and the function is
    registered by name so external tooling (e.g. praximetry-cloud's evals and
    optimizer) can re-run it.
    """
    if callable(name):  # bare @stage
        return _wrap(name, name.__name__)

    def deco(fn: F) -> F:
        return _wrap(fn, name or fn.__name__)

    return deco


def _wrap(fn: Callable[..., Any], stage_name: str) -> Any:
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            with stage_context(stage_name):
                return await fn(*args, **kwargs)
        awrapper.__praximetry_stage__ = stage_name  # type: ignore[attr-defined]
        STAGE_REGISTRY[stage_name] = awrapper
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with stage_context(stage_name):
            return fn(*args, **kwargs)

    wrapper.__praximetry_stage__ = stage_name  # type: ignore[attr-defined]
    STAGE_REGISTRY[stage_name] = wrapper
    return wrapper
