"""Developer-facing decorators. Minimal input: one decorator per agent stage."""

from __future__ import annotations

import functools
import inspect
from typing import Any, TypeVar, overload
from collections.abc import Callable

from ..runtime import STAGE_REGISTRY, policy_scope, stage_context

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
    # Generators run their body lazily, so the contexts are re-entered around each
    # step instead of held across yields: the consumer's own code between items is
    # not attributed to the stage, and every contextvar token is reset in the step
    # that set it (an async generator may resume in a different context each time).
    # Consequence: send()/throw() and the generator's return value are not forwarded.
    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
            gen = fn(*args, **kwargs)
            try:
                while True:
                    with stage_context(stage_name), policy_scope(stage_name):
                        try:
                            item = next(gen)
                        except StopIteration:
                            return
                    yield item
            finally:
                with stage_context(stage_name), policy_scope(stage_name):
                    gen.close()

        gen_wrapper.__praximetry_stage__ = stage_name  # type: ignore[attr-defined]
        STAGE_REGISTRY[stage_name] = gen_wrapper
        return gen_wrapper

    if inspect.isasyncgenfunction(fn):

        @functools.wraps(fn)
        async def agen_wrapper(*args: Any, **kwargs: Any) -> Any:
            gen = fn(*args, **kwargs)
            try:
                while True:
                    with stage_context(stage_name), policy_scope(stage_name):
                        try:
                            item = await gen.__anext__()
                        except StopAsyncIteration:
                            return
                    yield item
            finally:
                with stage_context(stage_name), policy_scope(stage_name):
                    await gen.aclose()

        agen_wrapper.__praximetry_stage__ = stage_name  # type: ignore[attr-defined]
        STAGE_REGISTRY[stage_name] = agen_wrapper
        return agen_wrapper

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            with stage_context(stage_name), policy_scope(stage_name):
                return await fn(*args, **kwargs)

        awrapper.__praximetry_stage__ = stage_name  # type: ignore[attr-defined]
        STAGE_REGISTRY[stage_name] = awrapper
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with stage_context(stage_name), policy_scope(stage_name):
            return fn(*args, **kwargs)

    wrapper.__praximetry_stage__ = stage_name  # type: ignore[attr-defined]
    STAGE_REGISTRY[stage_name] = wrapper
    return wrapper
