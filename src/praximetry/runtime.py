"""Runtime context: current run, current stage, active experiment overrides."""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Callable, ContextManager

from .config import get_config
from .models import Call, Run
from .store import get_store

_current_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar("run", default=None)
_current_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar("stage", default=None)
# Experiment overrides applied in-flight by instrumentation:
#   {"model": str | None, "prompt_transform": Callable[[list[dict]], list[dict]] | None}
_overrides: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "overrides", default=None
)

# Stage function registry so eval/optimize can re-run stages by name.
STAGE_REGISTRY: dict[str, Callable[..., Any]] = {}

# praximetry has no local optimizer. This is the extension point external tooling
# (praximetry-cloud's applied policy, or anything else) uses to make @stage-decorated
# functions transparently pick up a computed policy in production, with no code
# change. Unset by default: a no-op unless something registers a hook.
_policy_hook: Callable[[str], ContextManager[None]] | None = None


def set_policy_hook(fn: Callable[[str], ContextManager[None]] | None) -> None:
    """Register (or clear, with None) the policy hook applied around every @stage call."""
    global _policy_hook
    _policy_hook = fn


def current_run(create: bool = True) -> Run | None:
    run = _current_run.get()
    if run is None and create:
        cfg = get_config()
        run = Run(project=cfg.project)
        _current_run.set(run)
        get_store().save_run(run)
    return run


def current_stage() -> str | None:
    return _current_stage.get()


def get_overrides() -> dict[str, Any] | None:
    return _overrides.get()


@contextmanager
def run_context(name: str | None = None, experiment_id: str | None = None):
    """Open a fresh Run (used per eval example / experiment trial)."""
    cfg = get_config()
    run = Run(project=cfg.project, name=name, experiment_id=experiment_id)
    get_store().save_run(run)
    token = _current_run.set(run)
    try:
        yield run
    finally:
        run.ended_at = time.time()
        get_store().save_run(run)
        _current_run.reset(token)


@contextmanager
def stage_context(name: str):
    token = _current_stage.set(name)
    try:
        yield
    finally:
        _current_stage.reset(token)


@contextmanager
def override_context(model: str | None = None, prompt_transform=None):
    token = _overrides.set({"model": model, "prompt_transform": prompt_transform})
    try:
        yield
    finally:
        _overrides.reset(token)


@contextmanager
def policy_scope(stage: str):
    """Apply the registered policy hook for `stage`, if any (no-op otherwise)."""
    if _policy_hook is None:
        yield
        return
    with _policy_hook(stage):
        yield


def record_call(call: Call | None = None, **kwargs: Any) -> Call:
    """Persist an LLM call. Used by patchers; also public for manual logging."""
    if not get_config().enabled:
        return call or Call(run_id="disabled", **kwargs)
    if call is None:
        run = current_run()
        kwargs.setdefault("stage", current_stage())
        call = Call(run_id=run.id, **kwargs)
    get_store().save_call(call)
    return call
