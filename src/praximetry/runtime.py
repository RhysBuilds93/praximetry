"""Runtime context: current run, current stage, active experiment overrides."""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Callable, ContextManager

from . import cloud_sync
from .config import get_config
from .models import Call, Run
from .store import get_store

_current_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar("run", default=None)
_stage_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "stage_stack", default=()
)
# Most recently recorded call; asyncio.gather/create_task copy it so fan-out calls inherit the same parent.
_current_call: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_call", default=None)
_overrides: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "overrides", default=None
)

STAGE_REGISTRY: dict[str, Callable[..., Any]] = {}  # so eval/optimize can re-run stages by name

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
        cloud_sync.note_run(run)
    return run


def current_stage() -> str | None:
    stack = _stage_stack.get()
    return ">".join(stack) if stack else None


def get_overrides() -> dict[str, Any] | None:
    return _overrides.get()


@contextmanager
def run_context(name: str | None = None, experiment_id: str | None = None):
    """Open a fresh Run (used per eval example / experiment trial)."""
    cfg = get_config()
    run = Run(project=cfg.project, name=name, experiment_id=experiment_id)
    get_store().save_run(run)
    cloud_sync.note_run(run)
    token = _current_run.set(run)
    call_token = _current_call.set(None)
    try:
        yield run
    finally:
        _current_call.reset(call_token)
        run.ended_at = time.time()
        get_store().save_run(run)
        cloud_sync.note_run(run)
        _current_run.reset(token)


@contextmanager
def stage_context(name: str):
    token = _stage_stack.set(_stage_stack.get() + (name,))
    try:
        yield
    finally:
        _stage_stack.reset(token)


def capture_context() -> dict[str, Any]:
    """Serializable run/stage/call snapshot; carry explicitly across boundaries contextvars
    don't survive (e.g. LangGraph's per-step executor), restore with restore_context()."""
    run = _current_run.get()
    return {
        "run_id": run.id if run else None,
        "run_project": run.project if run else None,
        "stage_stack": _stage_stack.get(),
        "current_call_id": _current_call.get(),
    }


@contextmanager
def restore_context(ctx: dict[str, Any] | None):
    """Re-enter a context captured with capture_context(). No-op if ctx has no run_id."""
    if not ctx or not ctx.get("run_id"):
        yield
        return
    run = Run(id=ctx["run_id"], project=ctx.get("run_project") or "default")
    run_token = _current_run.set(run)
    stage_token = _stage_stack.set(tuple(ctx.get("stage_stack") or ()))
    call_token = _current_call.set(ctx.get("current_call_id"))
    try:
        yield
    finally:
        _current_call.reset(call_token)
        _stage_stack.reset(stage_token)
        _current_run.reset(run_token)


@contextmanager
def override_context(model: str | None = None, prompt_transform=None):
    token = _overrides.set({"model": model, "prompt_transform": prompt_transform})
    try:
        yield
    finally:
        _overrides.reset(token)


@contextmanager
def policy_scope(stage: str):
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
        kwargs.setdefault("parent_call_id", _current_call.get())
        call = Call(run_id=run.id, **kwargs)
    get_store().save_call(call)
    if cloud_sync.is_running():
        cloud_sync.enqueue(call)
    _current_call.set(call.id)
    return call
