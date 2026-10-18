"""Runtime context: current run, current stage, active experiment overrides."""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, ContextManager
from collections.abc import Callable
from weakref import WeakValueDictionary

from . import cloud_sync
from ._hooks import Hook
from .config import get_config
from .models import Call, Run
from .store import Store, get_store

logger = logging.getLogger(__name__)

_current_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar("run", default=None)
_stage_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "stage_stack", default=()
)


@dataclass
class _CallContext:
    run_id: str
    call_id: str
    child: _CallContext | None = None
    bridge: _CallContext | None = None


# Tasks inherit the same node, but each record forks a new local node. The child
# link bridges awaited work back to its caller without linking sibling branches.
_current_call: contextvars.ContextVar[_CallContext | None] = contextvars.ContextVar(
    "current_call", default=None
)
_overrides: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "overrides", default=None
)

# Context-local explicit runs take precedence over the lazily-created process run.
# Do not bind the fallback to a task's ContextVar: sibling tasks/threads would miss it.
_default_run: Run | None = None
_default_run_store: Store | None = None
_run_lock = threading.Lock()
_live_runs: WeakValueDictionary[str, Run] = WeakValueDictionary()
_call_contexts: WeakValueDictionary[str, _CallContext] = WeakValueDictionary()

STAGE_REGISTRY: dict[str, Callable[..., Any]] = {}  # so eval/optimize can re-run stages by name

policy_hook: Hook[Callable[[str], ContextManager[None]]] = Hook()


def set_policy_hook(fn: Callable[[str], ContextManager[None]] | None) -> None:
    """Register (or clear, with None) the policy hook applied around every @stage call."""
    policy_hook.set(fn)


def _save_run(run: Run) -> None:
    try:
        get_store().save_run(run)
    except Exception:  # persistence must not block or mask the customer's execution
        logger.warning("praximetry: failed to save run %s", run.id, exc_info=True)
    cloud_sync.note_run(run)


def current_run(create: bool = True) -> Run | None:
    """Return the explicit run, or the process-wide run shared by lazy fan-out."""
    global _default_run, _default_run_store
    run = _current_run.get()
    if run is not None:
        return run
    with _run_lock:
        if _default_run is None and not create:
            _current_call.set(None)
            return None
        cfg = get_config()
        store = get_store()
        if _default_run is not None and (
            _default_run.project != cfg.project or _default_run_store is not store
        ):
            _default_run = None
            _default_run_store = None
        parent = _current_call.get()
        if _default_run is None and create:
            _default_run = Run(project=cfg.project)
            _default_run_store = store
            _live_runs[_default_run.id] = _default_run
            _save_run(_default_run)
        if parent is not None and (
            _default_run is None or parent.run_id != _default_run.id
        ):
            _current_call.set(None)
        return _default_run


def current_stage() -> str | None:
    stack = _stage_stack.get()
    return ">".join(stack) if stack else None


def get_overrides() -> dict[str, Any] | None:
    return _overrides.get()


@contextmanager
def run_context(name: str | None = None, experiment_id: str | None = None):
    """Open an isolated Run; persistence failures are logged, never raised."""
    cfg = get_config()
    run = Run(project=cfg.project, name=name, experiment_id=experiment_id)
    with _run_lock:
        _live_runs[run.id] = run
    _save_run(run)
    token = _current_run.set(run)
    call_token = _current_call.set(None)
    try:
        yield run
    finally:
        _current_call.reset(call_token)
        run.ended_at = time.time()
        _save_run(run)
        _current_run.reset(token)


@contextmanager
def stage_context(name: str):
    token = _stage_stack.set(_stage_stack.get() + (name,))
    try:
        yield
    finally:
        _stage_stack.reset(token)


def capture_context() -> dict[str, Any]:
    """Capture run/stage/call/overrides for restore_context() across thread hops.

    Run identity is resolved locally without retaining it in the snapshot. A
    prompt_transform callable is preserved by reference, not JSON-serializable.
    """
    with _run_lock:
        # Capturing snapshots must not roll over a default run or clear ancestry
        # merely because configuration changed; recording handles that cutover.
        run = _current_run.get() or _default_run
        parent = _current_call.get()
        if parent is not None and run is not None and parent.run_id == run.id:
            latest = parent
            while latest.child is not None:
                latest = latest.child
            if latest is not parent:
                parent.child = latest
            call_id = latest.call_id
        else:
            call_id = run.last_call_id if run else None
    return {
        "run_id": run.id if run else None,
        "run_project": run.project if run else None,
        "stage_stack": _stage_stack.get(),
        "current_call_id": call_id,
        "overrides": dict(overrides) if (overrides := _overrides.get()) is not None else None,
    }


@contextmanager
def restore_context(ctx: dict[str, Any] | None):
    """Restore a snapshot, sharing its Run only while it is unfinished in this process."""
    if not ctx:
        yield
        return
    run = None
    parent = None
    if run_id := ctx.get("run_id"):
        with _run_lock:
            run = _live_runs.get(run_id)
            shared_run = run is not None and run.ended_at is None
            if run is None or run.ended_at is not None:
                run = Run(
                    id=run_id,
                    project=ctx.get("run_project") or "default",
                    last_call_id=ctx.get("current_call_id"),
                )
                _live_runs[run.id] = run
            if call_id := ctx.get("current_call_id"):
                # A restored branch owns its child link. Publish completed work
                # back to the original caller, never through a sibling's root.
                origin = _call_contexts.get(call_id) if shared_run else None
                if origin is not None and origin.run_id != run.id:
                    origin = None
                parent = _CallContext(run.id, call_id, bridge=origin)
    run_token = _current_run.set(run) if run is not None else None
    stage_token = _stage_stack.set(tuple(ctx.get("stage_stack") or ()))
    call_token = _current_call.set(parent)
    overrides = ctx.get("overrides")
    override_token = _overrides.set(dict(overrides) if overrides is not None else None)
    try:
        yield
    finally:
        _overrides.reset(override_token)
        _current_call.reset(call_token)
        _stage_stack.reset(stage_token)
        if run_token is not None:
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
    if policy_hook.fn is None:
        yield
        return
    with ExitStack() as stack:
        try:
            stack.enter_context(policy_hook.fn(stage))
        except Exception:  # a broken policy must not take down the customer's stage
            logger.warning("praximetry: policy hook failed for stage %s", stage, exc_info=True)
        yield


def reset_runtime() -> None:
    """Testing hook: discard the process run and clear local context state."""
    global _default_run, _default_run_store
    with _run_lock:
        _default_run = None
        _default_run_store = None
        _live_runs.clear()
        _call_contexts.clear()
    _current_run.set(None)
    _stage_stack.set(())
    _current_call.set(None)
    _overrides.set(None)


def record_call(call: Call | None = None, **kwargs: Any) -> Call:
    """Persist an LLM call. Manual kwargs are validated even when recording is disabled."""
    if not get_config().enabled:
        return call or Call(run_id="disabled", **kwargs)
    if call is None:
        run = current_run()
        kwargs.setdefault("stage", current_stage())
        parent = _current_call.get()
        kwargs.setdefault("parent_call_id", parent.call_id if parent else None)
        call = Call(run_id=run.id, **kwargs)
    else:
        run = current_run(create=False)
    get_store().save_call(call)
    if cloud_sync.is_running():
        if run is not None:
            cloud_sync.note_run(run)
        cloud_sync.enqueue(call)
    with _run_lock:
        parent = _current_call.get()
        local = _CallContext(call.run_id, call.id)
        if parent is not None and parent.run_id == call.run_id:
            parent.child = local
            if parent.bridge is not None:
                parent.bridge.child = local
        _call_contexts[call.id] = local
        _current_call.set(local)
        if run is not None:
            run.last_call_id = call.id
    return call
