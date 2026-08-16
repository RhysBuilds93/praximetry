"""Background sync of locally-recorded calls to the hosted /api/traces route.

Opt-in (see `praximetry.init(cloud=...)`), non-blocking, best-effort: `enqueue()`
never blocks or raises — it's a `queue.Queue.put_nowait` from the hot path in
`runtime.record_call`. A daemon thread drains the queue on a timer, groups
calls by run, and pushes each group via `CloudClient.push_trace`. Any failure
(network down, auth rejected, whatever) is caught, logged, and the batch is
dropped — the local sqlite store (already written by `record_call` before this
is ever reached) is completely unaffected either way; this is purely a
best-effort duplicate path to the cloud.

Mirrors `runtime.py`'s `set_policy_hook()` shape for the redaction extension
point: a single settable callable, no-op when unset.
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING
from collections.abc import Callable

from .models import Call, Run

if TYPE_CHECKING:
    from .eval.hosted import CloudClient

logger = logging.getLogger(__name__)

_DEFAULT_MAXSIZE = 1000
_DROP_LOG_INTERVAL_S = 5.0

_queue: queue.Queue[Call] = queue.Queue(maxsize=_DEFAULT_MAXSIZE)
_client: CloudClient | None = None
_flush_interval = 5.0
_thread: threading.Thread | None = None
_stop_event = threading.Event()
_lock = threading.Lock()

_run_cache: dict[str, Run] = {}
_redaction_hook: Callable[[Call], Call] | None = None

_drop_count = 0
_last_drop_log = 0.0


def note_run(run: Run) -> None:
    """Register (or refresh) a Run so a later flush can resolve the batch of
    calls sharing its run_id. Call this wherever a Run is created/updated."""
    _run_cache[run.id] = run


def set_redaction_hook(fn: Callable[[Call], Call] | None) -> None:
    """Register (or clear, with None) a hook applied to each Call right before
    it's serialized into the /api/traces payload. Never touches the locally
    stored Call — only what leaves the process over HTTP."""
    global _redaction_hook
    _redaction_hook = fn


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def start(
    client: CloudClient, flush_interval: float = 5.0, maxsize: int = _DEFAULT_MAXSIZE
) -> None:
    """Spawn (or reconfigure) the background sync worker. Idempotent while a
    worker is already running — calling twice does not spawn a second thread,
    but does swap in the new client so tests can recover a dead client."""
    global _client, _flush_interval, _thread, _queue
    with _lock:
        _client = client
        _flush_interval = flush_interval
        if _queue.maxsize != maxsize:
            _queue = queue.Queue(maxsize=maxsize)
        if _thread is not None and _thread.is_alive():
            return
        _stop_event.clear()
        _thread = threading.Thread(target=_worker_loop, name="praximetry-cloud-sync", daemon=True)
        _thread.start()


def stop(timeout: float = 2.0) -> None:
    """Best-effort final drain+push, then join the worker thread."""
    global _thread
    _stop_event.set()
    t = _thread
    if t is not None:
        t.join(timeout=timeout)
    _thread = None
    try:
        flush_now()
    except Exception:  # pragma: no cover - flush_now already swallows errors
        pass


def reset() -> None:
    """Testing hook: stop the worker and clear all module state so nothing
    leaks (thread, client, cached runs, redaction hook) across tests."""
    global _client, _run_cache, _redaction_hook, _drop_count, _last_drop_log, _queue
    stop(timeout=0.5)
    _client = None
    _run_cache = {}
    _redaction_hook = None
    _drop_count = 0
    _last_drop_log = 0.0
    _queue = queue.Queue(maxsize=_DEFAULT_MAXSIZE)


def _atexit_stop() -> None:
    try:
        stop(timeout=2.0)
    except Exception:  # pragma: no cover - must never raise/hang at exit
        pass


atexit.register(_atexit_stop)


def enqueue(call: Call) -> None:
    """Non-blocking enqueue. Drops (and rate-limit-logs) on a full queue —
    never blocks, never raises."""
    try:
        _queue.put_nowait(call)
    except queue.Full:
        _log_drop()


def _log_drop() -> None:
    global _drop_count, _last_drop_log
    _drop_count += 1
    now = time.monotonic()
    if now - _last_drop_log >= _DROP_LOG_INTERVAL_S:
        logger.warning("cloud_sync: dropped %d call(s) — queue full", _drop_count)
        _last_drop_log = now
        _drop_count = 0


def _worker_loop() -> None:
    while not _stop_event.is_set():
        _stop_event.wait(_flush_interval)
        _drain_and_push()


def flush_now() -> None:
    """Test-only (also used by stop()) synchronous drain+push — no waiting on
    the timer. Never raises."""
    _drain_and_push()


def _drain_and_push() -> None:
    client = _client
    calls: list[Call] = []
    while True:
        try:
            calls.append(_queue.get_nowait())
        except queue.Empty:
            break
    if not calls or client is None:
        return
    by_run: dict[str, list[Call]] = {}
    for c in calls:
        by_run.setdefault(c.run_id, []).append(c)
    for run_id, group in by_run.items():
        run = _run_cache.get(run_id)
        if run is None:
            logger.warning(
                "cloud_sync: no cached Run for run_id=%s — dropping %d call(s)", run_id, len(group)
            )
            continue
        try:
            hook = _redaction_hook
            payload = [hook(c) if hook else c for c in group]
            client.push_trace(run, payload)
        except Exception:
            logger.warning("cloud_sync: failed to push trace for run_id=%s", run_id, exc_info=True)
