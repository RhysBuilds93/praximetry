"""Transparent wrappers for streaming responses.

They proxy the underlying stream (iteration, context-manager, attribute access)
unchanged, accumulate text/usage as chunks flow by, and fire an `on_done`
callback with the accumulated state when the stream is exhausted or closed —
so streamed calls are recorded just like buffered ones.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)

Accumulate = Callable[[Any, dict[str, Any]], None]
OnDone = Callable[[dict[str, Any]], None]


def _new_state() -> dict[str, Any]:
    return {"text": "", "tin": 0, "tout": 0}


class SyncStreamWrapper:
    def __init__(self, stream: Any, accumulate: Accumulate, on_done: OnDone):
        self._stream = stream
        self._accumulate = accumulate
        self._on_done = on_done
        self._state = _new_state()
        self._done = False
        self._it: Any = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self._stream)
        try:
            chunk = next(self._it)
        except StopIteration:
            self._finish()
            raise
        except Exception as exc:
            self._finish(str(exc))
            raise
        try:
            self._accumulate(chunk, self._state)
        except Exception:  # never let instrumentation break a stream
            logger.debug("stream accumulate failed", exc_info=True)
        return chunk


    def _finish(self, error: str | None = None) -> None:
        if not self._done:
            self._done = True
            if error is not None:
                self._state["error"] = error
            try:
                self._on_done(self._state)
            except Exception:  # recording must not break or mask the customer's stream
                logger.warning("stream on_done failed", exc_info=True)


    def close(self) -> Any:
        self._finish()
        if hasattr(self._stream, "close"):
            return self._stream.close()
        return None


    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *exc: Any):
        self._finish(str(exc[1]) if len(exc) > 1 and exc[1] is not None else None)
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*exc)
        return False


    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncStreamWrapper:
    def __init__(self, stream: Any, accumulate: Accumulate, on_done: OnDone):
        self._stream = stream
        self._accumulate = accumulate
        self._on_done = on_done
        self._state = _new_state()
        self._done = False
        self._it: Any = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._it is None:
            self._it = self._stream.__aiter__()
        try:
            chunk = await self._it.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except Exception as exc:
            self._finish(str(exc))
            raise
        try:
            self._accumulate(chunk, self._state)
        except Exception:  # never let instrumentation break a stream
            logger.debug("stream accumulate failed", exc_info=True)
        return chunk


    def _finish(self, error: str | None = None) -> None:
        if not self._done:
            self._done = True
            if error is not None:
                self._state["error"] = error
            try:
                self._on_done(self._state)
            except Exception:  # recording must not break or mask the customer's stream
                logger.warning("stream on_done failed", exc_info=True)

    async def aclose(self) -> Any:
        self._finish()
        if hasattr(self._stream, "aclose"):
            return await self._stream.aclose()
        return None


    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any):
        self._finish(str(exc[1]) if len(exc) > 1 and exc[1] is not None else None)
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*exc)
        return False


    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
