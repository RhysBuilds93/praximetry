"""Shared shape for praximetry's module-level extension-point callables.

There are three: the policy hook (`runtime`), the redaction hook (`cloud_sync`),
and the capture hook (`instrument.patch`). Each is a single optional callable,
unset by default, that another layer (praximetry-cloud's optimizer,
`eval.capture`) swaps in. `Hook` replaces three near-identical `global`-juggling
blocks with one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generic, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Hook(Generic[F]):
    def __init__(self) -> None:
        self._default: F | None = None
        self._bound: ContextVar[F | None] = ContextVar(f"prax_hook_{id(self)}", default=None)

    @property
    def fn(self) -> F | None:
        bound = self._bound.get()
        return bound if bound is not None else self._default

    def set(self, fn: F | None) -> None:
        self._default = fn

    @contextmanager
    def bound(self, fn: F) -> Iterator[None]:
        """Install `fn` in this execution context, restoring it on exit."""
        token = self._bound.set(fn)
        try:
            yield
        finally:
            self._bound.reset(token)
