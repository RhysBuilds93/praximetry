"""Shared shape for praximetry's module-level extension-point callables.

There are three: the policy hook (`runtime`), the redaction hook (`cloud_sync`),
and the capture hook (`instrument.patch`). Each is a single optional callable,
unset by default, that another layer (praximetry-cloud's optimizer,
`eval.capture`) swaps in. `Hook` replaces three near-identical `global`-juggling
blocks with one.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generic, TypeVar
from collections.abc import Callable, Iterator

F = TypeVar("F", bound=Callable[..., Any])


class Hook(Generic[F]):
    def __init__(self) -> None:
        self.fn: F | None = None

    def set(self, fn: F | None) -> None:
        self.fn = fn

    @contextmanager
    def bound(self, fn: F) -> Iterator[None]:
        """Install `fn` for the duration of the block, restoring the previous value after."""
        prev, self.fn = self.fn, fn
        try:
            yield
        finally:
            self.fn = prev
