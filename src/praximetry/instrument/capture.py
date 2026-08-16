"""CaptureMechanism: how we get invoked for a call, decoupled from how the
response gets parsed (OutputAdapter). Split out because not every capture
style is "patch a client method" — LangChain exposes a callback/event stream
instead of a create() method to intercept.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from . import patch as _patch_module
from .adapters import OutputAdapter
from .providers import PROVIDERS


class CaptureMechanism(ABC):
    name: str

    @abstractmethod
    def install(self, adapter: OutputAdapter) -> bool:
        """Wire this mechanism in for `adapter`'s provider.

        Returns:
            False if the underlying SDK/framework isn't importable.
        """


class MonkeypatchCapture(CaptureMechanism):
    name = "monkeypatch"

    def install(self, adapter: OutputAdapter) -> bool:
        spec = next((s for s in PROVIDERS if s.name == adapter.name), None)
        if spec is None:
            return False
        return _patch_module._patch(spec)
