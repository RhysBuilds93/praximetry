"""Cheap, content-safe automatic call identities; no filesystem inspection."""
from __future__ import annotations

import hashlib
import json
import sys

def _source() -> str:
    frame = sys._getframe(1)
    try:
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if module != "praximetry" and not module.startswith("praximetry."):
                code = frame.f_code
                filename = code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
                qualname = getattr(code, "co_qualname", None)
                if qualname is None:
                    owner = frame.f_locals.get("self")
                    if owner is None:
                        owner = frame.f_locals.get("cls")
                    if isinstance(owner, type):
                        qualname = f"{owner.__qualname__}.{code.co_name}"
                    elif owner is not None:
                        qualname = f"{type(owner).__qualname__}.{code.co_name}"
                    else:
                        qualname = code.co_name
                return f"{module}.{qualname}@{filename}:{frame.f_lineno}"
            frame = frame.f_back
        return "unknown"
    finally:
        del frame


def derive_label(messages: list, tools: list, model: str, source: str | None = None) -> str:
    """Group a source location and stable instruction/tool/model shape.

    User text, assistant responses and conversation length are not identity inputs.
    Explicit source strings allow read-only importers to use only observed evidence.
    """
    instructions = [m for m in messages if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    shape = json.dumps([model, instructions, tools], sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(shape.encode()).hexdigest()[:12]
    return f"{source if source is not None else 'sdk:' + _source()}:{digest}"
