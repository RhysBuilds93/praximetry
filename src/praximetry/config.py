from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    project: str = "default"
    db_path: Path = field(default_factory=lambda: Path(".praximetry") / "praximetry.db")
    auto_instrument: bool = True
    enabled: bool = True

    @classmethod
    def from_env(cls) -> Config:
        cfg = cls()
        if p := os.environ.get("PRAXIMETRY_PROJECT"):
            cfg.project = p
        if db := os.environ.get("PRAXIMETRY_DB"):
            cfg.db_path = Path(db)
        if os.environ.get("PRAXIMETRY_DISABLED"):
            cfg.enabled = False
        return cfg


# Set by praximetry.init(); read by instrumentation.
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(cfg: Config) -> None:
    global _config
    _config = cfg
