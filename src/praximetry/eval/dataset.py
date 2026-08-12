"""Golden-example wire format — what the hosted corpus sends and receives.

`input` may be a scalar (passed as single positional arg), a list (positional
args) or a dict (keyword args) — matching how the stage function is called.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Example(BaseModel):
    id: str = ""
    stage: str
    input: Any
    expected: Any = None
    scorer: str = "exact"
    threshold: float = 0.5
    # Optional human quality rating (1-5), for triaging the corpus by hand —
    # "this passes, but it's a mediocre example". Never read by the eval gate:
    # scoring uses `scorer`/`threshold` only.
    rating: int | None = Field(default=None, ge=1, le=5)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    examples: list[Example] = Field(default_factory=list)
    path: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        examples = []
        for i, line in enumerate(path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("id", f"ex{i + 1}")
            examples.append(Example(**d))
        return cls(examples=examples, path=str(path))

    def for_stage(self, stage: str) -> "Dataset":
        return Dataset(examples=[e for e in self.examples if e.stage == stage], path=self.path)

    @property
    def stages(self) -> list[str]:
        return sorted({e.stage for e in self.examples})
