"""Core data models."""
from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


def _id() -> str:
    return uuid.uuid4().hex[:16]


class Call(BaseModel):
    """One LLM API call."""

    id: str = Field(default_factory=_id)
    run_id: str
    stage: str | None = None
    provider: str = "unknown"
    model: str = "unknown"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    response_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    ts: float = Field(default_factory=time.time)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Run(BaseModel):
    """A traced execution session (process lifetime or explicit run)."""

    id: str = Field(default_factory=_id)
    project: str = "default"
    name: str | None = None
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    experiment_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Opportunity(BaseModel):
    """A detected optimization opportunity."""

    kind: str  # oversized_model | prompt_bloat | verbose_payload | repeated_prefix
    stage: str | None = None
    title: str
    detail: str
    est_savings_usd_per_1k_calls: float = 0.0
    confidence: str = "medium"  # low | medium | high


class EvalResult(BaseModel):
    id: str = Field(default_factory=_id)
    experiment_id: str | None = None
    run_id: str | None = None
    stage: str | None = None
    example_id: str = ""
    scorer: str = "exact"
    score: float = 0.0
    passed: bool = False
    detail: str = ""
    ts: float = Field(default_factory=time.time)


class Experiment(BaseModel):
    """One variant tried by the optimization loop."""

    id: str = Field(default_factory=_id)
    name: str = ""
    stage: str | None = None
    variant: dict[str, Any] = Field(default_factory=dict)  # e.g. {"model": "...", "compact_prompts": true}
    created_at: float = Field(default_factory=time.time)
    quality: float | None = None  # mean eval score 0..1
    pass_rate: float | None = None
    cost_usd: float | None = None  # total eval-set cost
    input_tokens: int | None = None
    output_tokens: int | None = None
    baseline: bool = False
