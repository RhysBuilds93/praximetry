"""Fixed, provider-agnostic shape every OutputAdapter parses responses into."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ContentPart(BaseModel):
    type: Literal["text", "image", "audio", "file"]
    data: Any


class NormalizedOutput(BaseModel):
    output_text: str = ""
    reasoning_text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    structured_output: Any | None = None
    content_parts: list[ContentPart] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
