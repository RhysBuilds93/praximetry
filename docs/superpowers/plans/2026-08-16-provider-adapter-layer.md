# Provider Adapter Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize LLM output into fixed fields (`output_text`, `reasoning_text`, `tool_calls`, `structured_output`, `content_parts`) at capture time, across a two-layer adapter design (parsing vs. capture mechanism), so reasoning never contaminates the golden corpus.

**Architecture:** `OutputAdapter` per provider (pure parsing) + `CaptureMechanism` (monkeypatch or callback) that installs an adapter. `praximetry` owns capture; `praximetry-cloud` consumes the normalized `Call` fields.

**Tech Stack:** Python, pydantic, sqlite3, asyncpg (cloud), LangChain (`langchain-core`) for the proof-of-extensibility adapter.

**Spec:** `docs/superpowers/specs/2026-08-16-provider-adapter-layer-design.md`

## Global Constraints

- No hard dependency on any provider SDK — adapters import lazily, same as today's `extractors.py`/`providers.py`.
- `Call.response_text` is renamed to `Call.output_text`; no backward-compat alias.
- Reasoning never enters the golden corpus (`Example.expected`).
- SQLite migration: no backfill (no existing rows worth keeping). Postgres corpus: backfill required (real data).
- `git commit`: no `Co-Authored-By` trailer.

---

## Task 1: NormalizedOutput data model

**Files:**
- Create: `src/praximetry/instrument/output.py`
- Modify: `src/praximetry/models.py:15-30` (Call model)
- Modify: `src/praximetry/store.py:22-29` (`_TABLES["calls"]`), `store.py:98-109` (`save_call`), `store.py:169-174` (`_row_to_call`)
- Test: `tests/test_output.py`

**Interfaces:**
- Produces: `praximetry.instrument.output.NormalizedOutput`, `ToolCall`, `ContentPart` — used by every adapter in Tasks 3-5 and by `patch.py` in Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_output.py
from praximetry.instrument.output import ContentPart, NormalizedOutput, ToolCall


def test_normalized_output_defaults():
    out = NormalizedOutput()
    assert out.output_text == ""
    assert out.reasoning_text == ""
    assert out.tool_calls == []
    assert out.structured_output is None
    assert out.content_parts == []
    assert out.tokens_in == 0
    assert out.tokens_out == 0


def test_normalized_output_with_tool_call():
    out = NormalizedOutput(
        output_text="",
        tool_calls=[ToolCall(id="t1", name="lookup", arguments={"q": "x"})],
    )
    assert out.tool_calls[0].name == "lookup"
    assert out.tool_calls[0].arguments == {"q": "x"}


def test_content_part():
    part = ContentPart(type="image", data="base64...")
    assert part.type == "image"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_output.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'praximetry.instrument.output'`

- [ ] **Step 3: Write the implementation**

```python
# src/praximetry/instrument/output.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_output.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Update the Call model**

In `src/praximetry/models.py`, replace:

```python
    response_text: str = ""
```

with:

```python
    output_text: str = ""
    reasoning_text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    structured_output: Any | None = None
    content_parts: list[dict[str, Any]] = Field(default_factory=list)
```

(Stored as plain `dict`/`Any` on `Call`, not `ToolCall`/`ContentPart` pydantic types — `Call` is the storage/wire shape shared with `praximetry-cloud`'s JSON-column persistence; adapters convert `NormalizedOutput`'s typed lists to plain dicts via `.model_dump()` when building the `Call`.)

- [ ] **Step 6: Update SQLite schema**

In `src/praximetry/store.py`, in `_TABLES["calls"]`, replace:

```python
        ("messages", "TEXT"), ("response_text", "TEXT"),
```

with:

```python
        ("messages", "TEXT"), ("output_text", "TEXT"), ("reasoning_text", "TEXT"),
        ("tool_calls", "TEXT"), ("structured_output", "TEXT"), ("content_parts", "TEXT"),
```

In `save_call`, replace the INSERT column list and values tuple:

```python
    def save_call(self, call: Call) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO calls
                   (id, run_id, parent_call_id, stage, provider, model, messages,
                    output_text, reasoning_text, tool_calls, structured_output, content_parts,
                    input_tokens, output_tokens, cost_usd, latency_ms, ts, error, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (call.id, call.run_id, call.parent_call_id, call.stage, call.provider, call.model,
                 json.dumps(call.messages), call.output_text, call.reasoning_text,
                 json.dumps(call.tool_calls), json.dumps(call.structured_output),
                 json.dumps(call.content_parts),
                 call.input_tokens, call.output_tokens, call.cost_usd, call.latency_ms,
                 call.ts, call.error, json.dumps(call.metadata)),
            )
```

In `_row_to_call`, parse the new JSON columns:

```python
    @staticmethod
    def _row_to_call(r: sqlite3.Row) -> Call:
        d = dict(r)
        d["messages"] = json.loads(d["messages"] or "[]")
        d["tool_calls"] = json.loads(d["tool_calls"] or "[]")
        d["structured_output"] = json.loads(d["structured_output"]) if d.get("structured_output") else None
        d["content_parts"] = json.loads(d["content_parts"] or "[]")
        d["metadata"] = json.loads(d["metadata"] or "{}")
        return Call(**d)
```

- [ ] **Step 7: Run full test suite, fix any `response_text` references**

Run: `.venv/bin/python -m pytest -q`
Expected: failures in `tests/test_patch.py` and anywhere else referencing `call.response_text` / `response_text=` — these are fixed in Task 6 (patch.py rewire). Confirm the only failures are `response_text`-related; note them, do not fix here.

- [ ] **Step 8: Commit**

```bash
git add src/praximetry/instrument/output.py src/praximetry/models.py src/praximetry/store.py tests/test_output.py
git commit -m "Add NormalizedOutput model and split Call.response_text into structured fields"
```

---

## Task 2: Reasoning pattern registry

**Files:**
- Create: `src/praximetry/instrument/reasoning_patterns.py`
- Test: `tests/test_reasoning_patterns.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `praximetry.instrument.reasoning_patterns.split_embedded_reasoning(text: str, model: str) -> tuple[str, str]` — used by `OpenAIAdapter` (Task 3, since gpt-oss is served through the OpenAI-compatible surface) and by the `praximetry-cloud` backfill script (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reasoning_patterns.py
from praximetry.instrument.reasoning_patterns import split_embedded_reasoning


def test_strips_gpt_oss_reasoning_prefix():
    text = "<reasoning>because the message asks for a color, answer red</reasoning>red"
    output, reasoning = split_embedded_reasoning(text, "openai.gpt-oss-20b-1:0")
    assert output == "red"
    assert reasoning == "because the message asks for a color, answer red"


def test_no_match_is_noop():
    output, reasoning = split_embedded_reasoning("plain answer", "openai.gpt-oss-20b-1:0")
    assert output == "plain answer"
    assert reasoning == ""


def test_unknown_model_is_noop():
    text = "<reasoning>trace</reasoning>answer"
    output, reasoning = split_embedded_reasoning(text, "gpt-4o")
    assert output == text
    assert reasoning == ""


def test_multiline_reasoning_block():
    text = "<reasoning>\nstep one\nstep two\n</reasoning>\nfinal answer"
    output, reasoning = split_embedded_reasoning(text, "openai.gpt-oss-120b-1:0")
    assert output == "final answer"
    assert "step one" in reasoning and "step two" in reasoning
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reasoning_patterns.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/praximetry/instrument/reasoning_patterns.py
"""Registry of known text-embedded reasoning prefixes, keyed by model prefix.

Some providers (e.g. gpt-oss models served through Bedrock's OpenAI-compatible
endpoint) prepend every reply with a visible <reasoning>...</reasoning> block
with no structural separation from the answer. Providers that DO separate
reasoning structurally (Anthropic thinking blocks, OpenAI o1/o3's reasoning
field) never consult this table — their adapters populate reasoning_text
directly during parsing.
"""
from __future__ import annotations

import re

_REASONING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai.gpt-oss", re.compile(r"^\s*<reasoning>(.*?)</reasoning>\s*", re.S)),
]


def split_embedded_reasoning(text: str, model: str) -> tuple[str, str]:
    for prefix, pattern in _REASONING_PATTERNS:
        if model.startswith(prefix):
            m = pattern.match(text)
            if m:
                return text[m.end():].lstrip(), m.group(1).strip()
    return text, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reasoning_patterns.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/praximetry/instrument/reasoning_patterns.py tests/test_reasoning_patterns.py
git commit -m "Add reasoning pattern registry for text-embedded reasoning models"
```

---

## Task 3: OutputAdapter interface + OpenAIAdapter

**Files:**
- Create: `src/praximetry/instrument/adapters.py`
- Test: `tests/test_adapters.py`

**Interfaces:**
- Consumes: `NormalizedOutput`/`ToolCall` (Task 1), `split_embedded_reasoning` (Task 2).
- Produces: `OutputAdapter` ABC, `OpenAIAdapter` instance, `ADAPTERS: dict[str, OutputAdapter]` registry (grown in Tasks 4-5). `OutputAdapter.get_messages(kwargs: dict) -> list[dict]`, `.parse_response(resp: Any, model: str) -> NormalizedOutput`, `.accumulate(chunk: Any, state: dict) -> None`, `.finalize_stream(state: dict) -> NormalizedOutput`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_adapters.py
from praximetry.instrument.adapters import ADAPTERS, OpenAIAdapter


def test_openai_adapter_registered():
    assert isinstance(ADAPTERS["openai"], OpenAIAdapter)


def test_openai_get_messages():
    adapter = OpenAIAdapter()
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    assert adapter.get_messages(kwargs) == [{"role": "user", "content": "hi"}]


def test_openai_parse_response_plain_text():
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    resp = ChatCompletion(
        id="x", created=0, model="gpt-4o", object="chat.completion",
        choices=[Choice(finish_reason="stop", index=0,
                        message=ChatCompletionMessage(role="assistant", content="hello world"))],
        usage=CompletionUsage(prompt_tokens=11, completion_tokens=3, total_tokens=14),
    )
    out = OpenAIAdapter().parse_response(resp, "gpt-4o")
    assert out.output_text == "hello world"
    assert out.reasoning_text == ""
    assert out.tokens_in == 11 and out.tokens_out == 3


def test_openai_parse_response_splits_gpt_oss_reasoning():
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    text = "<reasoning>thinking about it</reasoning>final answer"
    resp = ChatCompletion(
        id="x", created=0, model="openai.gpt-oss-20b-1:0", object="chat.completion",
        choices=[Choice(finish_reason="stop", index=0,
                        message=ChatCompletionMessage(role="assistant", content=text))],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )
    out = OpenAIAdapter().parse_response(resp, "openai.gpt-oss-20b-1:0")
    assert out.output_text == "final answer"
    assert out.reasoning_text == "thinking about it"


def test_openai_parse_response_tool_calls():
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall, Function,
    )

    resp = ChatCompletion(
        id="x", created=0, model="gpt-4o", object="chat.completion",
        choices=[Choice(finish_reason="tool_calls", index=0, message=ChatCompletionMessage(
            role="assistant", content=None,
            tool_calls=[ChatCompletionMessageToolCall(
                id="call_1", type="function",
                function=Function(name="lookup", arguments='{"q": "weather"}'),
            )],
        ))],
        usage=CompletionUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
    )
    out = OpenAIAdapter().parse_response(resp, "gpt-4o")
    assert out.output_text == ""
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "lookup"
    assert out.tool_calls[0].arguments == {"q": "weather"}


def test_openai_accumulate_and_finalize():
    from types import SimpleNamespace as NS

    adapter = OpenAIAdapter()
    state: dict = {}
    chunks = [
        NS(choices=[NS(delta=NS(content="he", tool_calls=None))], usage=None),
        NS(choices=[NS(delta=NS(content="llo", tool_calls=None))], usage=None),
        NS(choices=[], usage=NS(prompt_tokens=4, completion_tokens=2)),
    ]
    for chunk in chunks:
        adapter.accumulate(chunk, state)
    out = adapter.finalize_stream(state)
    assert out.output_text == "hello"
    assert out.tokens_in == 4 and out.tokens_out == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapters.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/praximetry/instrument/adapters.py
"""OutputAdapter: pure, per-provider SDK object -> NormalizedOutput parsing.

Kept separate from capture mechanism (patch.py / capture.py) so adapters stay
unit-testable against real SDK response objects without any monkeypatching,
same convention the old extractors.py used.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from .output import NormalizedOutput, ToolCall
from .reasoning_patterns import split_embedded_reasoning


def _g(obj: Any, *path: str, default: Any = None) -> Any:
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return default
    return obj


class OutputAdapter(ABC):
    name: str

    @abstractmethod
    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]: ...

    @abstractmethod
    def parse_response(self, resp: Any, model: str) -> NormalizedOutput: ...

    @abstractmethod
    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput: ...


class OpenAIAdapter(OutputAdapter):
    name = "openai"

    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        return list(kwargs.get("messages", []))

    def parse_response(self, resp: Any, model: str) -> NormalizedOutput:
        tin = _g(resp, "usage", "prompt_tokens", default=0) or 0
        tout = _g(resp, "usage", "completion_tokens", default=0) or 0
        choices = getattr(resp, "choices", None)
        if not choices:
            return NormalizedOutput(tokens_in=tin, tokens_out=tout)
        message = _g(choices[0], "message")
        text = _g(message, "content", default="") or ""
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name,
                     arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (_g(message, "tool_calls", default=[]) or [])
        ]
        output_text, reasoning_text = split_embedded_reasoning(text, model)
        return NormalizedOutput(output_text=output_text, reasoning_text=reasoning_text,
                                 tool_calls=tool_calls, tokens_in=tin, tokens_out=tout)

    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None:
        state.setdefault("text", "")
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = _g(choices[0], "delta", "content", default="") or ""
            state["text"] += delta
            state["tout"] = state.get("tout", 0) + (1 if delta else 0)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            state["tin"] = getattr(usage, "prompt_tokens", state.get("tin", 0)) or state.get("tin", 0)
            ct = getattr(usage, "completion_tokens", None)
            if ct is not None:
                state["tout"] = ct

    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput:
        model = state.get("model", "")
        output_text, reasoning_text = split_embedded_reasoning(state.get("text", ""), model)
        return NormalizedOutput(output_text=output_text, reasoning_text=reasoning_text,
                                 tokens_in=state.get("tin", 0), tokens_out=state.get("tout", 0))


ADAPTERS: dict[str, OutputAdapter] = {
    "openai": OpenAIAdapter(),
    "litellm": OpenAIAdapter(),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_adapters.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/praximetry/instrument/adapters.py tests/test_adapters.py
git commit -m "Add OutputAdapter interface and OpenAIAdapter"
```

---

## Task 4: AnthropicAdapter

**Files:**
- Modify: `src/praximetry/instrument/adapters.py` (append `AnthropicAdapter`, register in `ADAPTERS`)
- Test: `tests/test_adapters.py` (append)

**Interfaces:**
- Consumes: `OutputAdapter` (Task 3).
- Produces: `AnthropicAdapter` registered as `ADAPTERS["anthropic"]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adapters.py`:

```python
from praximetry.instrument.adapters import AnthropicAdapter


def test_anthropic_adapter_registered():
    assert isinstance(ADAPTERS["anthropic"], AnthropicAdapter)


def test_anthropic_parse_response_text_only():
    from anthropic.types import Message, TextBlock, Usage

    msg = Message(
        id="x", model="claude-sonnet-5", role="assistant", type="message",
        stop_reason="end_turn", stop_sequence=None,
        content=[TextBlock(type="text", text="hi there")],
        usage=Usage(input_tokens=7, output_tokens=2),
    )
    out = AnthropicAdapter().parse_response(msg, "claude-sonnet-5")
    assert out.output_text == "hi there"
    assert out.reasoning_text == ""


def test_anthropic_parse_response_thinking_block():
    from anthropic.types import Message, TextBlock, ThinkingBlock, Usage

    msg = Message(
        id="x", model="claude-sonnet-5", role="assistant", type="message",
        stop_reason="end_turn", stop_sequence=None,
        content=[
            ThinkingBlock(type="thinking", thinking="let me work through this", signature="sig"),
            TextBlock(type="text", text="the answer is 4"),
        ],
        usage=Usage(input_tokens=7, output_tokens=9),
    )
    out = AnthropicAdapter().parse_response(msg, "claude-sonnet-5")
    assert out.output_text == "the answer is 4"
    assert out.reasoning_text == "let me work through this"


def test_anthropic_parse_response_tool_use():
    from anthropic.types import Message, ToolUseBlock, Usage

    msg = Message(
        id="x", model="claude-sonnet-5", role="assistant", type="message",
        stop_reason="tool_use", stop_sequence=None,
        content=[ToolUseBlock(type="tool_use", id="tu_1", name="lookup", input={"q": "weather"})],
        usage=Usage(input_tokens=7, output_tokens=5),
    )
    out = AnthropicAdapter().parse_response(msg, "claude-sonnet-5")
    assert out.output_text == ""
    assert out.tool_calls[0].name == "lookup"
    assert out.tool_calls[0].arguments == {"q": "weather"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapters.py -v -k anthropic`
Expected: FAIL with `ImportError: cannot import name 'AnthropicAdapter'`

- [ ] **Step 3: Write the implementation**

Append to `src/praximetry/instrument/adapters.py`:

```python
class AnthropicAdapter(OutputAdapter):
    name = "anthropic"

    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        msgs = list(kwargs.get("messages", []))
        system = kwargs.get("system")
        if system:
            text = system if isinstance(system, str) else str(system)
            msgs = [{"role": "system", "content": text}] + msgs
        return msgs

    def parse_response(self, resp: Any, model: str) -> NormalizedOutput:
        output_text = ""
        reasoning_text = ""
        tool_calls: list[ToolCall] = []
        for block in getattr(resp, "content", None) or []:
            btype = getattr(block, "type", "")
            if btype == "text":
                output_text += getattr(block, "text", "") or ""
            elif btype == "thinking":
                reasoning_text += getattr(block, "thinking", "") or ""
            elif btype == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name,
                                            arguments=dict(block.input)))
        tin = _g(resp, "usage", "input_tokens", default=0) or 0
        tout = _g(resp, "usage", "output_tokens", default=0) or 0
        return NormalizedOutput(output_text=output_text, reasoning_text=reasoning_text,
                                 tool_calls=tool_calls, tokens_in=tin, tokens_out=tout)

    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None:
        state.setdefault("text", "")
        state.setdefault("reasoning", "")
        etype = getattr(chunk, "type", "")
        if etype == "message_start":
            state["tin"] = _g(chunk, "message", "usage", "input_tokens", default=state.get("tin", 0))
        elif etype == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            dtype = getattr(delta, "type", "")
            if dtype == "text_delta":
                state["text"] += getattr(delta, "text", "") or ""
            elif dtype == "thinking_delta":
                state["reasoning"] += getattr(delta, "thinking", "") or ""
        elif etype == "message_delta":
            out = _g(chunk, "usage", "output_tokens", default=None)
            if out is not None:
                state["tout"] = out

    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput:
        return NormalizedOutput(output_text=state.get("text", ""),
                                 reasoning_text=state.get("reasoning", ""),
                                 tokens_in=state.get("tin", 0), tokens_out=state.get("tout", 0))


ADAPTERS["anthropic"] = AnthropicAdapter()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_adapters.py -v`
Expected: PASS (all tests including Task 3's)

- [ ] **Step 5: Commit**

```bash
git add src/praximetry/instrument/adapters.py tests/test_adapters.py
git commit -m "Add AnthropicAdapter with thinking-block and tool-use parsing"
```

---

## Task 5: GeminiAdapter

**Files:**
- Modify: `src/praximetry/instrument/adapters.py` (append `GeminiAdapter`, register)
- Test: `tests/test_adapters.py` (append)

**Interfaces:**
- Consumes: `OutputAdapter` (Task 3).
- Produces: `GeminiAdapter` registered as `ADAPTERS["gemini"]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adapters.py`:

```python
from praximetry.instrument.adapters import GeminiAdapter


def test_gemini_adapter_registered():
    assert isinstance(ADAPTERS["gemini"], GeminiAdapter)


def test_gemini_parse_response_text():
    from google.genai import types

    resp = types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=4),
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text="answer")]))],
    )
    out = GeminiAdapter().parse_response(resp, "gemini-2.0-flash")
    assert out.output_text == "answer"
    assert out.tokens_in == 10 and out.tokens_out == 4


def test_gemini_parse_response_function_call():
    from google.genai import types

    resp = types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=6, candidates_token_count=3),
        candidates=[types.Candidate(content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(
                name="lookup", args={"q": "weather"}))],
        ))],
    )
    out = GeminiAdapter().parse_response(resp, "gemini-2.0-flash")
    assert out.tool_calls[0].name == "lookup"
    assert out.tool_calls[0].arguments == {"q": "weather"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapters.py -v -k gemini`
Expected: FAIL with `ImportError: cannot import name 'GeminiAdapter'`

- [ ] **Step 3: Write the implementation**

Append to `src/praximetry/instrument/adapters.py`:

```python
import uuid


class GeminiAdapter(OutputAdapter):
    name = "gemini"

    def get_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        contents = kwargs.get("contents")
        config = kwargs.get("config")
        msgs = [{"role": "user", "content": contents if isinstance(contents, str) else str(contents)}]
        sys_instr = _g(config, "system_instruction") if config is not None else None
        if isinstance(sys_instr, str):
            msgs = [{"role": "system", "content": sys_instr}] + msgs
        return msgs

    def parse_response(self, resp: Any, model: str) -> NormalizedOutput:
        text = ""
        tool_calls: list[ToolCall] = []
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            for part in _g(candidates[0], "content", "parts", default=[]) or []:
                if getattr(part, "text", None):
                    text += part.text
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append(ToolCall(id=uuid.uuid4().hex[:16], name=fc.name,
                                                arguments=dict(fc.args or {})))
        tin = _g(resp, "usage_metadata", "prompt_token_count", default=0) or 0
        tout = _g(resp, "usage_metadata", "candidates_token_count", default=0) or 0
        return NormalizedOutput(output_text=text, tool_calls=tool_calls, tokens_in=tin, tokens_out=tout)

    def accumulate(self, chunk: Any, state: dict[str, Any]) -> None:
        state.setdefault("text", "")
        state["text"] += getattr(chunk, "text", "") or ""
        tin = _g(chunk, "usage_metadata", "prompt_token_count", default=None)
        tout = _g(chunk, "usage_metadata", "candidates_token_count", default=None)
        if tin:
            state["tin"] = tin
        if tout:
            state["tout"] = tout

    def finalize_stream(self, state: dict[str, Any]) -> NormalizedOutput:
        return NormalizedOutput(output_text=state.get("text", ""),
                                 tokens_in=state.get("tin", 0), tokens_out=state.get("tout", 0))


ADAPTERS["gemini"] = GeminiAdapter()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_adapters.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/praximetry/instrument/adapters.py tests/test_adapters.py
git commit -m "Add GeminiAdapter with function-call parsing"
```

---

## Task 6: CaptureMechanism + rewire patch.py, delete extractors.py

**Files:**
- Create: `src/praximetry/instrument/capture.py`
- Modify: `src/praximetry/instrument/patch.py` (full rewrite of `_record`, `_instrument`, `_patch`)
- Modify: `src/praximetry/instrument/providers.py` (`ProviderSpec` drops `get_messages`/`response_extract`/`accumulate`, gains `adapter: OutputAdapter`)
- Delete: `src/praximetry/instrument/extractors.py`
- Modify: `tests/test_patch.py` (drop extractor-specific tests now covered by `test_adapters.py`, update `response_text` references to `output_text`)

**Interfaces:**
- Consumes: `OutputAdapter`, `ADAPTERS` (Tasks 3-5), `NormalizedOutput` (Task 1).
- Produces: `CaptureMechanism` ABC, `MonkeypatchCapture`, used by `LangChainCallbackCapture` in Task 9.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_patch.py` (replacing the now-obsolete `test_openai_extractor_real_object` / `test_anthropic_extractor_real_object` / `test_gemini_extractor_real_object`, which move to `tests/test_adapters.py` and already exist there):

```python
def test_sync_buffered_records_normalized_fields():
    from praximetry.instrument.adapters import ADAPTERS

    create = P._instrument(lambda self, **k: _oai_resp("out", 5, 2), "openai",
                           ADAPTERS["openai"], False)
    create(None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    c = get_store().calls()[0]
    assert c.provider == "openai" and c.output_tokens == 2 and c.output_text == "out"
    assert c.reasoning_text == ""
    assert c.cost_usd > 0
```

Remove the three `test_*_extractor_real_object` functions from `tests/test_patch.py` (superseded by Task 3-5's `tests/test_adapters.py` coverage) and the `from praximetry.instrument import extractors as ex` import.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_patch.py -v`
Expected: FAIL — `P._instrument` still has the old 6-positional-arg signature (`get_messages, response_extract, accumulate` separately), so this 4-arg call errors.

- [ ] **Step 3: Write `capture.py`**

```python
# src/praximetry/instrument/capture.py
"""CaptureMechanism: how we get invoked for a call, decoupled from how the
response gets parsed (OutputAdapter). Split out because not every capture
style is "patch a client method" — LangChain exposes a callback/event stream
instead of a create() method to intercept.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .adapters import OutputAdapter


class CaptureMechanism(ABC):
    name: str

    @abstractmethod
    def install(self, adapter: OutputAdapter) -> bool:
        """Wire this mechanism in for `adapter`'s provider. Returns False if
        the underlying SDK/framework isn't importable."""
```

- [ ] **Step 4: Rewrite `patch.py`**

Replace `_record`, `_instrument`, and `_patch` in `src/praximetry/instrument/patch.py`:

```python
def _record(provider: str, model: str, messages: list, out: "NormalizedOutput",
            t0: float, error: str | None) -> None:
    record_call(
        provider=provider, model=model, messages=messages,
        output_text=out.output_text, reasoning_text=out.reasoning_text,
        tool_calls=[tc.model_dump() for tc in out.tool_calls],
        structured_output=out.structured_output,
        content_parts=[cp.model_dump() for cp in out.content_parts],
        input_tokens=out.tokens_in, output_tokens=out.tokens_out,
        cost_usd=pricing.cost_usd(model, out.tokens_in, out.tokens_out),
        latency_ms=(time.perf_counter() - t0) * 1000, error=error,
    )
```

On error, `_record` is called with `NormalizedOutput()` (all defaults) instead of the old `_record(provider, model, messages, "", 0, 0, t0, str(e))`.

```python
def _make_stream_done(provider: str, model: str, messages: list, adapter, t0: float):
    def on_done(state: dict[str, Any]) -> None:
        state["model"] = model
        out = adapter.finalize_stream(state)
        _record(provider, model, messages, out, t0, None)
    return on_done


def _instrument(original: Callable, provider: str, adapter, is_async: bool,
                messages_key: str = "messages", force_stream: bool = False) -> Callable:
    """Build a patched create() wrapping `original` for one provider/sync-ness."""
    if is_async:
        async def acreate(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs = _apply_overrides(kwargs, messages_key)
            model, messages = kwargs.get("model", "unknown"), adapter.get_messages(kwargs)
            if _capture_hook is not None:
                return _capture_hook({"provider": provider, "model": model,
                                       "messages": messages, "tools": kwargs.get("tools", [])})
            t0 = time.perf_counter()
            if force_stream or kwargs.get("stream"):
                resp = await original(self, *args, **kwargs)
                return AsyncStreamWrapper(resp, adapter.accumulate,
                                          _make_stream_done(provider, model, messages, adapter, t0))
            try:
                resp = await original(self, *args, **kwargs)
            except Exception as e:  # noqa: BLE001
                _record(provider, model, messages, NormalizedOutput(), t0, str(e))
                raise
            out = adapter.parse_response(resp, model)
            _record(provider, model, messages, out, t0, None)
            return resp
        acreate._praximetry_patched = True  # type: ignore[attr-defined]
        return acreate

    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = _apply_overrides(kwargs, messages_key)
        model, messages = kwargs.get("model", "unknown"), adapter.get_messages(kwargs)
        if _capture_hook is not None:
            return _capture_hook({"provider": provider, "model": model,
                                   "messages": messages, "tools": kwargs.get("tools", [])})
        t0 = time.perf_counter()
        if force_stream or kwargs.get("stream"):
            resp = original(self, *args, **kwargs)
            return SyncStreamWrapper(resp, adapter.accumulate,
                                     _make_stream_done(provider, model, messages, adapter, t0))
        try:
            resp = original(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            _record(provider, model, messages, NormalizedOutput(), t0, str(e))
            raise
        out = adapter.parse_response(resp, model)
        _record(provider, model, messages, out, t0, None)
        return resp
    create._praximetry_patched = True  # type: ignore[attr-defined]
    return create
```

Update `_patch()` to pass `spec.adapter` instead of the three extractor functions:

```python
def _patch(spec: ProviderSpec) -> bool:
    if spec.name in _patched:
        return True
    try:
        sync_host, async_host = spec.owner()
    except ImportError:
        return False

    for target in spec.targets:
        host = async_host if target.is_async else sync_host
        if target.optional and not hasattr(host, target.attr):
            continue
        original = getattr(host, target.attr)
        if target.self_less:
            inst = _instrument(_self_less(original), spec.name, spec.adapter,
                               is_async=target.is_async, messages_key=spec.messages_key,
                               force_stream=target.force_stream)
            setattr(host, target.attr, _self_less_caller(inst, target.is_async))
        else:
            new = _instrument(original, spec.name, spec.adapter,
                              is_async=target.is_async, messages_key=spec.messages_key,
                              force_stream=target.force_stream)
            setattr(host, target.attr, new)  # type: ignore[method-assign]

    _patched.add(spec.name)
    return True
```

Add the import at the top of `patch.py`: `from .output import NormalizedOutput`.

`MonkeypatchCapture` (in `capture.py`) wraps this as the `CaptureMechanism` used for all four SDK-patch providers:

```python
# append to src/praximetry/instrument/capture.py
from .providers import PROVIDERS
from . import patch as _patch_module


class MonkeypatchCapture(CaptureMechanism):
    name = "monkeypatch"

    def install(self, adapter: OutputAdapter) -> bool:
        spec = next((s for s in PROVIDERS if s.name == adapter.name), None)
        if spec is None:
            return False
        return _patch_module._patch(spec)
```

- [ ] **Step 5: Update `providers.py`**

Replace `ProviderSpec`'s `get_messages`/`response_extract`/`accumulate` fields with a single `adapter` field, and update `PROVIDERS` to reference `ADAPTERS` instead of `ex.*`:

```python
@dataclass
class ProviderSpec:
    name: str
    owner: Callable[[], tuple[Any, Any]]
    targets: list[PatchTarget]
    adapter: "OutputAdapter"
    messages_key: str = "messages"
```

```python
from .adapters import ADAPTERS, OutputAdapter

PROVIDERS: list[ProviderSpec] = [
    ProviderSpec(name="openai", owner=_openai_owner,
                 targets=[PatchTarget(attr="create", is_async=False),
                          PatchTarget(attr="create", is_async=True)],
                 adapter=ADAPTERS["openai"]),
    ProviderSpec(name="anthropic", owner=_anthropic_owner,
                 targets=[PatchTarget(attr="create", is_async=False),
                          PatchTarget(attr="create", is_async=True)],
                 adapter=ADAPTERS["anthropic"]),
    ProviderSpec(name="litellm", owner=_litellm_owner,
                 targets=[PatchTarget(attr="completion", is_async=False, self_less=True),
                          PatchTarget(attr="acompletion", is_async=True, self_less=True, optional=True)],
                 adapter=ADAPTERS["litellm"]),
    ProviderSpec(name="gemini", owner=_gemini_owner,
                 targets=[PatchTarget(attr="generate_content", is_async=False),
                          PatchTarget(attr="generate_content", is_async=True),
                          PatchTarget(attr="generate_content_stream", is_async=False,
                                      force_stream=True, optional=True)],
                 adapter=ADAPTERS["gemini"], messages_key="contents"),
]
```

Delete the `from . import extractors as ex` import and the `_openai_owner`/etc. functions stay as-is.

- [ ] **Step 6: Delete `extractors.py`**

```bash
rm src/praximetry/instrument/extractors.py
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS across the board. Fix any remaining `response_text`/`ex.` references surfaced by the run (check `tests/conftest.py`'s `FakeLLM` and any eval/capture tests that read `Call.response_text`).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Rewire patch.py onto OutputAdapter/CaptureMechanism, remove extractors.py"
```

---

## Task 7: praximetry-cloud consumer updates

**Files:**
- Modify (in `praximetry-cloud` repo): `src/praximetry_cloud/dashboard/server.py:222` (`example_from_call`)
- Modify: `src/praximetry_cloud/otel.py` (wherever it reads `response_text` — grep first)
- Modify: dashboard call-detail rendering template/component that displays `response_text` (grep for `response_text` in templates/frontend)
- Test: existing test covering `example_from_call` (grep `tests/` for it)

**Interfaces:**
- Consumes: `Call.output_text`/`reasoning_text`/`tool_calls` (Task 1, now shipped in `praximetry` — bump the `praximetry` dependency pin in `praximetry-cloud`'s `pyproject.toml` to the commit/tag that includes Tasks 1-6).

- [ ] **Step 1: Bump the `praximetry` dependency**

In `praximetry-cloud`'s `pyproject.toml`, update the `praximetry` git dependency to point at the commit from Task 6's final commit (or a tag cut from it). Run `uv pip install -e ".[dev]"` (or equivalent) to pick it up.

- [ ] **Step 2: Grep for every `response_text` reference**

Run: `grep -rn "response_text" src/ tests/` in the `praximetry-cloud` repo. Expect hits in `dashboard/server.py` (`example_from_call`), `otel.py`, and dashboard templates/frontend call-detail views.

- [ ] **Step 3: Write/update the failing test for `example_from_call`**

Locate the existing test (likely `tests/test_dashboard.py` or similar — grep `example_from_call`). Update its fixture `Call` to use `output_text=` instead of `response_text=`, and assert:

```python
def test_example_from_call_uses_output_text_not_reasoning():
    call = Call(id="c1", run_id="r1", stage="triage", provider="openai",
                model="openai.gpt-oss-20b-1:0",
                messages=[{"role": "user", "content": "classify this"}],
                output_text="security", reasoning_text="long stochastic trace")
    example = example_from_call(call)
    assert example.expected == "security"
    assert "trace" not in example.expected
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k example_from_call -v`
Expected: FAIL — `example_from_call` still reads `call.response_text`, which no longer exists on `Call` (AttributeError) or the assertion fails against old behavior.

- [ ] **Step 5: Update `example_from_call`**

In `src/praximetry_cloud/dashboard/server.py`, change:

```python
        expected=call.response_text,
```

to:

```python
        expected=call.output_text,
```

- [ ] **Step 6: Update `otel.py` and dashboard display**

For each `response_text` hit found in Step 2 outside `example_from_call`: replace reads with `output_text` for the answer, and where the UI should show the reasoning trace too (call detail view), add it as a separate labeled section reading `reasoning_text` — do not concatenate them back into one field.

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -k example_from_call -v`
Expected: PASS

Run full suite: `.venv/bin/python -m pytest tests/ -q`
Expected: only pre-existing unrelated failures (the `PRAXIMETRY_EXAMPLE_MODEL` golden-replay ones noted in prior session).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Read output_text/reasoning_text instead of response_text in corpus promotion and display"
```

---

## Task 8: Postgres corpus backfill script

**Files:**
- Create (in `praximetry-cloud` repo): `scripts/backfill_reasoning_prefix.py`
- Test: `tests/test_backfill_reasoning_prefix.py`

**Interfaces:**
- Consumes: `praximetry.instrument.reasoning_patterns.split_embedded_reasoning` (Task 2), `store.py`'s Postgres pool/connection helper (grep `store.py` for how `asyncpg` connections are obtained elsewhere, e.g. `save_golden_example`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill_reasoning_prefix.py
import pytest

from scripts.backfill_reasoning_prefix import find_contaminated, strip_reasoning


def test_strip_reasoning_removes_tag():
    cleaned = strip_reasoning("<reasoning>trace here</reasoning>security", "openai.gpt-oss-20b-1:0")
    assert cleaned == "security"


def test_strip_reasoning_noop_when_no_match():
    cleaned = strip_reasoning("plain text", "gpt-4o")
    assert cleaned == "plain text"


@pytest.mark.asyncio
async def test_find_contaminated_filters_rows(pg_pool):  # pg_pool: existing e2e Postgres fixture
    await pg_pool.execute(
        "INSERT INTO examples (id, tenant_id, stage, input, expected, scorer, threshold) "
        "VALUES ('call:c1', 't1', 'triage', 'in', '<reasoning>x</reasoning>security', 'similarity', 0.8)"
    )
    await pg_pool.execute(
        "INSERT INTO examples (id, tenant_id, stage, input, expected, scorer, threshold) "
        "VALUES ('call:c2', 't1', 'triage', 'in', 'clean answer', 'similarity', 0.8)"
    )
    rows = await find_contaminated(pg_pool)
    assert {r["id"] for r in rows} == {"call:c1"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_backfill_reasoning_prefix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.backfill_reasoning_prefix'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/backfill_reasoning_prefix.py
"""One-time cleanup: strip <reasoning> prefixes from existing golden-corpus
`expected` values (praximetry-cloud's Postgres store), now that
example_from_call stores output_text (already clean) going forward. Run once
after Task 7 ships. Usage: python -m scripts.backfill_reasoning_prefix [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio

import asyncpg
from praximetry.instrument.reasoning_patterns import split_embedded_reasoning

_TAGGED = "%<reasoning>%"


def strip_reasoning(expected: str, model: str) -> str:
    output_text, _ = split_embedded_reasoning(expected, model)
    return output_text


async def find_contaminated(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    return await pool.fetch(
        "SELECT id, tenant_id, expected, metadata->>'model' AS model "
        "FROM examples WHERE expected LIKE $1", _TAGGED
    )


async def run(dsn: str, dry_run: bool) -> None:
    pool = await asyncpg.create_pool(dsn)
    try:
        rows = await find_contaminated(pool)
        for row in rows:
            cleaned = strip_reasoning(row["expected"], row["model"] or "")
            if cleaned == row["expected"]:
                continue
            print(f"{row['tenant_id']}/{row['id']}: {len(row['expected'])} -> {len(cleaned)} chars")
            if not dry_run:
                await pool.execute(
                    "UPDATE examples SET expected=$1 WHERE id=$2 AND tenant_id=$3",
                    cleaned, row["id"], row["tenant_id"],
                )
        print(f"{'would update' if dry_run else 'updated'} {len(rows)} rows")
    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.dsn, args.dry_run))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_backfill_reasoning_prefix.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Dry-run against staging, then run for real**

```bash
.venv/bin/python -m scripts.backfill_reasoning_prefix --dsn "$STAGING_PG_DSN" --dry-run
```

Review the printed rows. If they look right:

```bash
.venv/bin/python -m scripts.backfill_reasoning_prefix --dsn "$STAGING_PG_DSN"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/backfill_reasoning_prefix.py tests/test_backfill_reasoning_prefix.py
git commit -m "Add one-time backfill for reasoning-contaminated golden examples"
```

---

## Task 9: LangChainCallbackCapture (proof of extensibility)

**Files:**
- Create: `src/praximetry/instrument/langchain.py`
- Modify: `pyproject.toml` (add `langchain = ["langchain-core>=0.3"]` optional extra; add `langchain-core` to `dev` extras)
- Test: `tests/test_langchain_capture.py`

**Interfaces:**
- Consumes: `CaptureMechanism` (Task 6), `NormalizedOutput`/`ToolCall` (Task 1), `record_call` (existing, `runtime.py`).
- Produces: `praximetry.instrument.langchain.LangChainCallbackCapture`, `install_langchain_capture() -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_langchain_capture.py
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage

from praximetry.instrument.langchain import LangChainCallbackCapture
from praximetry.store import get_store


def test_langchain_capture_records_call():
    capture = LangChainCallbackCapture()
    llm = FakeListLLM(responses=["the answer is 4"], callbacks=[capture])
    llm.invoke("what is 2+2?")

    calls = get_store().calls()
    assert calls
    c = calls[0]
    assert c.provider == "langchain"
    assert c.output_text == "the answer is 4"
    assert c.reasoning_text == ""


def test_langchain_capture_records_error():
    class BoomLLM(FakeListLLM):
        def _call(self, *a, **k):
            raise RuntimeError("boom")

    capture = LangChainCallbackCapture()
    llm = BoomLLM(responses=["unused"], callbacks=[capture])
    try:
        llm.invoke("trigger")
    except RuntimeError:
        pass
    calls = get_store().calls()
    assert calls[0].error and "boom" in calls[0].error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_langchain_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'praximetry.instrument.langchain'`

- [ ] **Step 3: Write the implementation**

```python
# src/praximetry/instrument/langchain.py
"""LangChain integration via its BaseCallbackHandler event stream, not
monkeypatching — LangChain exposes no client.create() method to patch. This
is the proof that CaptureMechanism generalizes beyond SDK-method-patching.
"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ..runtime import record_call
from .. import pricing
from .adapters import ADAPTERS
from .capture import CaptureMechanism
from .output import NormalizedOutput


class LangChainCallbackCapture(BaseCallbackHandler, CaptureMechanism):
    name = "langchain"

    def __init__(self) -> None:
        self._starts: dict[UUID, tuple[list[dict], float]] = {}

    def install(self, adapter: Any) -> bool:
        return True  # this class IS the handler; caller attaches it via `callbacks=[...]`

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        messages = [{"role": "user", "content": p} for p in prompts]
        self._starts[run_id] = (messages, time.perf_counter())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        messages, t0 = self._starts.pop(run_id, ([], time.perf_counter()))
        model = (response.llm_output or {}).get("model_name", "unknown")
        text = response.generations[0][0].text if response.generations and response.generations[0] else ""
        adapter = ADAPTERS.get(_provider_for_model(model))
        if adapter is not None:
            out = adapter.parse_response(_fake_openai_response(text), model)
        else:
            out = NormalizedOutput(output_text=text)
        record_call(
            provider="langchain", model=model, messages=messages,
            output_text=out.output_text, reasoning_text=out.reasoning_text,
            tool_calls=[tc.model_dump() for tc in out.tool_calls],
            structured_output=out.structured_output,
            content_parts=[cp.model_dump() for cp in out.content_parts],
            input_tokens=out.tokens_in, output_tokens=out.tokens_out,
            cost_usd=pricing.cost_usd(model, out.tokens_in, out.tokens_out),
            latency_ms=(time.perf_counter() - t0) * 1000, error=None,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        messages, t0 = self._starts.pop(run_id, ([], time.perf_counter()))
        record_call(
            provider="langchain", model="unknown", messages=messages,
            output_text="", reasoning_text="", tool_calls=[], structured_output=None,
            content_parts=[], input_tokens=0, output_tokens=0, cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000, error=str(error),
        )


def _provider_for_model(model: str) -> str:
    if model.startswith("gpt") or model.startswith("openai"):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    return "openai"


def _fake_openai_response(text: str) -> Any:
    from types import SimpleNamespace as NS
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))],
              usage=NS(prompt_tokens=0, completion_tokens=0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_langchain_capture.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Add the optional extra**

In `pyproject.toml`, add under `[project.optional-dependencies]`:

```
langchain = ["langchain-core>=0.3"]
```

and add `"langchain-core>=0.3",` to the `dev` list.

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS across the board.

- [ ] **Step 7: Commit**

```bash
git add src/praximetry/instrument/langchain.py pyproject.toml tests/test_langchain_capture.py
git commit -m "Add LangChain callback-based capture as extensibility proof"
```
