# Provider adapter layer

Status: draft
Spans: `praximetry` (OSS) and `praximetry-cloud`

## Problem

`praximetry`'s per-provider extraction (`instrument/extractors.py`) collapses
every LLM response into a single `response_text` string. That's lossy in two
different ways depending on the provider:

- Some SDKs return reasoning/thinking as a **structurally separate** field
  (Anthropic's `thinking` content blocks, OpenAI o1/o3's `reasoning` field).
  `extractors.py` only reads `type == "text"` blocks today, so this is
  silently dropped — never stored anywhere.
- Some providers embed reasoning **inside** the text field with no
  structural separation (gpt-oss models served through Bedrock's
  OpenAI-compatible endpoint prefix every reply with a visible
  `<reasoning>...</reasoning>` block). This one *does* get stored, baked
  into `response_text`.

Tool calls, structured/JSON outputs, and multi-modal content parts aren't
extracted at all today — also silently dropped.

This caused a real bug: `praximetry-cloud`'s golden-example promotion
(`dashboard/server.py`'s `example_from_call`) stores `expected=response_text`
verbatim. For gpt-oss-backed stages, that means `expected` contains one
sampled reasoning trace. Replaying the same prompt during CI-gated eval
produces a *different* reasoning trace with the same final answer, and the
structural `similarity` scorer — comparing full text including reasoning —
scored the correct answer near zero. (Fixed as an immediate patch in
`scorers.py` by stripping a leading `<reasoning>` block symmetrically before
comparison; that fix stays in place as defense-in-depth, see Rollout.)

The underlying cause is architectural: nothing in the capture path
distinguishes "the answer" from "everything else the model produced." This
project fixes that at the source.

## Goals

- A fixed, normalized output shape — `output_text`, `reasoning_text`,
  `tool_calls`, `structured_output`, `content_parts` — produced at capture
  time, so no downstream consumer (corpus promotion, eval scoring, otel
  ingestion, dashboard display) has to re-derive or guess at structure from
  raw text.
- An adapter interface that scales to many providers *and* many integration
  styles (SDK-method-patching today; framework callback hooks like
  LangChain's tomorrow) without every new target bloating a shared file or
  inventing its own capture plumbing from scratch.
- Reasoning never enters the golden corpus. Traces stay available (on
  `Call`, for observability) but evals/optimise only ever see `output_text`.

## Non-goals

- Azure OpenAI, OpenRouter, and other provider-compatible-surface adapters —
  future registry entries, not built here.
- LlamaIndex, CrewAI, or other framework adapters beyond LangChain — future
  `CaptureMechanism` implementations reusing the same base, not built here.
- Retroactively re-deriving reasoning for `praximetry`-side SQLite rows.
  There are no pre-existing rows that matter (each is a fresh per-tenant
  local DB); this is a non-issue, not a deferred backfill.

## Design

### 1. Data model

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class ContentPart(BaseModel):
    type: Literal["text", "image", "audio", "file"]
    data: Any

class NormalizedOutput(BaseModel):
    output_text: str = ""            # final answer only
    reasoning_text: str = ""         # "" when the provider/model has none
    tool_calls: list[ToolCall] = []
    structured_output: Any | None = None
    content_parts: list[ContentPart] = []
    tokens_in: int = 0
    tokens_out: int = 0
```

`Call.response_text` is renamed to `Call.output_text` (its meaning changes —
was raw completion, now clean final-answer text) with new columns
`reasoning_text`, `tool_calls` (JSON), `structured_output` (JSON),
`content_parts` (JSON).

### 2. Two-layer adapter design

Capture mechanism (how we intercept a call) is separated from output
normalization (how we parse what came back), because they vary
independently — LangChain proves this: it has no `client.create()` method to
patch, only a callback/event stream.

```python
class OutputAdapter(ABC):
    """Pure, stateless: SDK/framework objects -> NormalizedOutput."""
    name: str
    def get_messages(self, kwargs: dict) -> list[dict]: ...
    def parse_response(self, resp: Any, model: str) -> NormalizedOutput: ...
    def accumulate(self, chunk: Any, state: dict) -> None: ...
    def finalize_stream(self, state: dict) -> NormalizedOutput: ...

class CaptureMechanism(ABC):
    """How we get invoked: monkeypatch, or hook a framework's own events."""
    def install(self, adapter: OutputAdapter) -> bool: ...
```

- `OpenAIAdapter`, `AnthropicAdapter`, `GeminiAdapter` implement
  `OutputAdapter` directly — today's `extractors.py` logic, restructured as
  methods returning `NormalizedOutput` instead of a `tuple[str, int, int]`.
  LiteLLM continues to reuse `OpenAIAdapter` (OpenAI-shaped responses).
- `MonkeypatchCapture` generalizes today's `_patch()`/`ProviderSpec`/
  `PatchTarget` machinery to take any `OutputAdapter`.
- `LangChainCallbackCapture` is new: a `BaseCallbackHandler` subclass whose
  `on_llm_end`/`on_chat_model_end` etc. build a `NormalizedOutput`. It
  delegates to the matching provider's `OutputAdapter` when the underlying
  model is identifiable (`response.llm_output.get("model_name")`), falling
  back to a generic text-only parse otherwise.

Registry: `ADAPTERS: dict[str, OutputAdapter]` keyed by provider name.
Model-specific behavior within a provider (e.g. detecting gpt-oss served via
Bedrock) is a prefix check inside that adapter, same convention as
`pricing.py`'s prefix-matching.

Rejected alternative: a single `ProviderAdapter` class per target that both
installs itself and parses responses (today's `_instrument()` shape,
extended). Fewer moving parts, but `install()` becomes provider-specific
with no real shared contract, and a second callback-based framework later
would duplicate callback-wiring logic instead of reusing a `CaptureMechanism`
base.

### 3. Reasoning pattern registry

For providers where reasoning is embedded in text with no structural
separation:

```python
_REASONING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai.gpt-oss", re.compile(r"^\s*<reasoning>(.*?)</reasoning>\s*", re.S)),
]

def split_embedded_reasoning(text: str, model: str) -> tuple[str, str]:
    """Returns (output_text, reasoning_text). No-op if no pattern matches."""
    for prefix, pattern in _REASONING_PATTERNS:
        if model.startswith(prefix):
            m = pattern.match(text)
            if m:
                return text[m.end():].lstrip(), m.group(1).strip()
    return text, ""
```

Called from `OutputAdapter.parse_response()`/`finalize_stream()` only for
providers without SDK-native separation. Providers with native separation
(Anthropic `thinking` blocks, OpenAI o1/o3 `reasoning` field) populate
`reasoning_text` directly and never consult this table. New tagged-reasoning
models are added here with no adapter code changes — same low-ceremony
pattern as adding a `pricing.py` entry.

### 4. Storage migration

**`praximetry` (SQLite):** `ALTER TABLE calls RENAME COLUMN response_text TO
output_text`, add `reasoning_text`, `tool_calls`, `structured_output`,
`content_parts` as nullable new columns. No backfill needed — SQLite DBs are
per-tenant and none of the current ones carry data worth preserving through
this change (see Non-goals).

**`praximetry-cloud` (Postgres golden corpus):** `example_from_call` changes
`expected=call.response_text` to `expected=call.output_text` — the corpus
never stores reasoning going forward. A one-time backfill script re-parses
existing `Example.expected` values that match a known reasoning-tag pattern
(reusing `split_embedded_reasoning()`), strips the reasoning, and updates in
place. Given the corpus is new (populated mostly through recent e2e/demo
testing), this is a handful of rows, not a migration-framework-worthy job.
Ships with a dry-run mode and is tested against a seeded Postgres fixture
before running against live staging.

### 5. Consumer updates

- `example_from_call`: `expected=call.output_text`.
- `scorers.py`'s `_REASONING_PREFIX` strip (already deployed, PR #42) stays
  as cheap, idempotent defense-in-depth — protects any example that reaches
  the corpus another way (e.g. local JSONL via `dataset.py`) — but is no
  longer load-bearing once capture-side splitting and the backfill ship.
- `otel.py`, dashboard call detail view, pricing display: read
  `output_text`/`reasoning_text`/`tool_calls` directly instead of parsing
  `response_text`.

### 6. Testing

- `OutputAdapter`s: pure-function-style unit tests (matches today's
  `extractors.py` convention) — fixtures for Anthropic `thinking` blocks, a
  gpt-oss `<reasoning>` fixture, a tool-call fixture, per adapter.
- `CaptureMechanism`s: one integration test each. `MonkeypatchCapture` via
  existing `FakeLLM`-style doubles. `LangChainCallbackCapture` via
  LangChain's own `FakeListLLM`/callback test utilities.
- `split_embedded_reasoning()`: direct unit test per registry entry.
- Backfill script: dry-run mode, tested against a seeded Postgres fixture
  (existing e2e Postgres fixture pattern) before running against live
  staging.

## Build scope

Built in this project:
- `OutputAdapter`/`CaptureMechanism` interfaces and registries.
- `OpenAIAdapter`, `AnthropicAdapter`, `GeminiAdapter` (LiteLLM reuses
  `OpenAIAdapter`) migrated onto the new interface, including the gpt-oss
  reasoning-tag pattern.
- `LangChainCallbackCapture` — one proof-of-extensibility adapter, chosen
  over an Azure OpenAI adapter because it validates the harder, more novel
  case (a genuinely different integration style), not just another
  provider-compatible surface.
- `praximetry-cloud` consumer updates: `example_from_call`, `otel.py`,
  dashboard display, pricing.
- Postgres backfill script.

Explicitly deferred (future registry/`CaptureMechanism` entries, no
placeholder code):
- Azure OpenAI, OpenRouter, and other provider-compatible-surface adapters.
- LlamaIndex, CrewAI, or other framework adapters.

## Rollout order

1. `NormalizedOutput`/adapter interfaces + today's 4 providers in
   `praximetry` (OSS) — ships and is tested independently.
2. `praximetry-cloud` consumer updates against the new `Call` shape.
3. Postgres backfill.
4. `LangChainCallbackCapture` — proof-of-extensibility piece, not blocking
   the bug-driving use case, so it lands last.
