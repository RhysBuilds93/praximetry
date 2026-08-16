# Provider adapter layer

Status: draft · Spans `praximetry` + `praximetry-cloud`

## Problem

Extraction collapses every response into `response_text`. Reasoning
(structural, e.g. Anthropic `thinking` blocks, or embedded, e.g. gpt-oss
`<reasoning>` tags), tool calls, structured output, and multimodal parts are
all dropped or mixed into that one string. Root cause of the PR #42 bug:
promoted `expected` included a reasoning trace; replaying produced a
different trace, tanking the structural scorer on a correct answer.

## Goals

- Normalize output at capture time into fixed fields.
- Adapter interface that scales to many providers *and* integration styles
  (SDK patch vs. framework callbacks).
- Reasoning never enters the golden corpus.

## Non-goals

- Azure/OpenRouter/other provider adapters, LlamaIndex/CrewAI adapters —
  future registry entries only.
- Backfilling `praximetry`'s SQLite — no existing rows worth keeping.

## Design

**Data model**
```python
class NormalizedOutput(BaseModel):
    output_text: str = ""
    reasoning_text: str = ""
    tool_calls: list[ToolCall] = []
    structured_output: Any | None = None
    content_parts: list[ContentPart] = []
    tokens_in: int = 0
    tokens_out: int = 0
```
`Call.response_text` → `Call.output_text` + new `reasoning_text`,
`tool_calls`, `structured_output`, `content_parts` columns.

**Two-layer adapters** — capture (how we intercept) is separate from
normalization (how we parse), since LangChain has no method to patch, only
callbacks.
```python
class OutputAdapter(ABC):      # pure parsing, per provider
    def get_messages(kwargs) -> list[dict]: ...
    def parse_response(resp, model) -> NormalizedOutput: ...
    def accumulate(chunk, state) -> None: ...
    def finalize_stream(state) -> NormalizedOutput: ...

class CaptureMechanism(ABC):   # how we get invoked
    def install(adapter: OutputAdapter) -> bool: ...
```
`MonkeypatchCapture` generalizes today's `_patch()`. `LangChainCallbackCapture`
is new — a `BaseCallbackHandler` that delegates to the matching provider's
`OutputAdapter` when identifiable, else parses generically.

Rejected: fused `ProviderAdapter.install()` per target (today's shape,
extended) — `install()` ends up provider-specific with no shared contract,
and a second callback framework would duplicate wiring logic.

**Reasoning pattern registry** — for providers without structural
separation:
```python
_REASONING_PATTERNS = [("openai.gpt-oss", re.compile(r"^\s*<reasoning>(.*?)</reasoning>\s*", re.S))]
def split_embedded_reasoning(text, model) -> tuple[str, str]: ...  # no-op if no match
```
Same low-ceremony convention as `pricing.py` prefix matching. Providers with
native separation (Anthropic `thinking`, o1/o3 `reasoning`) skip this.

**Storage migration**
- SQLite: rename column, add new ones. No backfill.
- Postgres corpus: `example_from_call` → `expected=call.output_text`.
  One-time backfill script strips existing contaminated rows via
  `split_embedded_reasoning()`, dry-run mode, tested against seeded fixture
  first.

**Consumers**: `example_from_call`, `otel.py`, dashboard, pricing read the
new fields directly. `scorers.py`'s existing strip (PR #42) stays as
harmless defense-in-depth.

**Testing**: adapters unit-tested like today's `extractors.py` (fixtures per
reasoning/tool-call case); capture mechanisms get one integration test each
(`FakeLLM` doubles; LangChain's `FakeListLLM` for callbacks); backfill tested
dry-run against a seeded fixture before staging.

## Build scope

Built: adapter/registry interfaces, `OpenAIAdapter`/`AnthropicAdapter`/
`GeminiAdapter` (LiteLLM reuses OpenAI's), `LangChainCallbackCapture` (proof
of extensibility — picked over Azure since it's a genuinely different
integration style), cloud consumer updates, Postgres backfill.

Deferred: Azure/OpenRouter, LlamaIndex/CrewAI — no placeholder code.

## Rollout

1. OSS: `NormalizedOutput` + adapters for today's 4 providers.
2. Cloud consumer updates.
3. Postgres backfill.
4. `LangChainCallbackCapture`.
