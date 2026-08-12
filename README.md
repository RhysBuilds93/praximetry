# praximetry

Drop-in observability for LLM agents. Instrument once, and every LLM call your
agent makes — across OpenAI, Anthropic, LiteLLM, Gemini, or via OpenTelemetry —
is recorded locally: tokens, cost, latency, and the prompt/response, attributed
to the stage of your pipeline that made it.

```python
import praximetry

praximetry.init(project="my-agent")   # auto-instruments openai/anthropic/litellm/gemini SDKs

@praximetry.stage("summarize")
def summarize(text):
    ...  # your existing code, untouched
```

Then:

```
praximetry summary
```

```
calls=142  tokens_in=88213  tokens_out=9120  cost=$4.71

  summarize                claude-opus-4-8         n=90     in=61204   out=5310   $3.20
  classify                 claude-haiku-4-5         n=52     in=27009   out=3810   $1.51
```

## Install

```
pip install praximetry
```

## No SDK dependency

`praximetry` doesn't require openai/anthropic/etc. to be installed — the
per-provider patchers import lazily, so `init()` only instruments what's
actually present. For manual instrumentation (or providers without a patcher),
call `praximetry.record_call(...)` directly inside a `@praximetry.stage`
function.

## What this is (and isn't)

This package is the open-source recording/observability layer: instrumentation,
local storage, cost calculation, and the `praximetry eval` CI-gate command
(which only captures request shapes — it never scores or judges anything
locally). Auto-optimization and the hosted dashboard are a separate product —
see [praximetry.io](https://praximetry.io).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
