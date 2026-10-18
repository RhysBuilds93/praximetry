# praximetry

Drop-in observability for LLM agents. Instrument once, and every LLM call your
agent makes — across OpenAI, Anthropic, LiteLLM, Gemini, or via OpenTelemetry —
is recorded locally: tokens, cost, latency, and the prompt/response, attributed
to the stage of your pipeline that made it.

```python
import praximetry as px

px.init(project="my-agent")   # auto-instruments openai/anthropic/litellm/gemini SDKs

@px.stage("summarize")
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
call `px.record_call(...)` directly inside a `@px.stage` function. LiteLLM's
`completion` entry point supports positional `model` and `messages` arguments,
including when an active model override is applied.

Patching is atomic per provider: if an SDK target is missing or cannot be
replaced, its other targets keep their original methods. Repeated
instrumentation leaves existing wrappers in place, so a completion is recorded
once even after an instrumentation retry.
LiteLLM module calls support positional model/messages and later positional
arguments (such as timeout), including model/prompt overrides and capture.

## Stages and concurrent runs

Bare `@px.stage` and `@px.stage("")` use the callable's `__name__`, or its type
name when no function name is available (`partial` for `functools.partial`,
for example). Choose explicit, unique names for unrelated stages: registering
a different module/qualified function name logs a warning and replaces the
previous registration. Reloading the same module/qualified name does not warn.

Without an explicit run, the first recorded call lazily creates one process-wide
run, shared by concurrent asyncio tasks and worker threads. Changing the
configured project or reopening the store starts a new process-wide run on its
next use, leaving existing run rows intact and clearing each context's old
parent-call ancestry when it observes the replacement run. Use
`praximetry.runtime.run_context(...)` to isolate separate requests or trials;
explicit runs take precedence over the process-wide run.

Asyncio tasks inherit the active context. Across thread/executor boundaries,
pass `praximetry.runtime.capture_context()` to the worker and enter
`restore_context(snapshot)` there to preserve the explicit run, stage, parent
call and model/prompt overrides. In the same process, restored contexts share
an unfinished Run, including its latest call and metadata, but each restoration
starts an independent ancestry branch. Context capture follows the caller's own
branch (including children it dispatched), not a sibling restored worker or
unrelated concurrent branch. When the caller has no local ancestry, capture uses
the shared run's latest call. Capturing a snapshot does not trigger run rollover
or clear local ancestry after configuration changes; the next recorded call does.
Unknown or finished run IDs create a detached local Run with the captured ID,
so late workers do not mutate a finalized parent Run or replace its finalized
metadata in cloud uploads.
Snapshots containing a `prompt_transform` callable preserve it by reference
and cannot be JSON-serialized as-is.

Run persistence failures are logged with a traceback without preventing entry
to `run_context` or masking an exception from its body. Manual
`px.record_call(**kwargs)` still validates its Call fields when recording is
disabled; disabling recording skips persistence, not input validation.

Raw OpenAI/Anthropic responses are parsed only for recognized SDK
`LegacyAPIResponse`/synchronous `APIResponse` wrappers. A response model's own
`parse` field or method is left untouched.

The LangChain callback handler retains the model from invocation parameters
(preferred over serialized defaults), including for failed calls. Each handler
keeps at most 1024 unfinished LLM starts so abandoned runs cannot accumulate
without bound. Older starts are evicted first; their late completions still
record, but lack the evicted prompt/model and original latency start time.


## What this is (and isn't)

This package is the open-source recording/observability layer: instrumentation,
local storage, cost calculation, and the `praximetry eval` CI-gate command
(which only captures request shapes — it never scores or judges anything
locally). Auto-optimization and the hosted dashboard are a separate product —
see [praximetry.io](https://praximetry.io).

`praximetry optimize` and `praximetry apply` operate on the configured project
(`PRAXIMETRY_PROJECT`, default `default`). Pass `--project` to target another
project when a stage name exists in more than one.

SQLite store paths are anchored when the store is created, so later working-directory changes do
not redirect connections opened by worker threads.

## Hosted eval and CI exit codes

`praximetry eval` imports the stage module, fetches hosted golden examples,
captures requests locally, and sends those request shapes to the hosted scorer.
Capture initializes the supported SDK patchers and stops intercepted calls
before the real model request. Providers or SDK entry points without a supported
patcher cannot be safely captured; do not run those stages with live provider
credentials.

The command exits `0` when quality meets the threshold, `1` when quality is
below it, and `2` when the gate cannot run (for example, an unreachable API,
invalid stage module, or no usable captures).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
