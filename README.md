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

Set `PRAXIMETRY_API_KEY` to a dashboard-issued API key and
`PRAXIMETRY_API_URL` to the hosted API's HTTPS URL (or let Databricks discover
the app URL). The client **refuses** non-HTTPS URLs rather than sending the
bearer key insecurely. Plain HTTP is allowed only for local development on
`localhost`, `127.0.0.1`, or `::1`.

```bash
praximetry eval --project my-agent --module my_agent --fail-under 0.9 --fail-on-regression
```

The command exits `0` when quality meets the threshold, `1` when quality is
below it, and `2` when the gate cannot run (for example, an unreachable API,
invalid config or stage module, or no usable captures). With
`--fail-on-regression`, any example in the hosted comparison's `regressed`
list also causes exit `1`, **even when aggregate quality passes**. In project
mode this includes regressions in any evaluated stage. If there is no
baseline, it prints `no baseline` and still applies the quality threshold.
Without the flag, regressions do not change the exit code. Whenever a
comparison is returned, the CLI prints its baseline, overlap, added/removed
counts, quality delta, and a warning identifying any regressed examples.
If the baseline and current run have no overlapping scored examples, comparison
quality and delta are shown as `n/a`. With nothing comparable, there is no
regression failure, even with `--fail-on-regression`; the current run's quality
threshold still applies.
Nonzero server-reported `unscored` counts are printed per stage in both stage
and project modes; this is informational and does not add another gate.

Captures include optional `git_branch`, `git_commit`, and `default_branch`
metadata. Detection is independent for each field:

1. Explicit `--branch`, `--commit`, or `--default-branch` overrides.
2. GitHub Actions: `GITHUB_HEAD_REF` (PR branch), then `GITHUB_REF_NAME`;
   `GITHUB_SHA`; and `repository.default_branch` from the JSON event file at
   `GITHUB_EVENT_PATH`.
3. The local checkout: `git rev-parse --abbrev-ref HEAD`, `git rev-parse HEAD`,
   and `git symbolic-ref refs/remotes/origin/HEAD`, respectively.

Unresolved fields are omitted: missing git, a non-repository directory, or a
missing remote default never blocks evaluation. A detached checkout omits the
local branch but can still report its commit. For a different CI provider,
override only what it cannot discover:

```bash
praximetry eval --project my-agent --module my_agent \
  --branch feature/new-prompt --commit abc123 --default-branch main \
  --fail-on-regression
```

With no `--stage`, `--project`, or `--config`, the command uses the saved
dashboard default. Only the cloud's `{"detail":"no default eval saved"}` 404
response means no default is saved. Other 404s (including FastAPI's generic
`{"detail":"Not Found"}`) and non-JSON/HTML responses report an API URL or
server-version problem instead. API error body previews are capped at 500
characters, with `...` marking truncation.

### GitHub Actions example

In your agent repository, install the application and its dependencies and
point `--module` at the module registering your stages (`my_agent` below).
Set the repository variable `PRAXIMETRY_API_URL` and secret
`PRAXIMETRY_API_KEY`. Git metadata is detected automatically, including a PR's
head branch when checkout is detached:

```yaml
name: Agent eval
on:
  pull_request:
  push:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e . praximetry
      - name: Gate quality and per-example regressions
        env:
          PRAXIMETRY_API_URL: ${{ vars.PRAXIMETRY_API_URL }}
          PRAXIMETRY_API_KEY: ${{ secrets.PRAXIMETRY_API_KEY }}
        run: >
          praximetry eval --project my-agent --module my_agent
          --fail-under 0.9 --fail-on-regression
```

The job needs a usable API key; fork pull requests normally cannot access
repository secrets. Only run agent code you trust with that key, and do not
use `pull_request_target` to give untrusted pull-request code access to secrets.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
