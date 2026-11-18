# ADR 0004: Add stored-request replay without replacing stage capture

## Status

Accepted for the optional package workflow (PRA-195, PRA-198, PRA-203).

## Context

Automatic SDK recording does not require a rerunnable Python function. The
existing evaluator does: it imports customer stages and executes each hosted
call-level golden example until the first model request is captured. These
operations measure different things and must not be presented as equivalent.

## Decision

Keep `--stage` and `-m` fresh capture and legacy exit codes. Add `--label` for
local historical single-shot replay, with no customer module import. Reject
combining the two selectors. Bare optimize ranks the selected project's known-
price recorded targets by accumulated cost and submits the highest-cost target.

A Call carries its derived label and actual request tool definitions separately
from its explicit stage and response tool calls. Effective attribution is the
innermost explicit label/stage, otherwise the derived label. Derived identity
uses supported source/instruction/tool/model evidence, before override mutation;
it is deterministic for unchanged inputs, not durable across edits. Explicit
names are user-maintained logical targets. There is no fuzzy migration.

The Call keeps `stage` as the existing joined ancestry when explicit stages are
active and records the innermost explicit target separately. Its derived label
remains available when no explicit scope exists. This preserves existing
`current_stage()` and nested-stage consumers while giving selection, summary
diagnostics and scoped policies one unambiguous effective target.

Selection uses local Store Calls joined to their Run's project. Replay rejects
failed, incomplete/redacted, unsupported multi-turn and telemetry-only captures.
Request completeness must be explicitly recorded; old traces missing tools
cannot safely be interpreted as tool-free. Provider message content is preserved,
not normalized into a new cross-provider format.

For evaluation, a recorded Call's `metadata.example_id` must exactly match the
hosted call-level example in the selected project and effective target. Use the
newest matching Call. Never associate a recent arbitrary request with the first
golden example. Fresh capture instead obtains that association by running the
actual example input in the customer's process. Optimization may submit a
request identified by its Call ID without asserting golden association; the
backend decides whether it can trial that request. Existing client methods and
HTTP request fields remain unchanged. Run identity remains in the local Call;
no new run/example fields are added to the hosted wire model.

CLI replay reports explicitly say historical requests do not validate current
agent code. Current checkout provenance is not attached to historical replay.
Scoring an already-recorded response is distinct again: replay asks the selected
backend to execute the request, not merely judge the stored response.

`label(name)` only enters an explicit context; it does not register a function.
`current_stage()` retains its ancestry representation for context consumers;
new effective attribution uses the innermost scope. Context capture/restore and
async inheritance remain the existing propagation mechanism.

`apply --label` stores policies under `projects[project][target]`. It preserves
legacy `stages` entries exactly without interpreting ambiguous global keys as
scoped keys. Legacy `apply --stage` continues writing those entries. Scoped SDK
loading does not migrate or consume legacy entries. In-flight overrides and
policy hooks retain priority. Exact unmatched targets send the original request.
Materialized system rewrites and model swaps are executable locally; unsupported
named hosted transforms fail at apply rather than pretending to execute.

OTel grouping uses available span evidence in a distinct identity namespace;
missing source identity is not invented. Structured prompt roles are retained;
opaque prompts are not reclassified as system instructions. Telemetry cannot mutate outbound SDK
requests or apply policies. Customer re-execution remains local; backend model
execution uses that deployment's credentials, without authentication changes.

## Consequences

Decorator-free recording, optimization and correlated evaluation are additive.
Old incomplete traces must be recaptured. Explicit labels are recommended for
durable policies; changed derived labels need deliberate new optimization/apply.
Existing ancestry consumers keep their API, while persisted nested attribution
now follows the innermost explicit target.

Compatibility coverage includes fresh stage capture, legacy override writes,
project separation, request tool persistence, exact golden association, nested
async labels, original identity through mutation, policy priority and read-only
OTel behavior.

## Rejected alternatives

- Replacing decorators: removes the only existing fresh-code capture contract.
- Assigning every golden example the latest request: scores unrelated inputs.
- Backend trace selection or a new backend abstraction: unnecessary HTTP/API
  changes; the configured client already accepts captured request shapes.
- Guessing tools or migrating unscoped overrides: silently changes executable
  requests or cross-project policy meaning.
- Calling historical replay a CI proof of current code: retrieval, surrounding
  tools and changed code did not execute.
