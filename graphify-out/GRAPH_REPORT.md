# Graph Report - .  (2026-08-13)

## Corpus Check
- Corpus is ~19,308 words - fits in a single context window. You may not need a graph.

## Summary
- 482 nodes · 1041 edges · 23 communities (22 shown, 1 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.64)
- Token cost: 0 input · 46,854 output

## Community Hubs (Navigation)
- Example Agent Workflows
- Hosted Eval Client
- SDK Patch & Provider Wrapping
- Eval Capture & Dataset
- SQLite Store
- Live SDK & Patch Tests
- CLI Commands
- CI & README
- CLI Eval Tests
- CLI Optimize Tests
- OpenTelemetry Ingestion
- Config & Init
- Runtime Core & Tests
- Core Models & Runtime
- Pricing & Demo Agent
- Provider Response Extractors
- OTel Tests
- Test Fixtures (FakeLLM)
- Stage Decorator & Policy Scope
- OTel Graph Agent Example
- Package Root

## God Nodes (most connected - your core abstractions)
1. `stage()` - 48 edges
2. `CloudClient` - 34 edges
3. `real_chat()` - 32 edges
4. `get_store()` - 30 edges
5. `Example` - 24 edges
6. `default_model()` - 23 edges
7. `Store` - 23 edges
8. `capture_request()` - 21 edges
9. `Call` - 15 edges
10. `record_call()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `test_stage_decorator_registers_and_attributes()` --calls--> `get_store()`  [EXTRACTED]
  tests/test_core.py → src/praximetry/store.py
- `test_never_persists_a_call()` --calls--> `get_store()`  [INFERRED]
  tests/test_eval_capture.py → src/praximetry/store.py
- `test_record_spans_persists()` --calls--> `get_store()`  [EXTRACTED]
  tests/test_otel.py → src/praximetry/store.py
- `test_span_processor_records_real_span()` --calls--> `get_store()`  [EXTRACTED]
  tests/test_otel.py → src/praximetry/store.py
- `CI Workflow` --conceptually_related_to--> `praximetry README`  [INFERRED]
  .github/workflows/ci.yml → README.md

## Import Cycles
- 3-file cycle: `src/praximetry/instrument/__init__.py -> src/praximetry/instrument/patch.py -> src/praximetry/instrument/providers.py -> src/praximetry/instrument/__init__.py`
- 3-file cycle: `src/praximetry/__init__.py -> src/praximetry/instrument/__init__.py -> src/praximetry/instrument/patch.py -> src/praximetry/__init__.py`

## Hyperedges (group relationships)
- **CI Pipeline Flow (checkout -> setup-python -> install -> lint -> test)** — github_workflows_ci_checkout_step, github_workflows_ci_setup_python_step, github_workflows_ci_pip_install_step, github_workflows_ci_ruff_check_step, github_workflows_ci_pytest_step [EXTRACTED 1.00]
- **Lazy-Import Multi-Provider Support** — readme_openai, readme_anthropic, readme_litellm, readme_gemini, readme_no_sdk_dependency [INFERRED 0.85]

## Communities (23 total, 1 thin omitted)

### Community 0 - "Example Agent Workflows"
Cohesion: 0.06
Nodes (77): F, OpenAI, _categorize(), correlate(), critique_postmortem(), draft_postmortem(), fetch_alerts(), fetch_logs() (+69 more)

### Community 1 - "Hosted Eval Client"
Cohesion: 0.07
Nodes (38): Client, Response, CapturedRequest, BaseModel, client_from_env(), CloudClient, CloudError, RuntimeError (+30 more)

### Community 2 - "SDK Patch & Provider Wrapping"
Cohesion: 0.07
Nodes (31): Accumulate, OnDone, _apply_overrides(), capturing(), _instrument(), _patch(), Any, Auto-instrumentation: monkey-patch OpenAI, Anthropic, and LiteLLM SDKs. Covers… (+23 more)

### Community 3 - "Eval Capture & Dataset"
Cohesion: 0.11
Nodes (29): Exception, call_stage(), capture_request(), CaptureError, _CaptureSignal, Any, RuntimeError, Capture the request shape a stage would send to an LLM, without sending it.… (+21 more)

### Community 4 - "SQLite Store"
Cohesion: 0.16
Nodes (10): Connection, Row, EvalResult, Experiment, One variant tried by the optimization loop., Any, Path, Write a batch of serialized runs/calls/experiments/eval_results. (+2 more)

### Community 5 - "Live SDK & Patch Tests"
Cohesion: 0.11
Nodes (20): skipif, override_context(), get_store(), _anthropic_json(), _openai_json(), Full-stack verification against REAL SDK clients. Uses httpx.MockTransport so…, test_anthropic_real_client_async_buffered(), test_anthropic_real_client_buffered() (+12 more)

### Community 6 - "CLI Commands"
Cohesion: 0.11
Nodes (21): callback, command, apply(), eval_cmd(), _import_module(), _main(), optimize(), praximetry CLI: summary | eval `eval` is the customer-facing half of the hosted… (+13 more)

### Community 7 - "CI & README"
Cohesion: 0.13
Nodes (21): CI Workflow, actions/checkout@v4 Step, pip install -e ".[dev]" Step, pytest -q Step, Python Version Matrix (3.10, 3.11, 3.12), ruff check . Step, actions/setup-python@v5 Step, test Job (+13 more)

### Community 8 - "CLI Eval Tests"
Cohesion: 0.21
Nodes (16): FastAPI, `praximetry eval` CLI command, end to end against a stub hosted API. Corpus…, No --project: the pre-existing default (0.9) applies unchanged, and the config…, _register_project_stages(), _stub_app(), test_eval_empty_corpus_is_unusable(), test_eval_explicit_fail_under_overrides_fetched_config(), test_eval_fails_under_threshold() (+8 more)

### Community 9 - "CLI Optimize Tests"
Cohesion: 0.26
Nodes (14): _patch_client(), FastAPI, `praximetry optimize` and `praximetry apply` CLI commands, end to end against a…, `winner="missing"` (the default) means /api/optimize/winner 404s — no completed…, _stub_app(), test_apply_no_run_yet_is_an_error(), test_apply_no_winner_exits_zero_and_does_not_write(), test_apply_preserves_other_stages_in_overrides_json() (+6 more)

### Community 10 - "OpenTelemetry Ingestion"
Cohesion: 0.23
Nodes (15): _first(), instrument_otel(), _int(), is_genai_span(), make_span_processor(), map_span(), Any, OpenTelemetry ingestion. One receiver, not one-connector-per-framework: any… (+7 more)

### Community 11 - "Config & Init"
Cohesion: 0.23
Nodes (10): Config, get_config(), Global runtime configuration for praximetry., set_config(), init(), Path, praximetry — drop-in observability for LLM agents. Quickstart: import…, Initialize praximetry. Returns list of auto-instrumented providers. (+2 more)

### Community 12 - "Runtime Core & Tests"
Cohesion: 0.17
Nodes (11): current_stage(), Persist an LLM call. Used by patchers; also public for manual logging., Open a fresh Run (used per eval example / experiment trial)., record_call(), run_context(), Core: pricing, store, decorators, recording., test_parent_call_id_auto_links_concurrent_fanout(), test_parent_call_id_chains_sequential_calls() (+3 more)

### Community 13 - "Core Models & Runtime"
Cohesion: 0.22
Nodes (10): Call, Opportunity, BaseModel, A traced execution session (process lifetime or explicit run)., A detected optimization opportunity., Run, Runtime context: current run, current stage, active experiment overrides., Register (or clear, with None) the policy hook applied around every @stage call. (+2 more)

### Community 14 - "Pricing & Demo Agent"
Cohesion: 0.21
Nodes (10): classify(), draft_reply(), Demo agent: a tiny support-ticket pipeline instrumented with praximetry. Runs…, Offline stand-in for a chat API., _sim_llm(), _record(), cost_usd(), _lookup() (+2 more)

### Community 15 - "Provider Response Extractors"
Cohesion: 0.36
Nodes (12): anthropic_accumulate(), anthropic_messages(), anthropic_response(), _g(), gemini_accumulate(), gemini_messages(), gemini_response(), openai_accumulate() (+4 more)

### Community 16 - "OTel Tests"
Cohesion: 0.18
Nodes (3): OpenTelemetry ingestion: attribute mapping across conventions + live spans., test_record_spans_persists(), test_span_processor_records_real_span()

### Community 17 - "Test Fixtures (FakeLLM)"
Cohesion: 0.20
Nodes (8): Testing hook: force re-open against current config., reset_store(), fake_llm(), FakeLLM, fresh_env(), fixture, Simulates an instrumented SDK: honors overrides, records calls. Cheaper models…, Isolated DB + clean runtime state per test. PRAXIMETRY_DB is also set via env…

### Community 18 - "Stage Decorator & Policy Scope"
Cohesion: 0.38
Nodes (6): Any, Developer-facing decorators. Minimal input: one decorator per agent stage., _wrap(), policy_scope(), Apply the registered policy hook for `stage`, if any (no-op otherwise)., stage_context()

### Community 19 - "OTel Graph Agent Example"
Cohesion: 0.50
Nodes (3): Workflow 7: a graph agent observed purely through OpenTelemetry. This is the…, Every node emits a GenAI span. praximetry is not called anywhere in here., run_graph()

## Knowledge Gaps
- **6 isolated node(s):** `praximetry`, `Python Version Matrix (3.10, 3.11, 3.12)`, `actions/checkout@v4 Step`, `praximetry summary Command`, `GPL-3.0-or-later License` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `stage()` connect `Example Agent Workflows` to `Stage Decorator & Policy Scope`, `Config & Init`, `Pricing & Demo Agent`?**
  _High betweenness centrality (0.129) - this node is a cross-community bridge._
- **Why does `get_store()` connect `Live SDK & Patch Tests` to `Eval Capture & Dataset`, `SQLite Store`, `CLI Commands`, `OpenTelemetry Ingestion`, `Runtime Core & Tests`, `Core Models & Runtime`, `OTel Tests`, `OTel Graph Agent Example`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `Call` connect `Core Models & Runtime` to `Hosted Eval Client`, `SQLite Store`, `OpenTelemetry Ingestion`, `Config & Init`, `Runtime Core & Tests`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `CloudClient` (e.g. with `CapturedRequest` and `Dataset`) actually correct?**
  _`CloudClient` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `Example` (e.g. with `CapturedRequest` and `CaptureError`) actually correct?**
  _`Example` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `praximetry`, `Python Version Matrix (3.10, 3.11, 3.12)`, `actions/checkout@v4 Step` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Example Agent Workflows` be split into smaller, more focused modules?**
  _Cohesion score 0.06201550387596899 - nodes in this community are weakly interconnected._