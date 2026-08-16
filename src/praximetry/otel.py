"""OpenTelemetry ingestion.

One receiver, not one-connector-per-framework: any library that emits GenAI
spans (via native support, OpenLLMetry/traceloop, or OpenInference/Phoenix) is
captured by mapping its span attributes onto praximetry's `Call` model. This gives
observability across LangChain, LlamaIndex, DSPy, PydanticAI, CrewAI, etc.

Two ways in:
  * `instrument_otel()` — register an in-process span processor on the current
    TracerProvider; every finished GenAI span is recorded automatically.
  * `record_spans([...])` — ingest already-collected spans (dicts of
    {name, attributes}); used for offline/batch import and by tests.

Note: OTel is read-only telemetry. It powers observe + evaluate + *recommend*.
The optimizer's in-flight apply/override still needs SDK-level instrumentation
(praximetry.init), since you cannot mutate a request from a telemetry span.
"""
from __future__ import annotations

from typing import Any

from . import pricing
from .models import Call
from .runtime import current_run

# Attribute aliases across the common conventions (OTel GenAI semconv,
# OpenInference, OpenLLMetry/traceloop). First match wins.
_MODEL = ("gen_ai.response.model", "gen_ai.request.model", "llm.model_name",
          "llm.invocation_parameters.model", "model")
_PROVIDER = ("gen_ai.system", "llm.system", "llm.provider")
_IN_TOKENS = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens",
              "llm.token_count.prompt", "llm.usage.prompt_tokens")
_OUT_TOKENS = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens",
               "llm.token_count.completion", "llm.usage.completion_tokens")
_PROMPT = ("gen_ai.prompt", "llm.input_messages", "llm.prompts")
_COMPLETION = ("gen_ai.completion", "llm.output_messages", "llm.completion")


def _first(attrs: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in attrs and attrs[k] not in (None, ""):
            return attrs[k]
    return default


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def is_genai_span(attrs: dict[str, Any]) -> bool:
    """A span is LLM-related if it carries a model or GenAI/LLM attributes."""
    if _first(attrs, _MODEL):
        return True
    return any(k.startswith(("gen_ai.", "llm.")) for k in attrs)


def map_span(name: str, attributes: dict[str, Any]) -> Call | None:
    """Map one span's attributes onto a Call, or None if it isn't an LLM span."""
    if not is_genai_span(attributes):
        return None
    model = str(_first(attributes, _MODEL, "unknown"))
    provider = str(_first(attributes, _PROVIDER, "otel"))
    tin = _int(_first(attributes, _IN_TOKENS, 0))
    tout = _int(_first(attributes, _OUT_TOKENS, 0))
    # Span name is the framework's node/step name -> natural stage attribution.
    stage = str(_first(attributes, ("gen_ai.operation.name",), name) or name)
    run = current_run()
    return Call(
        run_id=run.id if run else "otel",
        stage=stage,
        provider=provider,
        model=model,
        messages=[{"role": "prompt", "content": str(_first(attributes, _PROMPT, ""))}],
        output_text=str(_first(attributes, _COMPLETION, "")),
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=pricing.cost_usd(model, tin, tout),
        metadata={"source": "otel", "span": name},
    )


def record_spans(spans: list[dict[str, Any]]) -> int:
    """Ingest a list of {name, attributes} span dicts. Returns count recorded."""
    from .store import get_store

    store = get_store()
    n = 0
    for s in spans:
        call = map_span(s.get("name", "span"), s.get("attributes", {}) or {})
        if call is not None:
            store.save_call(call)
            n += 1
    return n


def make_span_processor():
    """Build a SpanProcessor that records finished GenAI spans (needs opentelemetry-sdk)."""
    try:
        from opentelemetry.sdk.trace.export import SpanProcessor
    except ImportError as e:  # pragma: no cover - exercised via message only
        raise RuntimeError(
            "opentelemetry-sdk is required for OTel instrumentation. "
            "Install with: pip install 'praximetry[otel]'"
        ) from e

    from .store import get_store

    class PraximetrySpanProcessor(SpanProcessor):
        def on_start(self, span, parent_context=None):  # noqa: D401
            pass

        def on_end(self, span) -> None:
            attrs = dict(span.attributes or {})
            call = map_span(span.name, attrs)
            if call is not None:
                get_store().save_call(call)

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    return PraximetrySpanProcessor()


def instrument_otel(tracer_provider: Any = None) -> None:
    """Register praximetry's span processor on a TracerProvider (or the global one)."""
    if tracer_provider is None:
        from opentelemetry import trace

        tracer_provider = trace.get_tracer_provider()
    if not hasattr(tracer_provider, "add_span_processor"):
        raise RuntimeError(
            "No usable TracerProvider. Set one up first, e.g.:\n"
            "  from opentelemetry.sdk.trace import TracerProvider\n"
            "  from opentelemetry import trace\n"
            "  trace.set_tracer_provider(TracerProvider())\n"
            "then call praximetry.instrument_otel()."
        )
    tracer_provider.add_span_processor(make_span_processor())
