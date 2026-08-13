"""Workflow 7: a graph agent observed purely through OpenTelemetry.

This is the "we didn't write a connector for your framework" example. Nothing
below imports praximetry at the call site: the graph nodes emit ordinary OTel
GenAI spans, exactly like LangChain/LangGraph, LlamaIndex, CrewAI, DSPy or
PydanticAI do (natively, or via OpenLLMetry / OpenInference). praximetry registers
one span processor and picks all of it up - span name becomes the stage, and
`gen_ai.*` / `llm.*` attributes become the model, provider, tokens and cost.

The trade-off worth knowing: OTel is *read-only* telemetry. It gives you
observe -> evaluate -> recommend across every framework. To let the optimizer
*apply* a winner in-flight (swap the model, slim the prompt) you still want
`px.init()`, because you cannot mutate a request from a telemetry span.

Try:
    python -m praximetry.examples.workflows.otel_graph_agent
    praximetry summary
    praximetry-cloud detect
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

import praximetry as px

trace.set_tracer_provider(TracerProvider())
px.instrument_otel()          # one call; no per-framework connector
tracer = trace.get_tracer("shipping-graph")

# A "graph": nodes with a model each, the way a real LangGraph app is wired.
GRAPH = [
    # (node name, model, provider, prompt tokens, completion tokens)
    ("route_intent",      "gpt-4o",           "openai",    1400, 12),
    ("retrieve_context",  "gpt-4o",           "openai",     900, 40),
    ("draft_response",    "claude-opus-4-8",  "anthropic", 2200, 260),
    ("check_policy",      "claude-opus-4-8",  "anthropic", 1800, 8),
    ("finalise",          "claude-opus-4-8",  "anthropic",  600, 120),
]

QUERIES = [
    "Where is my parcel, tracking DX8812?",
    "I need to change the delivery address on DX9001",
    "The courier marked DX7740 as delivered but nothing arrived",
    "Can I upgrade DX6650 to next-day?",
    "Refund me for the late delivery on DX5512",
]


def run_graph(query: str) -> None:
    """Every node emits a GenAI span. praximetry is not called anywhere in here."""
    with tracer.start_as_current_span("graph.invoke") as root:
        root.set_attribute("graph.query", query)      # not a GenAI span -> ignored
        for node, model, provider, tin, tout in GRAPH:
            with tracer.start_as_current_span(node) as span:
                span.set_attribute("gen_ai.system", provider)
                span.set_attribute("gen_ai.request.model", model)
                span.set_attribute("gen_ai.usage.input_tokens", tin)
                span.set_attribute("gen_ai.usage.output_tokens", tout)
                span.set_attribute("gen_ai.prompt", query)


if __name__ == "__main__":
    for q in QUERIES:
        run_graph(q)
    trace.get_tracer_provider().force_flush()

    from praximetry.store import get_store
    from praximetry.currency import fmt

    totals = get_store().totals()
    print(f"  {totals['n']} LLM calls ingested from OTel spans, {fmt(totals['cost'])} total")
    for row in get_store().stage_summary():
        print(f"    {row['stage']:18s} {row['model']:18s} {row['n']:2d} calls  {fmt(row['cost'])}")
    print("\nNo connector, no decorators. Try: praximetry-cloud detect")
