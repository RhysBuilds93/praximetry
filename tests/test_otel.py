"""OpenTelemetry ingestion: attribute mapping across conventions + live spans."""
from praximetry import otel
from praximetry.store import get_store


# -- pure attribute mapping (no OTel SDK needed) -----------------------------

def test_map_gen_ai_semconv():
    call = otel.map_span("chat gpt-4o", {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 30,
    })
    assert call.provider == "openai" and call.model == "gpt-4o"
    assert call.input_tokens == 120 and call.output_tokens == 30
    assert call.cost_usd > 0
    assert call.metadata["source"] == "otel"


def test_map_openinference_convention():
    call = otel.map_span("llm", {
        "llm.model_name": "claude-sonnet-5",
        "llm.provider": "anthropic",
        "llm.token_count.prompt": 50,
        "llm.token_count.completion": 8,
    })
    assert call.model == "claude-sonnet-5" and call.provider == "anthropic"
    assert call.input_tokens == 50 and call.output_tokens == 8


def test_map_traceloop_prompt_tokens_alias():
    call = otel.map_span("openai.chat", {
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.usage.prompt_tokens": 200,
        "gen_ai.usage.completion_tokens": 40,
    })
    assert call.input_tokens == 200 and call.output_tokens == 40


def test_span_name_becomes_stage():
    call = otel.map_span("summarize_node", {
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 2,
    })
    assert call.stage == "summarize_node"  # framework node name -> stage


def test_operation_name_overrides_stage():
    call = otel.map_span("span-123", {
        "gen_ai.operation.name": "classify",
        "gen_ai.request.model": "gpt-4o",
    })
    assert call.stage == "classify"


def test_non_llm_span_ignored():
    assert otel.map_span("http.request", {"http.method": "GET", "http.status_code": 200}) is None


def test_record_spans_persists():
    n = otel.record_spans([
        {"name": "retrieve", "attributes": {"db.system": "pg"}},  # ignored
        {"name": "generate", "attributes": {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 20}},
    ])
    assert n == 1
    calls = get_store().calls()
    assert len(calls) == 1 and calls[0].stage == "generate" and calls[0].provider == "otel"


# -- live span through a real TracerProvider ---------------------------------

def test_span_processor_records_real_span():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    provider.add_span_processor(otel.make_span_processor())
    tracer = trace.get_tracer("test", tracer_provider=provider)

    with tracer.start_as_current_span("chat_completion") as span:
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute("gen_ai.request.model", "claude-haiku-4-5")
        span.set_attribute("gen_ai.usage.input_tokens", 300)
        span.set_attribute("gen_ai.usage.output_tokens", 25)

    provider.force_flush()
    calls = get_store().calls()
    assert len(calls) == 1
    c = calls[0]
    assert c.model == "claude-haiku-4-5" and c.provider == "anthropic"
    assert c.input_tokens == 300 and c.output_tokens == 25
    assert c.stage == "chat_completion" and c.cost_usd > 0


def test_instrument_otel_rejects_bad_provider():
    import pytest
    with pytest.raises(RuntimeError, match="TracerProvider"):
        otel.instrument_otel(tracer_provider=object())
