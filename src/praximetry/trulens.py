"""TruLens ingestion.

TruLens doesn't emit LLM calls as OTel spans in the widely-installed 1.x API --
cost/tokens accumulate on a `Record.cost` (`trulens.core.schema.base.Cost`)
wrapping a whole app invocation, which may cover several LLM calls. That Cost
object has no model field (confirmed against the TruLens source), so unlike
the other adapters, the caller must supply `model` themselves.
"""

from __future__ import annotations

from typing import Any

from . import pricing
from .models import Call
from .runtime import current_run


def record_trulens_record(
    record: Any,
    *,
    model: str,
    provider: str = "trulens",
    stage: str | None = None,
) -> Call:
    """Record one TruLens `Record`'s cost/tokens as a Call.

    `record.cost` is un-attributed totals for the whole app invocation, not a
    single LLM call -- expect a Call per Record, not per underlying request.
    """
    from .store import get_store

    cost = record.cost
    tin = int(getattr(cost, "n_prompt_tokens", 0) or 0)
    tout = int(getattr(cost, "n_completion_tokens", 0) or 0)
    run = current_run()
    metadata: dict[str, Any] = {"source": "trulens", "app_id": getattr(record, "app_id", None)}
    if pricing.is_unpriced(model):
        metadata["unpriced_model"] = True
    call = Call(
        run_id=run.id if run else "trulens",
        stage=stage,
        provider=provider,
        model=model,
        output_text=str(getattr(record, "main_output", "") or ""),
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=pricing.cost_usd(model, tin, tout),
        metadata=metadata,
    )
    get_store().save_call(call)
    return call
