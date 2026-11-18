"""Local single-request replay; never imports or executes customer code."""

from __future__ import annotations

from .. import pricing
from ..models import Call
from ..store import get_store
from .capture import CaptureError, CapturedRequest
from .dataset import Example


def captured_request_from_call(call: Call, *, example_id: str | None = None) -> CapturedRequest:
    """Map a complete stored request without inventing missing request fields.

    ``example_id`` must come from recorded correlation, never from the position
    of an example in a corpus. Optimization may use the call ID when no golden
    association is required; evaluation must check association separately.
    """
    target = call.target or call.stage or call.derived_label
    reason = None
    if not target:
        reason = "no explicit or derived target"
    elif call.error:
        reason = "the recorded request failed"
    elif call.metadata.get("request_complete") is not True:
        reason = "request completeness was not recorded (legacy traces may omit tool definitions)"
    elif call.metadata.get("redacted") or call.metadata.get("incomplete"):
        reason = "request data is redacted or incomplete"
    elif call.metadata.get("multi_turn"):
        reason = "multi-turn interactions cannot be replayed from one request"
    elif call.provider in ("unknown", "otel") or call.model == "unknown":
        reason = "provider or model is unknown"
    elif not call.messages or any(
        not isinstance(message, dict)
        or message.get("role") not in ("system", "developer", "user")
        or not message.get("content")
        for message in call.messages
    ):
        reason = "missing messages or unsupported multi-turn/provider message shape"
    elif call.metadata.get("source") == "otel":
        reason = "OTel telemetry does not establish a complete executable request"
    if reason:
        raise CaptureError(
            f"Call '{call.id}' is not eligible for stored-request replay: {reason}. "
            "Record a complete single-shot request or use --stage NAME -m your_module."
        )
    return CapturedRequest(
        example_id=example_id or str(call.metadata.get("example_id") or call.id),
        stage=target,
        provider=call.provider,
        model=call.model,
        messages=call.messages,
        tools=call.tools,
    )


def recorded_calls(project: str, target: str | None = None) -> list[Call]:
    if not project:
        raise CaptureError("A non-empty project is required for stored-request replay.")
    calls = get_store().calls(project=project, limit=-1)
    known = sorted({name for call in calls if (name := call.target or call.stage or call.derived_label)})
    if target is not None:
        calls = [call for call in calls if (call.target or call.stage or call.derived_label) == target]
        if not calls:
            raise CaptureError(
                f"Label '{target}' not found in project '{project}'. Known labels: {known}"
            )
    return calls


def replay_examples(calls: list[Call], examples: list[Example]) -> list[CapturedRequest]:
    """Select the latest request with an exact recorded golden association."""
    captures = []
    for example in examples:
        if example.level != "call":
            continue
        matches = [
            call for call in calls
            if (call.target or call.stage or call.derived_label) == example.stage
            and call.metadata.get("example_id") == example.id
        ]
        if not matches:
            raise CaptureError(
                f"No recorded request correlated with golden example '{example.stage}/{example.id}'. "
                "Record metadata={'example_id': '<golden id>'} on the matching Call, "
                "or use --stage NAME -m your_module for fresh capture."
            )
        captures.append(captured_request_from_call(matches[0], example_id=example.id))
    return captures


def optimization_request(project: str, target: str | None) -> CapturedRequest:
    calls = recorded_calls(project, target)
    if target is None:
        costs: dict[str, float] = {}
        for call in calls:
            name = call.target or call.stage or call.derived_label
            if name and not call.metadata.get("unpriced_model") and not pricing.is_unpriced(call.model):
                costs[name] = costs.get(name, 0.0) + call.cost_usd
        if not costs:
            raise CaptureError("No recorded labels with known pricing; pass --label explicitly.")
        target = max(costs, key=costs.get)
        calls = [call for call in calls if (call.target or call.stage or call.derived_label) == target]
    if not calls:
        raise CaptureError(f"No recorded requests in project '{project}'; run your agent first.")
    return captured_request_from_call(calls[0])
