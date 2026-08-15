"""Workflow: multi-turn iterative tool use.

`decide_action` (the model picks a tool or "done") and `call_tool`
(non-LLM execution) alternate, with the tool's observation fed back into
the next `decide_action` call -- a bounded while-loop, distinct from
`retry_validation_loop`'s shape because the loop here is driven by the
model choosing to continue, not by a validator rejecting output.

Try:
    python -m praximetry.examples.workflows.tool_use_loop
"""
import praximetry as px
from praximetry import runtime

from ._real import default_model, real_chat

px.init(project="tool-use-loop")

SYSTEM = "You are a travel assistant. Reply with exactly one of: search_flights, search_hotels, done."

MAX_TURNS = 4

VALID_ACTIONS = ("search_flights", "search_hotels", "done")

TOOLS = {
    "search_flights": lambda request: "3 flights found, cheapest $210",
    "search_hotels": lambda request: "5 hotels found, cheapest $89/night",
}


def _parse_action(raw: str) -> str:
    low = raw.lower()
    return next((a for a in VALID_ACTIONS if a in low), "done")


@px.stage("decide_action")
def decide_action(request: str, observations: list[str] | None = None) -> str:
    observations = observations or []
    raw = real_chat(
        default_model(),
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Request: {request}\nSo far: {observations}"}],
    )
    return _parse_action(raw)


@px.stage("call_tool")
def call_tool(tool: str, request: str) -> str:
    result = TOOLS[tool](request)
    # Record as a zero-cost non-LLM call to maintain call graph structure
    # without masquerading as a paid model invocation.
    runtime.record_call(response_text=result, cost_usd=0)
    return result


def handle(request: str) -> str:
    observations: list[str] = []
    for _ in range(MAX_TURNS):
        action = decide_action(request, observations)
        if action == "done":
            break
        observations.append(call_tool(action, request))
    return "; ".join(observations) or "no action needed"


REQUESTS = [
    "Book me a flight to Lisbon",
    "I need a hotel in Lisbon for 3 nights",
    "Find me a flight and hotel for a trip to Lisbon",
    "Plan a weekend stay in Lisbon",
]

if __name__ == "__main__":
    for req in REQUESTS:
        print(f"  {req}\n  -> {handle(req)}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
