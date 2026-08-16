"""Workflow 4: tool-calling retail support agent, modelled on Sierra's tau-bench.

tau-bench (https://github.com/sierra-research/tau-bench) evaluates a customer
service agent that talks to a simulated user, calls domain API tools, and must
obey a written business policy. Crucially it does not score the chat text: it
compares the *final database state* against the expected state, and checks the
agent didn't violate policy on the way (e.g. cancelling a delivered order
instead of returning it).

We mirror that shape exactly:
  plan_action  (LLM)  -> chooses a tool + args from the request and the policy
  execute      (tool) -> mutates the order database
  write_reply  (LLM)  -> customer-facing message
and the goldens score the resulting DB state with `json_match`, so the optimizer
is optimizing against *task success*, not against string similarity.

Try:
    python -m praximetry.examples.workflows.tau_retail_agent
    praximetry-cloud detect
    praximetry eval --stage plan_action -m praximetry.examples.workflows.tau_retail_agent --fail-under 0.9
    praximetry optimize --stage plan_action -m praximetry.examples.workflows.tau_retail_agent
    praximetry apply --stage plan_action
"""

from __future__ import annotations

import copy
import json
import re

import praximetry as px

from ._real import default_model, premium_model, real_chat

px.init(project="tau-retail")

# The business policy the agent must comply with. Written the way real ones are:
# appended to over time, so the same rules appear more than once.
POLICY = (
    "You are a retail support agent for Nimbus Goods.\n"
    "Pending orders may be cancelled. Pending orders may be cancelled.\n"
    "Delivered orders may NOT be cancelled - they must be returned instead.\n"
    "Exchanges are only allowed on delivered orders.\n"
    "Delivered orders may NOT be cancelled - they must be returned instead.\n"
    "Never issue a refund without a return. Always confirm the order id.\n"
)
SYSTEM = POLICY * 2

TOOL_SPEC = {
    "tools": [
        {
            "name": "get_order",
            "args": {"order_id": "str"},
            "description": "Fetch an order's status, items and totals",
        },
        {
            "name": "cancel_order",
            "args": {"order_id": "str"},
            "description": "Cancel a PENDING order only",
        },
        {
            "name": "return_order",
            "args": {"order_id": "str"},
            "description": "Start a return for a DELIVERED order (required before any refund)",
        },
        {
            "name": "exchange_item",
            "args": {"order_id": "str", "size": "str"},
            "description": "Exchange an item on a DELIVERED order for a different size",
        },
    ]
}

FEW_SHOT = "".join(
    f"Example {i}: {req} -> {tool}\n"
    for i, (req, tool) in enumerate(
        [
            ("cancel my order #W1001 (pending)", "cancel_order"),
            ("where is order #W1002 (delivered)", "get_order"),
            ("I want to send back #W1003 (delivered)", "return_order"),
            ("exchange #W1003 for size M (delivered)", "exchange_item"),
            ("stop order #W1005 shipping (shipped)", "cancel_order"),
            ("track #W1006 please (pending)", "get_order"),
            # The customer's wording says "cancel", but the order is already
            # delivered -- policy means that request maps to return_order, not
            # cancel_order. This is the case the agent gets wrong most often.
            ("cancel #W1007, it doesn't fit (delivered)", "return_order"),
        ],
        start=1,
    )
)

# The "database". Reset per run so evals are deterministic.
SEED_DB: dict[str, dict] = {
    "#W1001": {"status": "pending", "item": "trail runners", "size": "9", "total": 84.0},
    "#W1002": {"status": "delivered", "item": "rain shell", "size": "M", "total": 129.0},
    "#W1003": {"status": "delivered", "item": "wool socks", "size": "L", "total": 22.0},
    "#W1004": {"status": "pending", "item": "daypack", "size": "one", "total": 65.0},
    "#W1005": {"status": "shipped", "item": "headtorch", "size": "one", "total": 31.0},
}
DB: dict[str, dict] = copy.deepcopy(SEED_DB)


def reset_db() -> None:
    global DB
    DB = copy.deepcopy(SEED_DB)


def _order_id(text: str) -> str:
    m = re.search(r"#w\d+", text, re.I)
    return m.group(0).upper() if m else ""


@px.stage("plan_action")
def plan_action(request: str) -> dict:
    """Pick the tool call. This is the stage that has to respect the policy."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Tools:\n{json.dumps(TOOL_SPEC, indent=4)}\n\n"
            f"Order book:\n{json.dumps(DB, indent=4)}\n\n"
            f"{FEW_SHOT}\nRequest: {request}\n\n"
            "First look up the order's current status in the order book above, "
            "then pick the tool the policy requires for that status -- not the "
            "verb the customer happened to use. A customer asking to 'cancel' an "
            "order that is already delivered still means return_order, per policy. "
            "Reply with ONLY a JSON object of the form "
            '{"tool": <tool name>, "order_id": <order id>, "size": <optional>}, '
            "nothing else.",
        },
    ]
    raw = real_chat(premium_model(), messages)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"tool": "get_order", "order_id": _order_id(request)}


@px.stage("execute")
def execute(action: dict) -> dict:
    """Non-LLM tool execution. Enforces the policy server-side too."""
    oid, tool = action.get("order_id", ""), action.get("tool")
    order = DB.get(oid)
    if not order:
        return {"error": "unknown_order", "order_id": oid}
    if tool == "cancel_order":
        if order["status"] != "pending":
            return {"error": "policy_violation", "order_id": oid, "status": order["status"]}
        order["status"] = "cancelled"
    elif tool == "return_order":
        if order["status"] != "delivered":
            return {"error": "policy_violation", "order_id": oid, "status": order["status"]}
        order["status"] = "return_pending"
    elif tool == "exchange_item":
        if order["status"] != "delivered":
            return {"error": "policy_violation", "order_id": oid, "status": order["status"]}
        order["status"] = "exchange_pending"
        order["size"] = action.get("size", order["size"])
    return {"order_id": oid, **order}


@px.stage("write_reply")
def write_reply(request: str, result: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Result: {json.dumps(result)}\nWrite one sentence back to the customer "
            f"about their request: {request}",
        },
    ]
    return real_chat(default_model(), messages)


@px.stage("handle_request")
def handle_request(request: str) -> dict:
    """Full task. Returns the final order state - this is what tau-bench scores."""
    action = plan_action(request)
    result = execute(action)
    write_reply(request, result)
    return result


REQUESTS = [
    "Please cancel order #W1001, I ordered by mistake",
    "I want to cancel #W1002, the rain shell doesn't fit",  # delivered -> must return
    "Where is my order #W1005?",
    "Can I swap #W1002 for a large?",
    "I'd like to send back #W1003 for a refund",
    "Stop order #W1004 from shipping please",
    "Track #W1003 for me",
    "Cancel #W1005, I changed my mind",  # shipped -> not cancellable
]

if __name__ == "__main__":
    reset_db()
    for r in REQUESTS:
        print(f"  {r[:40]:42s} -> {handle_request(r)}")
    print("\nTraffic recorded. Try: praximetry-cloud detect")
