"""Demo agent: a tiny support-ticket pipeline instrumented with praximetry.

Runs fully offline (simulated LLM) so you can explore recorded traffic without keys:

    python -m praximetry.examples.demo_agent     # generates traffic into .praximetry/
    praximetry summary
"""
import json
import random

import praximetry as px
from praximetry import pricing

px.init(project="demo-support-bot")

SYSTEM = "You are a meticulous support-ticket triage agent. " * 8  # deliberately bloated

KNOWLEDGE = {"routing_table": {"billing": ["charge", "refund", "invoice"],
                               "hardware": ["screen", "battery", "keyboard"],
                               "account": ["password", "login", "email"]}}


def _sim_llm(model: str, messages: list[dict]) -> str:
    """Offline stand-in for a chat API."""
    text = messages[-1]["content"].lower()
    # Simulate a model that reads the actual ticket, not the reference material.
    if "ticket:" in text:
        text = text.split("ticket:", 1)[1]
    answer = next((cat for cat, kws in KNOWLEDGE["routing_table"].items()
                   if any(k in text for k in kws)), "general")
    tin = sum(len(str(m.get("content", ""))) // 4 for m in messages)
    tout = len(answer) // 4 + random.randint(3, 12)
    px.record_call(provider="sim", model=model, messages=messages,
                   output_text=answer, input_tokens=tin, output_tokens=tout,
                   cost_usd=pricing.cost_usd(model, tin, tout), latency_ms=random.uniform(200, 900))
    return answer


@px.stage("classify")
def classify(ticket: str) -> str:
    # Anti-patterns on purpose: premium model, bloated system prompt, pretty JSON —
    # this is exactly the kind of traffic praximetry-cloud's detectors flag.
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Routing table:\n{json.dumps(KNOWLEDGE, indent=4)}\n\nTicket: {ticket}"},
    ]
    return _sim_llm("claude-opus-4-8", messages)


@px.stage("draft_reply")
def draft_reply(ticket: str, category: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Draft a reply for a {category} ticket: {ticket}"}]
    return _sim_llm("claude-opus-4-8", messages)


TICKETS = [
    "I was charged twice for my subscription, need a refund",
    "My laptop screen flickers constantly",
    "Can't login after changing my email",
    "Battery drains in an hour",
    "Where is my invoice for March?",
    "Keyboard keys are sticking",
    "Password reset link never arrives",
]

if __name__ == "__main__":
    for t in TICKETS:
        cat = classify(t)
        draft_reply(t, cat)
        print(f"  [{cat:<8}] {t}")
    print("\nDone. Try: praximetry summary")
