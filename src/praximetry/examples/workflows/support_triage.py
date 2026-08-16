"""Workflow 1: RAG-style support-ticket triage (classify -> retrieve -> respond).

Mirrors a production pattern: a premium model everywhere, a bloated system
prompt with duplicated policy text, a large few-shot block, and pretty-printed
JSON knowledge passed on every call. praximetry should find all of it.

Try:
    python -m praximetry.examples.workflows.support_triage
    praximetry-cloud detect
    praximetry eval --stage classify -m praximetry.examples.workflows.support_triage --fail-under 0.9
    praximetry optimize --stage classify -m praximetry.examples.workflows.support_triage
    praximetry apply --stage classify
"""

import json

import praximetry as px

from ._real import default_model, premium_model, real_chat

px.init(project="support-triage")

# Deliberately duplicated policy lines (common when prompts grow by accretion).
POLICY = (
    "Always be professional. Never promise refunds without approval. "
    "Always be professional. Escalate legal threats to a human. "
    "Never promise refunds without approval. Answer in under 100 words. "
)
SYSTEM = "You are TriageBot for AcmeCloud support. " + POLICY * 3

FEW_SHOT = "".join(
    f"Example {i}: {t} -> {c}\n"
    for i, (t, c) in enumerate(
        [
            ("I was double charged", "billing"),
            ("Screen went black", "hardware"),
            ("Reset my password", "account"),
            ("Invoice missing VAT", "billing"),
            ("Fan is very loud", "hardware"),
            ("Change my email", "account"),
            ("Charged after cancelling", "billing"),
            ("Trackpad not clicking", "hardware"),
        ],
        start=1,
    )
)

KB = {
    "categories": {
        "billing": {
            "keywords": ["charge", "refund", "invoice", "payment", "subscription", "vat"],
            "sla_hours": 24,
            "team": "finance-support",
        },
        "hardware": {
            "keywords": ["screen", "battery", "keyboard", "fan", "trackpad", "camera"],
            "sla_hours": 48,
            "team": "device-support",
        },
        "account": {
            "keywords": ["password", "login", "email", "2fa", "username"],
            "sla_hours": 12,
            "team": "identity-support",
        },
    }
}


@px.stage("classify")
def classify(ticket: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Knowledge base:\n{json.dumps(KB, indent=4)}\n\n{FEW_SHOT}\nTicket: {ticket}\n\n"
            "Reply with exactly one word: the matching category "
            f"({', '.join(KB['categories'])}), or \"general\" if none fit.",
        },
    ]
    return real_chat(premium_model(), messages).strip().lower()


@px.stage("retrieve")
def retrieve(category: str) -> dict:
    """Non-LLM stage: fetch routing info (still attributed in traces via stage)."""
    return KB["categories"].get(category, {"team": "general-support", "sla_hours": 72})


@px.stage("respond")
def respond(ticket: str, category: str) -> str:
    info = retrieve(category)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Team: {info.get('team')}. SLA: {info.get('sla_hours')}h. "
            f"Write a short acknowledgement for this {category} ticket: {ticket}\n\n"
            f"Your reply must literally include the words \"{category}\" and "
            f"\"{info.get('team')}\".",
        },
    ]
    return real_chat(default_model(), messages)


def handle(ticket: str) -> str:
    return respond(ticket, classify(ticket))


TICKETS = [
    "I was charged twice for my subscription, need a refund",
    "My laptop screen flickers constantly",
    "Can't login after changing my email",
    "Battery drains in an hour",
    "Where is my invoice for March?",
    "Keyboard keys are sticking",
    "Password reset link never arrives",
    "The camera shows a green tint",
    "You billed me after I cancelled",
    "2FA codes are rejected",
]

if __name__ == "__main__":
    for t in TICKETS:
        print(f"  {handle(t)[:80]}")
    print("\nTraffic recorded. Try: praximetry-cloud detect")
