"""Workflow: human-in-the-loop approval.

`draft_action` proposes an action, `await_approval` is a non-LLM stage
that resolves a pending decision (the same two-phase pattern the
dashboard's own Expert Input review queue already uses -- a decision made
outside the model, still attributed as a stage), and the outcome branches
to `execute` or `discard`. The "human" decision is a deterministic policy
scanning the proposed action for destructive keywords rather than a real
pause, so runs stay reproducible -- swap `await_approval` for an actual
pending-decision store to make it a genuine pause/resume in production.

Try:
    python -m praximetry.examples.workflows.human_approval
"""
import praximetry as px
from praximetry import runtime

from ._real import default_model, real_chat

px.init(project="human-approval")

SYSTEM = "You are an ops agent. Propose exactly one remediation action, one line."

DESTRUCTIVE_KEYWORDS = ("delete", "purge", "drop", "remove")


@px.stage("draft_action")
def draft_action(request: str) -> str:
    return real_chat(
        default_model(),
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}],
    )


@px.stage("await_approval")
def await_approval(action: str) -> str:
    """Non-LLM stage: deterministic stand-in for a human reviewer. Destructive
    actions (delete/purge/drop/remove) are rejected; everything else is approved."""
    low = action.lower()
    result = "rejected" if any(k in low for k in DESTRUCTIVE_KEYWORDS) else "approved"
    # Record as a zero-cost non-LLM call to maintain call graph structure
    # without masquerading as a paid model invocation.
    runtime.record_call(response_text=result, cost_usd=0)
    return result


@px.stage("execute")
def execute(action: str) -> str:
    return real_chat(default_model(), [{"role": "user", "content": f"Confirm execution of: {action}"}])


@px.stage("discard")
def discard(action: str) -> str:
    return real_chat(default_model(), [{"role": "user", "content": f"Log the rejection of: {action}"}])


def handle(request: str) -> str:
    action = draft_action(request)
    decision = await_approval(action)
    return execute(action) if decision == "approved" else discard(action)


REQUESTS = [
    "Restart the checkout service, it's returning 500s",
    "Purge the stale user sessions table",
    "Restart the failing background worker",
    "Delete all logs older than the retention window",
]

if __name__ == "__main__":
    for req in REQUESTS:
        print(f"  {req}\n  -> {handle(req)}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
