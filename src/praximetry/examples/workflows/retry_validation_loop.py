"""Try:
    python -m praximetry.examples.workflows.retry_validation_loop
"""
import json

import praximetry as px
from praximetry import runtime

from ._real import default_model, real_chat

px.init(project="retry-validation-loop")

MAX_ATTEMPTS = 3


@px.stage("generate")
def generate(request: str, feedback: str = "", attempt: int = 1) -> str:
    if attempt == 1:
        messages = [{"role": "user", "content": f"Give a one-line status update for: {request}"}]
    else:
        messages = [
            {"role": "system", "content": 'Reply with strict JSON: {"status": ..., "summary": ...}. '
                                           "No other text."},
            {"role": "user", "content": f"Request: {request}\nReviewer feedback: {feedback}"},
        ]
    return real_chat(default_model(), messages)


@px.stage("validate")
def validate(draft: str) -> str:
    try:
        obj = json.loads(draft)
    except json.JSONDecodeError:
        result = "invalid: not valid JSON"
    else:
        if "status" not in obj or "summary" not in obj:
            result = "invalid: missing required keys"
        else:
            result = "valid"

    runtime.record_call(response_text=result, cost_usd=0)
    return result


def handle(request: str) -> str:
    feedback, draft = "", ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        draft = generate(request, feedback, attempt)
        verdict = validate(draft)
        if verdict == "valid":
            break
        feedback = verdict
    return draft


REQUESTS = [
    "Summarize this week's deploys",
    "Summarize open incidents",
    "Summarize the on-call handoff notes",
    "Summarize this sprint's velocity",
]

if __name__ == "__main__":
    for req in REQUESTS:
        print(f"  {req}\n  -> {handle(req)}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
