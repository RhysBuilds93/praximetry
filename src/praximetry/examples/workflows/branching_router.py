"""Workflow: branching router -- classify_request fans to exactly one of
route_billing/route_technical/route_general per run.

Try:
    python -m praximetry.examples.workflows.branching_router
"""
import praximetry as px

from ._real import default_model, premium_model, real_chat

px.init(project="branching-router")

SYSTEM = "You are a support routing agent. Reply with exactly one word: billing, technical, or general."

DEPARTMENTS = ("billing", "technical", "general")


def _parse_department(raw: str) -> str:
    low = raw.lower()
    return next((d for d in DEPARTMENTS if d in low), "general")


@px.stage("classify_request")
def classify_request(request: str) -> str:
    raw = real_chat(
        premium_model(),
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Departments: {', '.join(DEPARTMENTS)}\nRequest: {request}"}],
    )
    return _parse_department(raw)


def _routed(stage_name: str):
    @px.stage(stage_name)
    def route(request: str) -> str:
        return real_chat(
            default_model(),
            [{"role": "user", "content": f"Draft a short reply to this {stage_name.removeprefix('route_')} "
                                          f"request: {request}"}],
        )

    return route


route_billing = _routed("route_billing")
route_technical = _routed("route_technical")
route_general = _routed("route_general")

ROUTES = {"billing": route_billing, "technical": route_technical, "general": route_general}


def handle(request: str) -> str:
    department = classify_request(request)
    return ROUTES[department](request)


REQUESTS = [
    "I was charged twice for my subscription this month",
    "The app crashes every time I open the settings page",
    "Do you have an office in Berlin?",
    "My refund from last week never showed up",
    "Getting a timeout error when I try to export a report",
    "What are your support hours?",
]

if __name__ == "__main__":
    for req in REQUESTS:
        print(f"  {req}\n  -> {handle(req)}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
