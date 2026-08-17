"""Incident-response agent: a deliberately graph-shaped pipeline.

Where `demo_agent.py` is a straight line (classify -> draft_reply), this one
exists to exercise the three structures a call graph actually has to represent,
all of which fall out of `Call.parent_call_id` automatically:

  branching   triage picks a category, and the run then takes one of three
              different playbook stages -- so `correlate` has several distinct
              successor stages across the sample incidents.

  loop/retry  draft_postmortem -> critique_postmortem -> (rejected) ->
              draft_postmortem ... -> publish_postmortem. That is a genuine
              cycle in the stage-transition graph, bounded by MAX_REVISIONS.

  fan-out     `gather_signals` is an async @stage that awaits asyncio.gather
              over 2-3 async sub-stages. asyncio copies the context into each
              spawned task, so every sub-call independently inherits the
              gather_signals call as its parent: N calls sharing one
              parent_call_id, which is exactly what "these ran concurrently"
              looks like in the data. No extra API to call -- it is free.

Runs fully offline (simulated LLM), like demo_agent.py:

    python -m praximetry.examples.incident_agent     # generates traffic into .praximetry/
    praximetry summary
"""

import asyncio
import random

import praximetry as px
from praximetry import pricing

px.init(project="incident-responder")

SYSTEM = "You are an on-call incident response agent. Be precise and cite signals. " * 6

# Which playbook a category routes to -- the branch table.
PLAYBOOKS = {
    "database": "db_playbook",
    "network": "network_playbook",
    "security": "security_playbook",
}

CATEGORY_KEYWORDS = {
    "database": ["query", "replica", "deadlock", "postgres", "connection pool"],
    "network": ["latency", "packet", "dns", "timeout", "load balancer"],
    "security": ["breach", "credential", "exfiltration", "unauthorized", "token"],
}

MAX_REVISIONS = 3


def _sim_llm(model: str, messages: list[dict], answer: str) -> str:
    """Offline stand-in for a chat API. Records the call, returns `answer`."""
    tin = sum(len(str(m.get("content", ""))) // 4 for m in messages)
    tout = max(len(answer) // 4, 1) + random.randint(2, 9)
    px.record_call(
        provider="sim",
        model=model,
        messages=messages,
        output_text=answer,
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=pricing.cost_usd(model, tin, tout),
        latency_ms=random.uniform(200, 900),
    )
    return answer


def _categorize(incident: str) -> str:
    low = incident.lower()
    return next(
        (cat for cat, kws in CATEGORY_KEYWORDS.items() if any(k in low for k in kws)), "database"
    )


# -- branch point --------------------------------------------------------------


@px.stage("triage")
def triage(incident: str) -> str:
    """Pick the category. Downstream stages genuinely differ per category."""
    category = _categorize(incident)
    return _sim_llm(
        "claude-opus-4-8",
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Categories: {list(PLAYBOOKS)}\n\nIncident: {incident}"},
        ],
        category,
    )


# -- concurrent fan-out --------------------------------------------------------


@px.stage("fetch_logs")
async def fetch_logs(incident: str) -> str:
    await asyncio.sleep(0)  # yield, so the siblings genuinely interleave
    return _sim_llm(
        "claude-sonnet-5",
        [{"role": "user", "content": f"Pull error logs for: {incident}"}],
        "logs: 412 errors clustered in a 90s window",
    )


@px.stage("fetch_metrics")
async def fetch_metrics(incident: str) -> str:
    await asyncio.sleep(0)
    return _sim_llm(
        "claude-sonnet-5",
        [{"role": "user", "content": f"Pull metrics for: {incident}"}],
        "metrics: p99 latency 4.2s, saturation 0.93",
    )


@px.stage("fetch_alerts")
async def fetch_alerts(incident: str) -> str:
    await asyncio.sleep(0)
    return _sim_llm(
        "claude-sonnet-5",
        [{"role": "user", "content": f"Pull paging alerts for: {incident}"}],
        "alerts: 2 pages, both acknowledged",
    )


@px.stage("gather_signals")
async def gather_signals(incident: str, category: str) -> list[str]:
    """Fan out over the signal sources this category needs.

    The planning call below happens *before* the gather, so it is the call that
    every spawned sub-stage inherits as its parent -- the sibling group.
    """
    sources = ["logs", "metrics"] + (["alerts"] if category == "security" else [])
    _sim_llm(
        "claude-sonnet-5",
        [{"role": "user", "content": f"Which signals for a {category} incident? {incident}"}],
        ",".join(sources),
    )
    fetchers = {"logs": fetch_logs, "metrics": fetch_metrics, "alerts": fetch_alerts}
    return list(await asyncio.gather(*(fetchers[s](incident) for s in sources)))


@px.stage("correlate")
def correlate(incident: str, signals: list[str]) -> str:
    """Join stage: consumes the combined result of the concurrent branches."""
    # A "clear" cause is one where the metrics and logs agree; security
    # incidents stay ambiguous on the first pass, which drives the retry loop.
    clear = "saturation" in " ".join(signals) and "alerts" not in " ".join(signals)
    verdict = "clear cause: connection pool saturation" if clear else "cause: inconclusive"
    return _sim_llm(
        "claude-opus-4-8",
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Incident: {incident}\nSignals:\n" + "\n".join(signals)},
        ],
        verdict,
    )


# -- the three branch destinations ---------------------------------------------


def _playbook(stage_name: str, steps: str):
    @px.stage(stage_name)
    def run_playbook(incident: str, correlation: str) -> str:
        return _sim_llm(
            "claude-sonnet-5",
            [{"role": "user", "content": f"Incident: {incident}\nFinding: {correlation}"}],
            steps,
        )

    return run_playbook


db_playbook = _playbook("db_playbook", "failed over the primary, drained the pool")
network_playbook = _playbook("network_playbook", "rerouted traffic, flushed the DNS cache")
security_playbook = _playbook("security_playbook", "rotated credentials, revoked sessions")

PLAYBOOK_FNS = {"database": db_playbook, "network": network_playbook, "security": security_playbook}


# -- retry loop ----------------------------------------------------------------


@px.stage("draft_postmortem")
def draft_postmortem(incident: str, correlation: str, remediation: str, feedback: str) -> str:
    # Only once the critique has pushed back does the draft commit to a root
    # cause -- so an inconclusive correlation always costs at least one revision.
    has_cause = "clear cause" in correlation or bool(feedback)
    body = f"Incident: {incident}. Remediation: {remediation}."
    if has_cause:
        body += " root cause: identified and confirmed."
    return _sim_llm(
        "claude-opus-4-8",
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Write the postmortem.\nFinding: {correlation}\n"
                f"Remediation: {remediation}\nReviewer says: {feedback or 'n/a'}",
            },
        ],
        body,
    )


@px.stage("critique_postmortem")
def critique_postmortem(draft: str) -> str:
    verdict = "approve" if "root cause:" in draft else "reject: no root cause stated"
    return _sim_llm(
        "claude-opus-4-8",
        [{"role": "user", "content": f"Review this postmortem:\n{draft}"}],
        verdict,
    )


@px.stage("publish_postmortem")
def publish_postmortem(draft: str) -> str:
    return _sim_llm(
        "claude-sonnet-5",
        [{"role": "user", "content": f"Publish:\n{draft}"}],
        "published",
    )


# -- orchestration -------------------------------------------------------------


async def handle(incident: str) -> str:
    category = triage(incident)
    signals = await gather_signals(incident, category)
    correlation = correlate(incident, signals)
    remediation = PLAYBOOK_FNS[category](incident, correlation)

    feedback = ""
    draft = ""
    for _ in range(MAX_REVISIONS):
        draft = draft_postmortem(incident, correlation, remediation, feedback)
        verdict = critique_postmortem(draft)
        if verdict.startswith("approve"):
            break
        feedback = verdict
    publish_postmortem(draft)
    return draft


INCIDENTS = [
    "Checkout API timing out, connection pool exhausted on the primary postgres",
    "Elevated p99 latency behind the load balancer, packet loss in eu-west",
    "Unauthorized token use detected against the admin API",
    "Replica lag climbing, deadlock storm on the orders table",
    "DNS resolution failing intermittently for the internal service mesh",
    "Credential exfiltration suspected from a stale CI runner",
]

if __name__ == "__main__":
    for incident in INCIDENTS:
        asyncio.run(handle(incident))
        print(f"  [{_categorize(incident):<8}] {incident}")
    print("\nDone. Try: praximetry summary")
