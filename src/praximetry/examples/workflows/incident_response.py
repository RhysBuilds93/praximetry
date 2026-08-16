"""Workflow 5: incident response -- the graph-shaped workflow.

The other workflows here are pipelines: a fixed sequence of stages. This one
exists because the Observe network graph has to render three structures that a
pipeline never produces, and they need real traffic to verify against:

  branching   `correlate` hands off to one of three playbook stages depending
              on what `triage` decided, so the same node has genuinely
              different successors across runs.

  loop/retry  draft_postmortem -> critique_postmortem -> (rejected) ->
              draft_postmortem, bounded by MAX_REVISIONS. A real cycle in the
              stage graph, which the layout has to render as a back-arc rather
              than pretend is a forward edge.

  fan-out     `gather_signals` is an async stage awaiting asyncio.gather over
              2-3 async sub-stages. asyncio copies the context into each task,
              so each sub-call inherits the gather_signals call as its parent:
              several calls sharing one parent_call_id is what concurrency
              looks like in the recorded data.

This mirrors the simpler `examples/incident_agent.py` fixture at the repo
root, but lives here (inside the installed package, under
`praximetry.examples.workflows`) because it's also used as a fixture by
praximetry-cloud's own eval/optimize test suite, which imports it from the
installed package rather than vendoring a copy.

Try:
    python -m praximetry.examples.workflows.incident_response
    praximetry eval --stage triage -m praximetry.examples.workflows.incident_response
"""

import asyncio

import praximetry as px

from ._real import default_model, real_chat

px.init(project="incident-response")

SYSTEM = "You are an on-call incident response agent. Be precise and cite signals. " * 6

CATEGORIES = ["database", "network", "security"]

MAX_REVISIONS = 3


def _parse_category(raw: str) -> str:
    low = raw.lower()
    return next((c for c in CATEGORIES if c in low), "database")


@px.stage("triage")
def triage(incident: str) -> str:
    raw = real_chat(
        default_model(),
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Categories: {CATEGORIES}\n\nIncident: {incident}\n\n"
                "Reply with exactly one category word, nothing else.",
            },
        ],
    )
    return _parse_category(raw)


@px.stage("fetch_logs")
async def fetch_logs(incident: str) -> str:
    await asyncio.sleep(0)  # yield, so the siblings genuinely interleave
    return real_chat(
        default_model(),
        [
            {
                "role": "user",
                "content": f"You are a log-search tool. Report one concise, plausible line of "
                f"error-log evidence for this incident, starting with 'logs:'. "
                f"Incident: {incident}",
            }
        ],
    )


@px.stage("fetch_metrics")
async def fetch_metrics(incident: str) -> str:
    await asyncio.sleep(0)
    return real_chat(
        default_model(),
        [
            {
                "role": "user",
                "content": f"You are a metrics tool. Report one concise, plausible line of "
                f"metrics evidence for this incident, starting with 'metrics:'. "
                f"Incident: {incident}",
            }
        ],
    )


@px.stage("fetch_alerts")
async def fetch_alerts(incident: str) -> str:
    await asyncio.sleep(0)
    return real_chat(
        default_model(),
        [
            {
                "role": "user",
                "content": f"You are a paging tool. Report one concise, plausible line of "
                f"alert evidence for this incident, starting with 'alerts:'. "
                f"Incident: {incident}",
            }
        ],
    )


FETCHERS = {"logs": fetch_logs, "metrics": fetch_metrics, "alerts": fetch_alerts}


@px.stage("gather_signals")
async def gather_signals(incident: str, category: str = "") -> list[str]:
    """Fan out over the signal sources this category needs.

    The planning call happens *before* the gather, so it is the call every
    spawned sub-stage inherits as its parent -- that is the sibling group.
    """
    category = category or _parse_category(triage(incident))
    raw = real_chat(
        default_model(),
        [
            {
                "role": "user",
                "content": f"Which signal sources are needed for a {category} incident? "
                f"Options: {list(FETCHERS)}. Always include 'logs' and 'metrics' as the "
                "baseline signals for any incident. For a security incident, also include "
                "'alerts'. Reply with a comma-separated subset, nothing else. "
                f"Incident: {incident}",
            }
        ],
    )
    sources = [s.strip().lower() for s in raw.split(",") if s.strip().lower() in FETCHERS]
    if not sources:
        sources = ["logs", "metrics"]
    return list(await asyncio.gather(*(FETCHERS[s](incident) for s in sources)))


@px.stage("correlate")
def correlate(incident: str, signals: list[str] | None = None) -> str:
    """Join stage: consumes the combined result of the concurrent branches."""
    return real_chat(
        default_model(),
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Incident: {incident}\nSignals:\n"
                + "\n".join(signals or [])
                + "\n\nCorrelate the signals into a finding. If they clearly point to "
                "one cause, reply starting with 'clear cause: <cause>'. If they "
                "don't converge, reply exactly 'cause: inconclusive'.",
            },
        ],
    )


def _playbook(stage_name: str, guidance: str):
    @px.stage(stage_name)
    def run_playbook(incident: str, correlation: str = "") -> str:
        return real_chat(
            default_model(),
            [
                {
                    "role": "user",
                    "content": f"Incident: {incident}\nFinding: {correlation}\n\n"
                    f"You are running the {stage_name} runbook ({guidance}). "
                    "State the remediation steps taken, concisely, past tense.",
                }
            ],
        )

    return run_playbook


db_playbook = _playbook("db_playbook", "failing over the primary, draining the pool")
network_playbook = _playbook("network_playbook", "rerouting traffic, flushing the DNS cache")
security_playbook = _playbook("security_playbook", "rotating credentials, revoking sessions")

PLAYBOOKS = {"database": db_playbook, "network": network_playbook, "security": security_playbook}


@px.stage("draft_postmortem")
def draft_postmortem(
    incident: str, correlation: str = "", remediation: str = "", feedback: str = ""
) -> str:
    # The draft only commits to a root cause once the correlation is clear or the
    # critique has pushed back, so an inconclusive first pass costs a revision.
    has_cause = "clear cause" in correlation or bool(feedback)
    cause_instruction = (
        "The finding supports a root cause: include a line starting exactly "
        "with 'root cause:' stating it."
        if has_cause
        else "The finding is inconclusive: do not state a root cause yet."
    )
    return real_chat(
        default_model(),
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Write the postmortem.\nFinding: {correlation}\n"
                f"Remediation: {remediation}\nReviewer says: {feedback or 'n/a'}\n\n"
                f"{cause_instruction}",
            },
        ],
    )


@px.stage("critique_postmortem")
def critique_postmortem(draft: str) -> str:
    return real_chat(
        default_model(),
        [
            {
                "role": "user",
                "content": f"Review this postmortem:\n{draft}\n\n"
                "A root cause is clear enough to approve once it names a specific "
                "technical or process failure (not a vague placeholder like "
                "'identified and confirmed'). It does not need an exhaustive "
                "5-whys chain all the way to a first cause -- one concrete, "
                "specific cause is sufficient. If it meets that bar, reply "
                "exactly 'approve'. Otherwise reply 'reject: <short reason>'.",
            }
        ],
    )


@px.stage("publish_postmortem")
def publish_postmortem(draft: str) -> str:
    # This stage only logs that an already-approved postmortem went out --
    # remediation status is the critique stage's job to gate on, not this
    # one's. State that plainly so the model doesn't second-guess or
    # editorialize about pending remediation instead of confirming.
    return real_chat(
        default_model(),
        [
            {
                "role": "user",
                "content": f"This postmortem has already been approved for publishing. Confirm "
                "publication with a single line starting exactly with 'Published:' "
                f"followed by the incident name. No caveats, no questions.\n\n{draft}",
            }
        ],
    )


async def handle(incident: str) -> str:
    category = triage(incident)
    signals = await gather_signals(incident, category)
    correlation = correlate(incident, signals)
    remediation = PLAYBOOKS[category](incident, correlation)

    feedback, draft = "", ""
    for _ in range(MAX_REVISIONS):
        draft = draft_postmortem(incident, correlation, remediation, feedback)
        verdict = critique_postmortem(draft)
        # Some providers (e.g. gpt-oss models via Bedrock's OpenAI-compatible
        # endpoint) prefix every reply with a visible <reasoning>...</reasoning>
        # block, so the verdict never actually starts with "approve" even when
        # it ends with it -- a substring check is robust to that, and to a
        # plain "approve" reply from providers that don't do this.
        if "approve" in verdict.lower():
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
    for inc in INCIDENTS:
        asyncio.run(handle(inc))
        print(f"  {inc}")
    print("\nTraffic recorded. Try: praximetry-cloud detect")
