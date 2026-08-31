"""Try:
python -m praximetry.examples.workflows.incident_response

Graph-shaped workflow: `correlate` branches to one of three playbooks,
`draft_postmortem`/`critique_postmortem` form a bounded retry cycle, and
`gather_signals` fans out over concurrently-dispatched tools. Built on
LangGraph, same as research_supervisor -- context is threaded explicitly via
runtime.capture_context()/restore_context() carried in graph state, since
contextvars don't survive LangGraph's per-node executor or a new thread.
"""

from __future__ import annotations

import concurrent.futures

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

import praximetry as px
from praximetry import runtime

from ._real import chat_model, clean_content, default_model

px.init(project="incident-response")

CATEGORIES = ["database", "network", "security"]
MAX_REVISIONS = 3
MAX_PLAYBOOK_TURNS = 3


def _clean_tool_calls(message: AIMessage, valid_names) -> AIMessage:
    """Resolve tool-call names against the known-valid set.

    gpt-oss has leaked Harmony channel/format tags glued onto tool-call names
    in several different shapes -- rather than chase each new variant, treat
    any call name that starts with exactly one valid name as that name with a
    leaked suffix.
    """
    if not message.tool_calls:
        return message
    for call in message.tool_calls:
        name = call["name"]
        if name in valid_names:
            continue
        matches = [v for v in valid_names if name.startswith(v)]
        if len(matches) == 1:
            call["name"] = matches[0]
    return message


def _ask(prompt: str, *, system: bool = False) -> str:
    messages = (
        [SystemMessage("You are an on-call incident response agent. Be precise and cite signals.")]
        if system
        else []
    )
    messages.append(HumanMessage(prompt))
    reply = chat_model(default_model()).invoke(messages)
    return clean_content(reply, default_model())


def _record(result: str) -> str:
    runtime.record_call(output_text=result, cost_usd=0)
    return result


def _parse_category(raw: str) -> str:
    low = raw.lower()
    return next((c for c in CATEGORIES if c in low), "database")


# -- triage ----------------------------------------------------------------


@px.stage("triage")
def _triage(incident: str) -> str:
    raw = _ask(
        f"Categories: {CATEGORIES}\n\nIncident: {incident}\n\n"
        "Reply with exactly one category word, nothing else.",
        system=True,
    )
    return _parse_category(raw)


# -- signal gathering: a real tool-calling agent, concurrent dispatch -------


@tool
@px.stage("fetch_logs")
def fetch_logs(incident: str) -> str:
    """Search application logs for evidence relevant to this incident."""
    return _record(f"logs: connection errors spiking for '{incident}'")


@tool
@px.stage("fetch_metrics")
def fetch_metrics(incident: str) -> str:
    """Query infrastructure metrics for evidence relevant to this incident."""
    return _record(f"metrics: error rate and latency both elevated during '{incident}'")


@tool
@px.stage("fetch_alerts")
def fetch_alerts(incident: str) -> str:
    """Check the paging/alerting system for evidence relevant to this incident."""
    return _record(f"alerts: paging fired for '{incident}'")


GATHER_TOOLS = [fetch_logs, fetch_metrics, fetch_alerts]
GATHER_TOOL_BY_NAME = {t.name: t for t in GATHER_TOOLS}

OPTIONAL_GATHER_TOOLS = [fetch_alerts]
OPTIONAL_GATHER_TOOL_BY_NAME = {t.name: t for t in OPTIONAL_GATHER_TOOLS}

GATHER_SYSTEM = (
    "Logs and metrics have already been pulled as baseline signals for this "
    "incident. Decide whether any further signals are worth gathering -- call "
    "fetch_alerts if paging/alerting data would help (e.g. a security "
    "incident), otherwise call nothing."
)


@px.stage("gather_signals")
def _gather_signals(incident: str, category: str) -> list[str]:
    # Logs and metrics are the mandatory baseline for any incident, not a
    # judgment call, so they're fetched directly rather than left to the
    # model to remember -- the agent only decides on optional extras below.
    calls = [
        {"name": n, "args": {"incident": incident}, "id": n}
        for n in ("fetch_logs", "fetch_metrics")
    ]

    llm = chat_model(default_model()).bind_tools(OPTIONAL_GATHER_TOOLS)
    reply = _clean_tool_calls(
        llm.invoke(
            [
                SystemMessage(GATHER_SYSTEM),
                HumanMessage(f"Category: {category}\nIncident: {incident}"),
            ]
        ),
        OPTIONAL_GATHER_TOOL_BY_NAME.keys(),
    )
    calls += [c for c in reply.tool_calls if c["name"] in OPTIONAL_GATHER_TOOL_BY_NAME]

    ctx = runtime.capture_context()

    def run_one(call: dict) -> str:
        with runtime.restore_context(ctx):
            return GATHER_TOOL_BY_NAME[call["name"]].invoke(call["args"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(calls), 1)) as pool:
        return list(pool.map(run_one, calls))


# -- correlate: join stage ---------------------------------------------------


@px.stage("correlate")
def _correlate(incident: str, signals: list[str]) -> str:
    return _ask(
        f"Incident: {incident}\nSignals:\n"
        + "\n".join(signals)
        + "\n\nCorrelate the signals into a finding. If they clearly point to "
        "one cause, reply starting with 'clear cause: <cause>'. If they "
        "don't converge, reply exactly 'cause: inconclusive'.",
        system=True,
    )


# -- playbooks: one tool-calling agent per category, real remediation tools -


@tool
@px.stage("fail_over_primary")
def fail_over_primary(incident: str) -> str:
    """Fail the primary database over to its replica."""
    return _record(f"failed over primary -> replica for '{incident}'")


@tool
@px.stage("drain_connection_pool")
def drain_connection_pool(incident: str) -> str:
    """Drain and reset the exhausted connection pool."""
    return _record(f"drained and reset the connection pool for '{incident}'")


@tool
@px.stage("reroute_traffic")
def reroute_traffic(incident: str) -> str:
    """Reroute traffic away from the affected network path."""
    return _record(f"rerouted traffic away from the affected path for '{incident}'")


@tool
@px.stage("flush_dns_cache")
def flush_dns_cache(incident: str) -> str:
    """Flush the DNS cache on affected resolvers."""
    return _record(f"flushed DNS cache on affected resolvers for '{incident}'")


@tool
@px.stage("rotate_credentials")
def rotate_credentials(incident: str) -> str:
    """Rotate credentials suspected of compromise."""
    return _record(f"rotated suspected-compromised credentials for '{incident}'")


@tool
@px.stage("revoke_sessions")
def revoke_sessions(incident: str) -> str:
    """Revoke active sessions tied to the suspected compromise."""
    return _record(f"revoked active sessions tied to '{incident}'")


PLAYBOOK_TOOLS = {
    "database": [fail_over_primary, drain_connection_pool],
    "network": [reroute_traffic, flush_dns_cache],
    "security": [rotate_credentials, revoke_sessions],
}
PLAYBOOK_STAGE = {
    "database": "db_playbook",
    "network": "network_playbook",
    "security": "security_playbook",
}


def _playbook_agent(stage_name: str, tools: list):
    tool_by_name = {t.name: t for t in tools}

    @px.stage(stage_name)
    def run(incident: str, correlation: str) -> str:
        llm = chat_model(default_model()).bind_tools(tools)
        messages: list = [
            SystemMessage(
                f"You are running the {stage_name} runbook. Call whichever remediation "
                "tool(s) are needed, then reply with a single sentence confirming what "
                "was done, past tense."
            ),
            HumanMessage(f"Incident: {incident}\nFinding: {correlation}"),
        ]
        for _ in range(MAX_PLAYBOOK_TURNS):
            reply = _clean_tool_calls(llm.invoke(messages), tool_by_name.keys())
            messages.append(reply)
            if not reply.tool_calls:
                return reply.content
            for call in reply.tool_calls:
                t = tool_by_name.get(call["name"])
                result = t.invoke(call["args"]) if t else f"unknown tool {call['name']!r}"
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        last = messages[-1]
        return (
            last.content
            if isinstance(last, AIMessage)
            else "remediation attempted, no confirmation reached"
        )

    return run


PLAYBOOK_AGENTS = {
    cat: _playbook_agent(PLAYBOOK_STAGE[cat], tools) for cat, tools in PLAYBOOK_TOOLS.items()
}


# -- postmortem: draft/critique retry loop, then publish ---------------------


@px.stage("draft_postmortem")
def _draft_postmortem(incident: str, correlation: str, remediation: str, feedback: str) -> str:
    has_cause = "clear cause" in correlation or bool(feedback)
    cause_instruction = (
        "The finding supports a root cause: include a line starting exactly "
        "with 'root cause:' stating it."
        if has_cause
        else "The finding is inconclusive: do not state a root cause yet."
    )
    return _ask(
        f"Write the postmortem.\nFinding: {correlation}\n"
        f"Remediation: {remediation}\nReviewer says: {feedback or 'n/a'}\n\n"
        f"{cause_instruction}",
        system=True,
    )


@px.stage("critique_postmortem")
def _critique_postmortem(draft: str) -> str:
    return _ask(
        f"Review this postmortem:\n{draft}\n\n"
        "Approve only if it clears every one of these bars: (1) a root cause "
        "naming a specific technical or process failure, not a vague "
        "placeholder like 'identified and confirmed' -- one concrete cause is "
        "enough, it doesn't need a full 5-whys chain; (2) a quantified impact "
        "-- a concrete duration, error rate, or affected-user figure, not just "
        "'user-facing degradation'; (3) confirmation the remediation was "
        "executed and verified, not just planned. If all three are met, "
        "reply exactly 'approve'. Otherwise reply 'reject: <short reason>' "
        "naming which bar(s) it missed."
    )


@px.stage("publish_postmortem")
def _publish_postmortem(draft: str) -> str:
    return _ask(
        f"This postmortem has already been approved for publishing. Confirm "
        "publication with a single line starting exactly with 'Published:' "
        f"followed by the incident name. No caveats, no questions.\n\n{draft}"
    )


# -- orchestration: LangGraph state machine ----------------------------------


class IncidentState(TypedDict):
    incident: str
    category: str
    signals: list[str]
    correlation: str
    remediation: str
    draft: str
    feedback: str
    revisions: int
    result: str | None
    _px_ctx: dict | None


def triage_node(state: IncidentState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        category = _triage(state["incident"])
        ctx = runtime.capture_context()
    return {"category": category, "_px_ctx": ctx}


def gather_node(state: IncidentState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        signals = _gather_signals(state["incident"], state["category"])
        ctx = runtime.capture_context()
    return {"signals": signals, "_px_ctx": ctx}


def correlate_node(state: IncidentState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        correlation = _correlate(state["incident"], state["signals"])
        ctx = runtime.capture_context()
    return {"correlation": correlation, "_px_ctx": ctx}


def playbook_node(state: IncidentState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        remediation = PLAYBOOK_AGENTS[state["category"]](state["incident"], state["correlation"])
        ctx = runtime.capture_context()
    return {"remediation": remediation, "_px_ctx": ctx}


def draft_node(state: IncidentState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        draft = _draft_postmortem(
            state["incident"], state["correlation"], state["remediation"], state.get("feedback", "")
        )
        ctx = runtime.capture_context()
    return {"draft": draft, "revisions": state.get("revisions", 0) + 1, "_px_ctx": ctx}


def critique_node(state: IncidentState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        feedback = _critique_postmortem(state["draft"])
        ctx = runtime.capture_context()
    return {"feedback": feedback, "_px_ctx": ctx}


def route_after_critique(state: IncidentState) -> str:
    if "approve" in state["feedback"].lower() or state["revisions"] >= MAX_REVISIONS:
        return "publish"
    return "draft"


def publish_node(state: IncidentState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        result = _publish_postmortem(state["draft"])
        ctx = runtime.capture_context()
    return {"result": result, "_px_ctx": ctx}


graph = StateGraph(IncidentState)
graph.add_node("triage", triage_node)
graph.add_node("gather", gather_node)
graph.add_node("correlate", correlate_node)
graph.add_node("playbook", playbook_node)
graph.add_node("draft", draft_node)
graph.add_node("critique", critique_node)
graph.add_node("publish", publish_node)
graph.set_entry_point("triage")
graph.add_edge("triage", "gather")
graph.add_edge("gather", "correlate")
graph.add_edge("correlate", "playbook")
graph.add_edge("playbook", "draft")
graph.add_edge("draft", "critique")
graph.add_conditional_edges(
    "critique", route_after_critique, {"draft": "draft", "publish": "publish"}
)
graph.add_edge("publish", END)
app = graph.compile()


def handle(incident: str) -> str:
    with runtime.run_context(name=incident[:60]):
        seed_ctx = runtime.capture_context()
        final_state = app.invoke(
            {
                "incident": incident,
                "category": "",
                "signals": [],
                "correlation": "",
                "remediation": "",
                "draft": "",
                "feedback": "",
                "revisions": 0,
                "result": None,
                "_px_ctx": seed_ctx,
            }
        )
    return final_state["result"] or final_state["draft"]


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
        result = handle(inc)
        print(f"  {inc}\n  -> {result}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
