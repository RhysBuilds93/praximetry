"""Try:
python -m praximetry.examples.workflows.supervisor_delegation

A supervisor classifies which domains a request touches, dispatches to 1-3
domain agents concurrently, then synthesizes their findings. Built on
LangGraph, same as research_supervisor -- context is threaded explicitly via
runtime.capture_context()/restore_context() carried in graph state, since
contextvars don't survive a new thread. dispatch_node doesn't advance the
carried context, so the agents and synthesize all genuinely parent to
supervisor's own call, matching the intended sibling-group shape.
"""

from __future__ import annotations

import concurrent.futures
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import praximetry as px
from praximetry import runtime

from ._real import chat_model, clean_content, default_model, premium_model

px.init(project="supervisor-delegation")

SUPERVISOR_SYSTEM = (
    "You are a triage supervisor. Decide which domains this request touches. "
    'Respond with JSON only, no other text: {"domains": [...]}, where each '
    "entry is one of billing, technical, general -- at least one."
)


class DomainClassification(BaseModel):
    domains: list[Literal["billing", "technical", "general"]] = Field(
        description="Every relevant domain, at least one."
    )


@px.stage("supervisor")
def _supervisor(request: str) -> list[str]:
    model = premium_model()
    reply = chat_model(model).invoke([SystemMessage(SUPERVISOR_SYSTEM), HumanMessage(request)])
    result = DomainClassification.model_validate_json(clean_content(reply, model))
    return list(dict.fromkeys(result.domains)) or ["general"]


def _agent(stage_name: str):
    @px.stage(stage_name)
    def agent(request: str) -> str:
        reply = chat_model(default_model()).invoke(
            [HumanMessage(f"As the {stage_name}, give a one-line finding for: {request}")]
        )
        return clean_content(reply, default_model())

    return agent


AGENTS = {
    "billing": _agent("billing_agent"),
    "technical": _agent("technical_agent"),
    "general": _agent("general_agent"),
}


@px.stage("synthesize")
def _synthesize(request: str, findings: list[str]) -> str:
    reply = chat_model(premium_model()).invoke(
        [HumanMessage(f"Request: {request}\nFindings: {' '.join(findings)}\nSynthesize a reply.")]
    )
    return clean_content(reply, premium_model())


class DelegationState(TypedDict):
    request: str
    domains: list[str]
    findings: list[str]
    result: str | None
    _px_ctx: dict | None


def supervisor_node(state: DelegationState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        domains = _supervisor(state["request"])
        ctx = runtime.capture_context()
    return {"domains": domains, "_px_ctx": ctx}


def dispatch_node(state: DelegationState) -> dict:
    ctx = state.get("_px_ctx")

    def run_one(domain: str) -> str:
        with runtime.restore_context(ctx):
            return AGENTS[domain](state["request"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(state["domains"])) as pool:
        findings = list(pool.map(run_one, state["domains"]))
    return {"findings": findings}


def synthesize_node(state: DelegationState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        result = _synthesize(state["request"], state["findings"])
    return {"result": result}


graph = StateGraph(DelegationState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("dispatch", dispatch_node)
graph.add_node("synthesize", synthesize_node)
graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "dispatch")
graph.add_edge("dispatch", "synthesize")
graph.add_edge("synthesize", END)
app = graph.compile()


def handle(request: str) -> str:
    with runtime.run_context(name=request[:60]):
        seed_ctx = runtime.capture_context()
        final_state = app.invoke(
            {"request": request, "domains": [], "findings": [], "result": None, "_px_ctx": seed_ctx}
        )
    return final_state["result"]


REQUESTS = [
    "I was charged twice and the app also keeps crashing",
    "My invoice shows the wrong amount",
    "Getting an error every time I try to log in",
    "Do you offer student discounts?",
    "Refund me for the crash that lost my work",
]

if __name__ == "__main__":
    for req in REQUESTS:
        print(f"  {req}\n  -> {handle(req)}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
