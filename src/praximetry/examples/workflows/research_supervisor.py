"""Run with `python -m praximetry.examples.workflows.research_supervisor`."""

from __future__ import annotations

import concurrent.futures
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

import praximetry as px
from praximetry import runtime

from ._real import default_model, premium_model

px.init(project="research-supervisor")

MAX_AGENT_TURNS = 4


def _model(name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=name,
        base_url=os.environ.get("AI_ENDPOINT", "https://api.openai.com/v1"),
        api_key=os.environ["AI_API_KEY"],
    )


def _clean_tool_calls(message: AIMessage) -> AIMessage:
    """Strip leaked Harmony channel tags (e.g. "name<|channel|>commentary") off tool-call names."""
    if not message.tool_calls:
        return message
    for call in message.tool_calls:
        call["name"] = call["name"].split("<|", 1)[0]
    return message


_REASONING_BLOCK = re.compile(r"^\s*<reasoning>.*?</reasoning>\s*", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Drop a leaked <reasoning>...</reasoning> block (PRA-91: capture instead of discarding)."""
    return _REASONING_BLOCK.sub("", text, count=1)


def _record(result: str) -> str:
    runtime.record_call(response_text=result, cost_usd=0)
    return result


@tool
@px.stage("web_search")
def web_search(query: str) -> str:
    """Search the public web for pages relevant to the query."""
    return _record(
        f"4 results for '{query}': 2 industry blog posts, a TechCrunch "
        f"piece, and a competitor's own announcement, all from the last 90 days."
    )


@tool
@px.stage("fetch_url")
def fetch_url(url: str) -> str:
    """Fetch the contents of a specific URL found by web_search."""
    return _record(f"Fetched {url}: ~1,100 words, last updated 2026-06, no paywall.")


@tool
@px.stage("extract_facts")
def extract_facts(text: str) -> str:
    """Extract structured facts from fetched page text."""
    return _record(
        "Extracted: 3 pricing mentions, 1 customer quote, 1 stat "
        "('62% of buyers compare at least 3 vendors')."
    )


@tool
@px.stage("kb_search")
def kb_search(query: str) -> str:
    """Search the internal knowledge base for relevant docs."""
    return _record(
        f"3 KB matches for '{query}': 'Refund Policy v4', "
        f"'Checkout Flow Overview', 'Q2 Support Macros'."
    )


@tool
@px.stage("fetch_doc")
def fetch_doc(doc_id: str) -> str:
    """Fetch the full text of a specific internal doc."""
    return _record(f"Fetched '{doc_id}': last edited 6 weeks ago by the support team.")


@tool
@px.stage("summarize_doc")
def summarize_doc(text: str) -> str:
    """Summarize a fetched internal doc down to the relevant point."""
    return _record(
        "Summary: refunds are approved within 30 days, 10% restocking "
        "fee, 5 business day processing."
    )


@tool
@px.stage("competitor_search")
def competitor_search(query: str) -> str:
    """Identify the competitor(s) most relevant to the query."""
    return _record(
        f"Top match for '{query}': Competitor X (closest feature overlap), "
        f"Competitor Z (closest price point)."
    )


@tool
@px.stage("fetch_pricing_page")
def fetch_pricing_page(competitor: str) -> str:
    """Fetch a competitor's public pricing page."""
    return _record(f"{competitor} pricing: $9.82/unit list, 12% volume discount over 500 units.")


@tool
@px.stage("compare_pricing")
def compare_pricing(their_price: str) -> str:
    """Compare a fetched competitor price against our own list price."""
    return _record(
        f"Comparison vs {their_price}: we're priced ~3% lower at list, " f"before volume discounts."
    )


@tool
@px.stage("review_search")
def review_search(query: str) -> str:
    """Find user reviews relevant to the query."""
    return _record(
        f"Found 340 reviews mentioning '{query}' across 3 review sites " f"in the last quarter."
    )


@tool
@px.stage("fetch_reviews")
def fetch_reviews(source: str) -> str:
    """Fetch review text from a specific source."""
    return _record(f"Pulled 340 review excerpts from {source}.")


@tool
@px.stage("score_sentiment")
def score_sentiment(reviews: str) -> str:
    """Score the sentiment of a batch of fetched reviews."""
    return _record(
        "Sentiment: 4.7/5 average, 72% positive, main complaint theme " "is shipping delays."
    )


@tool
@px.stage("query_metrics")
def query_metrics(metric: str) -> str:
    """Query internal product metrics."""
    return _record(f"'{metric}': 3.5% of sessions over the trailing 30 days, up from 3.1%.")


@tool
@px.stage("fetch_report")
def fetch_report(name: str) -> str:
    """Fetch a saved internal analytics report."""
    return _record(f"Fetched report '{name}': last refreshed yesterday, 14 rows.")


@tool
@px.stage("compute_trend")
def compute_trend(series: str) -> str:
    """Compute the trend direction/magnitude for a metric series."""
    return _record(f"Trend for {series}: +0.4pp MoM, accelerating for 2 consecutive months.")


AGENT_TOOLS = {
    "web_researcher": [web_search, fetch_url, extract_facts],
    "kb_researcher": [kb_search, fetch_doc, summarize_doc],
    "competitor_researcher": [competitor_search, fetch_pricing_page, compare_pricing],
    "sentiment_researcher": [review_search, fetch_reviews, score_sentiment],
    "data_researcher": [query_metrics, fetch_report, compute_trend],
}

AGENT_SYSTEM = (
    "You are the {name}. You MUST call at least one tool before answering -- never "
    "answer from your own knowledge. Call tools one at a time, using each result to "
    "decide the next call. Only reply with plain text, a one-sentence finding, once "
    "you have real tool results to base it on."
)


def _run_agent(name: str, request: str) -> str:
    llm = _model(default_model()).bind_tools(AGENT_TOOLS[name])
    messages: list = [
        SystemMessage(AGENT_SYSTEM.format(name=name)),
        HumanMessage(request),
    ]
    tool_by_name = {t.name: t for t in AGENT_TOOLS[name]}

    for _ in range(MAX_AGENT_TURNS):
        reply = _clean_tool_calls(llm.invoke(messages))
        messages.append(reply)
        if not reply.tool_calls:
            return _strip_reasoning(reply.content)
        for call in reply.tool_calls:
            result = tool_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    last = messages[-1]
    return _strip_reasoning(last.content) if isinstance(last, AIMessage) else "no finding reached"


def _agent_dispatch_stage(name: str):
    """@px.stage-wrapped dispatch call, separate from the @tool wrapper so dispatch_node
    can call it directly under runtime.restore_context (see dispatch_node)."""

    @px.stage(name)
    def call(request: str) -> str:
        return _run_agent(name, request)

    return call


AGENT_DISPATCH_STAGES = {name: _agent_dispatch_stage(name) for name in AGENT_TOOLS}


def _agent_dispatch_tool(name: str):
    @tool(name, description=f"Dispatch to the {name} for its finding on this request.")
    def dispatch(request: str) -> str:
        return AGENT_DISPATCH_STAGES[name](request)

    return dispatch


AGENT_DISPATCH_TOOLS = [_agent_dispatch_tool(name) for name in AGENT_TOOLS]


@tool
def finalize_report(request: str, findings: str) -> str:
    """Call once you have enough findings to answer the request. `findings` is
    everything gathered so far, concatenated."""
    return _synthesize(request, findings)


@px.stage("synthesize")
def _synthesize(request: str, findings: str) -> str:
    reply = _model(premium_model()).invoke(
        [
            HumanMessage(f"Request: {request}\nFindings: {findings}\nSynthesize a reply."),
        ]
    )
    return _strip_reasoning(reply.content)


SUPERVISOR_SYSTEM = (
    "You are a research supervisor. Dispatch to whichever specialist agents are "
    "relevant to the request -- call several in the same turn when they're "
    "independent, or one at a time when a later choice depends on an earlier "
    "finding. Never ask the user a clarifying question -- proceed with reasonable "
    "assumptions instead. Once you have enough findings, call finalize_report with "
    "all of them concatenated; do not reply with plain text before that."
)


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    request: str
    result: str | None
    _px_ctx: dict | None  # runtime.capture_context() snapshot; only supervisor_node writes it


@px.stage("supervisor")
def _supervisor_llm_call(state: ResearchState) -> AIMessage:
    llm = _model(premium_model()).bind_tools(AGENT_DISPATCH_TOOLS + [finalize_report])
    return _clean_tool_calls(llm.invoke(state["messages"]))


def supervisor_node(state: ResearchState) -> dict:
    with runtime.restore_context(state.get("_px_ctx")):
        reply = _supervisor_llm_call(state)
        ctx = runtime.capture_context()
    return {"messages": [reply], "_px_ctx": ctx}


def dispatch_node(state: ResearchState) -> dict:
    """Hand-rolled vs. ToolNode so each concurrent dispatch restores px context first."""
    last = state["messages"][-1]
    calls = [c for c in last.tool_calls if c["name"] != "finalize_report"]
    ctx = state.get("_px_ctx")

    def run_one(call: dict) -> ToolMessage:
        with runtime.restore_context(ctx):
            result = AGENT_DISPATCH_STAGES[call["name"]](call["args"]["request"])
        return ToolMessage(content=result, tool_call_id=call["id"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(calls), 1)) as pool:
        results = list(pool.map(run_one, calls))
    return {"messages": results}


def finalize_node(state: ResearchState) -> dict:
    call = state["messages"][-1].tool_calls[0]
    with runtime.restore_context(state.get("_px_ctx")):
        result = finalize_report.invoke(call["args"])
    return {
        "messages": [ToolMessage(content=result, tool_call_id=call["id"])],
        "result": result,
    }


def route(state: ResearchState) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    if any(c["name"] == "finalize_report" for c in last.tool_calls):
        return "finalize"
    return "dispatch"


graph = StateGraph(ResearchState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("dispatch", dispatch_node)
graph.add_node("finalize", finalize_node)
graph.set_entry_point("supervisor")
graph.add_conditional_edges(
    "supervisor", route, {"dispatch": "dispatch", "finalize": "finalize", END: END}
)
graph.add_edge("dispatch", "supervisor")
graph.add_edge("finalize", END)
app = graph.compile()


def handle(request: str) -> str:
    with runtime.run_context(name=request[:60]):
        seed_ctx = runtime.capture_context()
        final_state = app.invoke(
            {
                "messages": [SystemMessage(SUPERVISOR_SYSTEM), HumanMessage(request)],
                "request": request,
                "result": None,
                "_px_ctx": seed_ctx,
            }
        )
    return final_state["result"] or _strip_reasoning(final_state["messages"][-1].content)


REQUESTS = [
    "How does our pricing compare to competitors, and what are users saying about it?",
    "What do our internal docs and recent metrics say about the checkout drop-off?",
    "Summarize public sentiment and any competitor moves in the last quarter.",
    "What does the knowledge base say about our refund policy, and how are reviews trending?",
    "Pull together web coverage, competitor pricing, and our own usage data for the board deck.",
]

if __name__ == "__main__":
    for req in REQUESTS:
        print(f"  {req}\n  -> {handle(req)}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
