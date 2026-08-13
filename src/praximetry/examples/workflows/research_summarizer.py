"""Workflow 3: research agent (plan -> summarize chunks -> synthesize).

The map-reduce document pattern: a cheap planning step and N per-chunk
summaries all running on a frontier model, where per-chunk work is the obvious
downgrade target. Also demonstrates async stages.

Try:
    python -m praximetry.examples.workflows.research_summarizer
    praximetry eval --stage summarize_chunk -m praximetry.examples.workflows.research_summarizer --fail-under 0.9
    praximetry optimize --stage summarize_chunk -m praximetry.examples.workflows.research_summarizer
    praximetry apply --stage summarize_chunk
"""
import asyncio

import praximetry as px

from ._real import default_model, premium_model, real_chat

px.init(project="research-agent")

DOCS = {
    "solar": ("Solar deployment grew 28% year over year. Panel prices fell 12%. "
              "Grid connection queues remain the main bottleneck in the US and EU. "
              "Battery co-location now appears in 40% of new utility projects."),
    "wind": ("Offshore wind auctions cleared at record-low subsidies. "
             "Turbine blade recycling remains unsolved at scale. "
             "Floating platforms opened deep-water sites in Japan and Norway."),
    "storage": ("Grid-scale battery costs dropped below $100/kWh. "
                "Four-hour duration is now standard for new solar pairings. "
                "Sodium-ion entered commercial pilots as a lithium alternative."),
}


@px.stage("plan")
def plan(question: str) -> list[str]:
    messages = [
        {"role": "user", "content":
            f"Which topics answer: {question}? Options: {list(DOCS)}\n\n"
            "Reply with only the matching option names, comma-separated, "
            "nothing else."},
    ]
    answer = real_chat(default_model(), messages)
    topics = [t.strip().lower() for t in answer.split(",") if t.strip().lower() in DOCS]
    return topics or ["solar"]


@px.stage("summarize_chunk")
async def summarize_chunk(topic: str) -> str:
    messages = [
        {"role": "system", "content":
            "Summarize the single most important fact in one sentence. "
            "Preserve any specific numbers or percentages from the text verbatim."},
        {"role": "user", "content": f"topic: {topic} text: {DOCS.get(topic, '')}"},
    ]
    return real_chat(premium_model(), messages)


@px.stage("synthesize")
def synthesize(question: str, summaries: list[str]) -> str:
    messages = [
        {"role": "user", "content": f"Q: {question}\nNotes: {' '.join(summaries)}\nSynthesize."},
    ]
    return real_chat(default_model(), messages)


async def research(question: str) -> str:
    topics = plan(question)
    summaries = await asyncio.gather(*(summarize_chunk(t) for t in topics))
    return synthesize(question, list(summaries))


if __name__ == "__main__":
    for q in ["How are solar and storage costs trending?",
              "What is blocking wind expansion?",
              "Status of solar, wind and storage?"]:
        print(f"  Q: {q}\n  A: {asyncio.run(research(q))[:100]}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
