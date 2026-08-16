"""Workflow 5: multi-hop tool-using research agent, modelled on GAIA.

GAIA (https://huggingface.co/datasets/gaia-benchmark/GAIA) is the "general
assistant" benchmark: real questions that need several tool-assisted hops, with
a single short `Final answer` scored by exact match, and questions bucketed into
Levels 1-3 by how many hops/tools they need. Exact match is a brutal scorer, and
that's the point - it's exactly the setup where a naive "just use the cheapest
model" cost cut shows up immediately as a quality regression.

Pipeline:
    decompose    (LLM)  -> the hops needed, as knowledge-graph keys
    lookup       (tool) -> resolves each hop against a retrieval index
    final_answer (LLM)  -> one short answer, nothing else
    answer       (task) -> the whole chain, scored `exact` against Final answer

Try:
    python -m praximetry.examples.workflows.gaia_multihop
    praximetry eval --stage final_answer -m praximetry.examples.workflows.gaia_multihop --fail-under 0.9
    praximetry optimize --stage final_answer -m praximetry.examples.workflows.gaia_multihop
    praximetry apply --stage final_answer
"""

from __future__ import annotations

import json

import praximetry as px

from ._real import default_model, premium_model, real_chat

px.init(project="gaia-multihop")

INDEX: dict[str, str] = {
    "deepmind.acquirer": "Google",
    "google.ipo_year": "2004",
    "google.hq_city": "Mountain View",
    "transformer_paper.title": "Attention Is All You Need",
    "transformer_paper.year": "2017",
    "transformer_paper.lead_author": "Ashish Vaswani",
    "python.creator": "Guido van Rossum",
    "python.creator_birth_country": "Netherlands",
    "netherlands.capital": "Amsterdam",
    "apollo11.commander": "Neil Armstrong",
    "apollo11.landing_year": "1969",
    "neil_armstrong.birth_state": "Ohio",
    "ohio.capital": "Columbus",
}

QUESTIONS: list[str] = [
    "In which year did the company that acquired DeepMind go public?",
    "What is the capital of the country where Python's creator was born?",
    "What is the capital of the home state of the Apollo 11 commander?",
    "In what year was the paper that introduced the Transformer published?",
    "Who was the lead author of the paper that introduced the Transformer?",
]

VERBOSE_GUIDE = (
    "You are a meticulous research assistant. Think step by step.\n"
    "Think step by step and use tools where needed.\n"
    "Report your Final answer as a number OR as few words as possible.\n"
    "Do not apologise. Do not add units unless asked. Do not add a full stop.\n"
    "Report your Final answer as a number OR as few words as possible.\n"
)


@px.stage("decompose")
def decompose(question: str) -> list[str]:
    messages = [
        {"role": "system", "content": VERBOSE_GUIDE},
        {
            "role": "user",
            "content": f"Available index keys:\n{json.dumps(sorted(INDEX), indent=4)}\n\n"
            "List the lookup keys needed, in order, comma-separated, using only "
            f"keys from the list above and nothing else.\nQuestion: {question}",
        },
    ]
    raw = real_chat(default_model(), messages)
    return [k.strip() for k in raw.split(",") if k.strip() in INDEX]


@px.stage("lookup")
def lookup(hops: list[str]) -> list[str]:
    return [f"- {h} = {INDEX[h]}" for h in hops if h in INDEX]


@px.stage("final_answer")
def final_answer(question: str, evidence: list[str]) -> str:
    messages = [
        {"role": "system", "content": VERBOSE_GUIDE},
        {
            "role": "user",
            "content": "Evidence:\n"
            + "\n".join(evidence)
            + f"\n\nGive only the final answer, no sentence.\nQuestion: {question}",
        },
    ]
    return real_chat(premium_model(), messages)


@px.stage("answer")
def answer(question: str) -> str:
    return final_answer(question, lookup(decompose(question)))


if __name__ == "__main__":
    for q in QUESTIONS:
        print(f"  {q}\n    -> {answer(q)}")
    print("\nTraffic recorded. Try: praximetry-cloud detect")
