"""Try:
python -m praximetry.examples.workflows.rag_retrieval
"""

import math
import re
from collections import Counter

import praximetry as px
from praximetry import runtime

from ._real import default_model, real_chat

px.init(project="rag-retrieval")

CORPUS = {
    "refunds": "Refunds are issued to the original payment method within 5-7 business days.",
    "shipping": "Standard shipping takes 3-5 business days; express takes 1-2 business days.",
    "warranty": "All products carry a 12-month warranty covering manufacturing defects.",
    "returns": "Items can be returned within 30 days of delivery in original condition.",
}

SYSTEM = "Answer using only the provided context. Be concise."


def _vectorize(text: str) -> Counter:
    return Counter(re.findall(r"[a-z]+", text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (norm_a * norm_b)


CORPUS_VECTORS = {key: _vectorize(text) for key, text in CORPUS.items()}


@px.stage("embed_query")
def embed_query(question: str) -> Counter:
    result = _vectorize(question)
    runtime.record_call(response_text=str(dict(result)), cost_usd=0)
    return result


@px.stage("vector_search")
def vector_search(query_vector: Counter, top_k: int = 2) -> list[str]:
    scored = sorted(
        CORPUS_VECTORS, key=lambda k: _cosine(query_vector, CORPUS_VECTORS[k]), reverse=True
    )
    result = scored[:top_k]
    runtime.record_call(response_text=str(result), cost_usd=0)
    return result


@px.stage("generate")
def generate(question: str, chunk_keys: list[str]) -> str:
    context = "\n".join(CORPUS[k] for k in chunk_keys)
    return real_chat(
        default_model(),
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )


def handle(question: str) -> str:
    vector = embed_query(question)
    chunks = vector_search(vector)
    return generate(question, chunks)


QUESTIONS = [
    "How long do refunds take?",
    "What's your return policy?",
    "How long is the warranty?",
    "How fast is express shipping?",
]

if __name__ == "__main__":
    for q in QUESTIONS:
        print(f"  {q}\n  -> {handle(q)}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
