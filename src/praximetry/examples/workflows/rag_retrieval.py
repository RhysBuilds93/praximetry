"""Try:
python -m praximetry.examples.workflows.rag_retrieval

Uses BM25 rather than an embedding model: the example's AI_ENDPOINT
(Bedrock's OpenAI-compatible chat endpoint) doesn't serve /embeddings.
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

import praximetry as px
from praximetry import runtime

from ._real import chat_model, clean_content, default_model

px.init(project="rag-retrieval")

CORPUS = [
    Document(
        page_content="Refunds are issued to the original payment method within 5-7 business days.",
        metadata={"key": "refunds"},
    ),
    Document(
        page_content="Standard shipping takes 3-5 business days; express takes 1-2 business days.",
        metadata={"key": "shipping"},
    ),
    Document(
        page_content="All products carry a 12-month warranty covering manufacturing defects.",
        metadata={"key": "warranty"},
    ),
    Document(
        page_content="Items can be returned within 30 days of delivery in original condition.",
        metadata={"key": "returns"},
    ),
]

SYSTEM = "Answer using only the provided context. Be concise."

_bm25 = BM25Okapi([doc.page_content.lower().split() for doc in CORPUS])


@px.stage("embed_query")
def embed_query(question: str) -> list[str]:
    tokens = question.lower().split()
    runtime.record_call(output_text=str(tokens), cost_usd=0)
    return tokens


@px.stage("vector_search")
def vector_search(query_tokens: list[str], top_k: int = 2) -> list[Document]:
    scores = _bm25.get_scores(query_tokens)
    ranked = sorted(range(len(CORPUS)), key=lambda i: scores[i], reverse=True)[:top_k]
    result = [CORPUS[i] for i in ranked]
    runtime.record_call(output_text=str([d.metadata["key"] for d in result]), cost_usd=0)
    return result


@px.stage("generate")
def generate(question: str, chunks: list[Document]) -> str:
    context = "\n".join(d.page_content for d in chunks)
    reply = chat_model(default_model()).invoke(
        [
            SystemMessage(SYSTEM),
            HumanMessage(f"Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    return clean_content(reply, default_model())


def handle(question: str) -> str:
    with px.run_context():
        tokens = embed_query(question)
        chunks = vector_search(tokens)
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
