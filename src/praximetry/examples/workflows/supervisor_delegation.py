"""Try:
python -m praximetry.examples.workflows.supervisor_delegation
"""

import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import praximetry as px

from ._real import chat_model, clean_content, default_model, premium_model

px.init(project="supervisor-delegation")

SYSTEM = (
    "You are a triage supervisor. Decide which domains this request touches. "
    'Respond with JSON only, no other text: {"domains": [...]}, where each '
    "entry is one of billing, technical, general -- at least one."
)


class DomainClassification(BaseModel):
    domains: list[Literal["billing", "technical", "general"]] = Field(
        description="Every relevant domain, at least one."
    )


@px.stage("supervisor")
async def supervisor(request: str) -> list[str]:
    await asyncio.sleep(0)
    model = premium_model()
    reply = await chat_model(model).ainvoke([SystemMessage(SYSTEM), HumanMessage(request)])
    result = DomainClassification.model_validate_json(clean_content(reply, model))
    domains = list(dict.fromkeys(result.domains))
    return domains or ["general"]


def _agent(stage_name: str):
    @px.stage(stage_name)
    async def agent(request: str) -> str:
        await asyncio.sleep(0)
        reply = await chat_model(default_model()).ainvoke(
            [HumanMessage(f"As the {stage_name}, give a one-line finding for: {request}")]
        )
        return clean_content(reply, default_model())

    return agent


billing_agent = _agent("billing_agent")
technical_agent = _agent("technical_agent")
general_agent = _agent("general_agent")

AGENTS = {"billing": billing_agent, "technical": technical_agent, "general": general_agent}


@px.stage("synthesize")
def synthesize(request: str, findings: list[str]) -> str:
    # Parents to `supervisor`, not any agent -- see ARCHITECTURES.md's gather/join note.
    joined = " ".join(findings)
    reply = chat_model(premium_model()).invoke(
        [HumanMessage(f"Request: {request}\nFindings: {joined}\nSynthesize a reply.")]
    )
    return clean_content(reply, premium_model())


async def handle(request: str) -> str:
    domains = await supervisor(request)
    findings = list(await asyncio.gather(*(AGENTS[d](request) for d in domains)))
    return synthesize(request, findings)


REQUESTS = [
    "I was charged twice and the app also keeps crashing",
    "My invoice shows the wrong amount",
    "Getting an error every time I try to log in",
    "Do you offer student discounts?",
    "Refund me for the crash that lost my work",
]

if __name__ == "__main__":
    for req in REQUESTS:
        print(f"  {req}\n  -> {asyncio.run(handle(req))}\n")
    print("Traffic recorded. Try: praximetry-cloud detect")
