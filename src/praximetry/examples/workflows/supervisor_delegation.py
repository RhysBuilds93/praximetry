"""Run with `python -m praximetry.examples.workflows.supervisor_delegation`."""

import asyncio

import praximetry as px

from ._real import default_model, premium_model, real_chat

px.init(project="supervisor-delegation")

SYSTEM = "You are a triage supervisor. List every relevant domain, comma-separated: billing, technical, general."

DOMAINS = ("billing", "technical", "general")


def _parse_domains(raw: str) -> list[str]:
    low = raw.lower()
    found = [d for d in DOMAINS if d in low]
    return found or ["general"]


@px.stage("supervisor")
async def supervisor(request: str) -> list[str]:
    await asyncio.sleep(0)
    raw = real_chat(
        premium_model(),
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}],
    )
    return _parse_domains(raw)


def _agent(stage_name: str):
    @px.stage(stage_name)
    async def agent(request: str) -> str:
        await asyncio.sleep(0)
        return real_chat(
            default_model(),
            [
                {
                    "role": "user",
                    "content": f"As the {stage_name}, give a one-line finding for: {request}",
                }
            ],
        )

    return agent


billing_agent = _agent("billing_agent")
technical_agent = _agent("technical_agent")
general_agent = _agent("general_agent")

AGENTS = {"billing": billing_agent, "technical": technical_agent, "general": general_agent}


@px.stage("synthesize")
def synthesize(request: str, findings: list[str]) -> str:
    # Parents to `supervisor`, not any agent -- see ARCHITECTURES.md's gather/join note.
    joined = " ".join(findings)
    return real_chat(
        premium_model(),
        [
            {
                "role": "user",
                "content": f"Request: {request}\nFindings: {joined}\nSynthesize a reply.",
            }
        ],
    )


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
