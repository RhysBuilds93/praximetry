"""TruLens Record/Cost ingestion, using stand-ins for the real dataclasses."""

from dataclasses import dataclass

from praximetry import trulens
from praximetry.store import get_store


@dataclass
class FakeCost:
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    cost: float = 0.0


@dataclass
class FakeRecord:
    app_id: str = "my-app"
    main_output: str = "the answer"
    cost: FakeCost = None


def test_record_trulens_record():
    record = FakeRecord(cost=FakeCost(n_prompt_tokens=100, n_completion_tokens=20))
    call = trulens.record_trulens_record(record, model="gpt-4o", stage="qa_chain")

    assert call.provider == "trulens"
    assert call.model == "gpt-4o"
    assert call.stage == "qa_chain"
    assert call.input_tokens == 100 and call.output_tokens == 20
    assert call.output_text == "the answer"
    assert call.metadata["app_id"] == "my-app"
    assert call.cost_usd > 0

    saved = get_store().calls()
    assert saved[-1].id == call.id
