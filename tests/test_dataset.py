"""Dataset.load()'s JSONL parsing and the for_stage/stages helpers -- untested elsewhere."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from praximetry.eval.dataset import Dataset, Example


def test_load_skips_blank_lines_and_auto_assigns_ids(tmp_path):
    p = tmp_path / "examples.jsonl"
    p.write_text(
        '{"stage": "classify", "input": "a"}\n'
        "\n"
        '{"id": "custom", "stage": "classify", "input": "b"}\n'
        "  \n"
        '{"stage": "summarize", "input": {"text": "c"}}\n'
    )

    ds = Dataset.load(p)

    assert len(ds.examples) == 3
    assert ds.examples[1].id == "custom"
    assert len({e.id for e in ds.examples}) == 3
    assert ds.path == str(p)


def test_load_preserves_scalar_list_and_dict_input_shapes(tmp_path):
    p = tmp_path / "examples.jsonl"
    p.write_text(
        '{"stage": "s", "input": "scalar"}\n'
        '{"stage": "s", "input": ["a", "b"]}\n'
        '{"stage": "s", "input": {"k": "v"}}\n'
    )

    ds = Dataset.load(p)

    assert [e.input for e in ds.examples] == ["scalar", ["a", "b"], {"k": "v"}]


def test_for_stage_filters_and_keeps_path():
    ds = Dataset(
        examples=[
            Example(stage="classify", input="a"),
            Example(stage="summarize", input="b"),
        ],
        path="corpus.jsonl",
    )

    filtered = ds.for_stage("classify")

    assert [e.stage for e in filtered.examples] == ["classify"]
    assert filtered.path == "corpus.jsonl"


def test_stages_is_sorted_and_deduped():
    ds = Dataset(
        examples=[
            Example(stage="summarize", input="a"),
            Example(stage="classify", input="b"),
            Example(stage="classify", input="c"),
        ]
    )

    assert ds.stages == ["classify", "summarize"]


def test_level_defaults_to_call_for_old_payloads(tmp_path: Path) -> None:
    path = tmp_path / "old.jsonl"
    path.write_text('{"stage": "s", "input": "a"}\n')

    assert Example(stage="s", input="a").level == "call"
    assert Dataset.load(path).examples[0].level == "call"


def test_level_round_trips_through_jsonl_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "levels.jsonl"
    originals = [
        Example(id="c", stage="s", input="a", level="call"),
        Example(id="t", stage="s", input="b", level="stage"),
    ]
    path.write_text("\n".join(e.model_dump_json() for e in originals))

    assert Dataset.load(path).examples == originals
    assert [Example(**e.model_dump()).level for e in originals] == ["call", "stage"]


def test_invalid_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Example.model_validate({"stage": "s", "input": "a", "level": "trace"})
