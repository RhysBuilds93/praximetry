import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from praximetry import config, runtime
from praximetry.cli import app
from praximetry.eval import CaptureError, hosted
from praximetry.eval.replay import captured_request_from_call
from praximetry.models import Call


def test_mapping_preserves_request_tools_and_provider_shape():
    tools = [{"name": "search", "input_schema": {"type": "object"}}]
    cases = [
        ([{"role": "system", "content": "classify"}, {"role": "user", "content": "hi"}], tools),
        ([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], tools),
        ([{"role": "user", "content": "no tools"}], []),
    ]
    for messages, request_tools in cases:
        call = Call(run_id="r", stage="explicit", derived_label="automatic", provider="anthropic",
                    model="claude-sonnet-4-5", messages=messages, metadata={"request_complete": True},
                    tools=request_tools)
        capture = captured_request_from_call(Call.model_validate_json(call.model_dump_json()))
        assert capture.stage == "explicit"
        assert capture.messages == messages
        assert capture.tools == request_tools
        assert not call.tool_calls


@pytest.mark.parametrize("changes", [
    {"messages": []}, {"metadata": {"redacted": True}},
    {"messages": [{"role": "assistant", "content": "previous reply"}]},
    {"metadata": {"multi_turn": True}},
])
def test_ineligible_requests_are_not_reconstructed(changes):
    fields = dict(run_id="r", stage="x", provider="openai", model="gpt-4o",
                  messages=[{"role": "user", "content": "hi"}])
    fields.update(changes)
    fields["metadata"] = {"request_complete": True, **fields.get("metadata", {})}
    with pytest.raises(CaptureError, match="not eligible"):
        captured_request_from_call(Call(**fields))


def test_label_replay_uses_project_and_exact_example_correlation(monkeypatch):
    api = FastAPI()
    submissions = []

    @api.get("/api/eval/corpus")
    def corpus(stage: str, project: str):
        assert project == "test"
        return [{"id": "golden", "stage": stage, "input": "unused", "expected": "ok"}]

    @api.get("/api/eval/config")
    def eval_config():
        return {"fail_under": 0.9}

    @api.post("/api/eval/captures")
    def captures(body: dict):
        submissions.append(body)
        return {"quality": 1, "pass_rate": 1, "cost_usd": 0.01, "count": 1}

    with runtime.run_context():
        runtime.record_call(stage="shared", provider="openai", model="gpt-4o",
                            messages=[{"role": "user", "content": "correct project"}],
                            metadata={"example_id": "golden"})
    config.get_config().project = "other"
    with runtime.run_context():
        runtime.record_call(stage="shared", provider="openai", model="gpt-4o",
                            messages=[{"role": "user", "content": "wrong project"}],
                            metadata={"example_id": "golden"})
    with TestClient(api) as http:
        monkeypatch.setattr(hosted, "client_from_env", lambda: hosted.CloudClient("", "test", http))
        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--label", "shared", "--project", "test"])
        assert result.exit_code == 0, result.output
        assert "historical single-shot requests" in result.output
        assert "current agent code" in result.output
        assert submissions[0]["captures"][0]["messages"][0]["content"] == "correct project"
        assert "git_commit" not in submissions[0]
        missing = runner.invoke(app, ["eval", "--label", "unknown", "--project", "test"])
        assert missing.exit_code == 2
        assert "shared" in missing.output


def test_replay_requires_a_non_empty_project():
    from praximetry.eval.replay import recorded_calls

    with pytest.raises(CaptureError, match="non-empty project"):
        recorded_calls("", "anything")


def test_summary_uses_effective_target_and_reports_stale_policies(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with runtime.run_context():
        with runtime.stage_context("outer"):
            with runtime.label("inner"):
                runtime.record_call(provider="openai", model="gpt-4o")
    path = tmp_path / ".praximetry" / "overrides.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"projects": {"test": {"stale": {"model": "x"}}}}))
    result = CliRunner().invoke(app, ["summary"])
    assert result.exit_code == 0, result.output
    assert "inner" in result.output
    assert "outer>inner" not in result.output
    assert "stale" in result.output


def test_replay_never_assigns_unrelated_golden_example(monkeypatch):
    from praximetry.eval.dataset import Example
    from praximetry.eval.replay import replay_examples

    call = Call(run_id="r", stage="x", provider="openai", model="gpt-4o",
                messages=[{"role": "user", "content": "unrelated"}])
    with pytest.raises(CaptureError, match="correlated"):
        replay_examples([call], [Example(id="golden", stage="x", input="input")])


def test_optimize_ranks_known_cost_and_label_apply_preserves_legacy(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    received = []
    api = FastAPI()

    @api.post("/api/optimize/captures")
    def optimize(body: dict):
        received.append(body)
        return {"stage": body["stage"], "examples": 1, "winner": {"model": "gpt-4o-mini"}}

    @api.get("/api/optimize/winner")
    def winner():
        return {"model": "gpt-4o-mini", "transforms": []}

    for stage, model, cost in [("cheap", "gpt-4o", 1), ("costly", "gpt-4o", 5),
                               ("unpriced", "unknown-model", 100)]:
        runtime.record_call(stage=stage, provider="openai", model=model, cost_usd=cost,
                            messages=[{"role": "user", "content": stage}])
    path = tmp_path / ".praximetry" / "overrides.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"stages": {"legacy": {"model": "old"}}}))
    with TestClient(api) as http:
        monkeypatch.setattr(hosted, "client_from_env", lambda: hosted.CloudClient("", "test", http))
        runner = CliRunner()
        result = runner.invoke(app, ["optimize"])
        assert result.exit_code == 0, result.output
        assert received[0]["stage"] == "costly"
        assert received[0]["project"] == "test"
        applied = runner.invoke(app, ["apply", "--label", "costly"])
        assert applied.exit_code == 0, applied.output
        document = json.loads(path.read_text())
        assert document["stages"] == {"legacy": {"model": "old"}}
        assert document["projects"]["test"]["costly"]["model"] == "gpt-4o-mini"
        missing = runner.invoke(app, ["optimize", "--label", "missing"])
        assert missing.exit_code == 1
        assert "costly" in missing.output


def test_summary_does_not_merge_equal_labels_across_projects():
    for project in ("test", "other"):
        config.get_config().project = project
        with runtime.run_context():
            runtime.record_call(stage="shared", provider="fake", model="gpt-4o")
    config.get_config().project = "test"
    runner = CliRunner()
    result = runner.invoke(app, ["summary", "--project", "test"])
    assert result.exit_code == 0, result.output
    assert "calls=1" in result.output
    assert "shared" in result.output
    assert runner.invoke(app, ["summary", "--project", "other"]).exit_code == 0


def test_optimize_relays_multi_turn_rejection(monkeypatch):
    api = FastAPI()

    @api.post("/api/optimize/captures")
    def reject(body: dict):
        from fastapi import HTTPException

        raise HTTPException(422, "multi-turn, not trialable from one capture")

    runtime.record_call(stage="conversation", provider="openai", model="gpt-4o",
                        messages=[{"role": "user", "content": "hello"}])
    with TestClient(api) as http:
        monkeypatch.setattr(hosted, "client_from_env", lambda: hosted.CloudClient("", "test", http))
        result = CliRunner().invoke(app, ["optimize", "--label", "conversation"])
    assert result.exit_code == 1, result.output
    assert "not trialable from one capture" in result.output
