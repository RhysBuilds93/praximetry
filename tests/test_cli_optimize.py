"""`praximetry optimize` and `praximetry apply` CLI commands, end to end
against a stub hosted API. The real hosted server (trial loop, scoring,
policy storage) lives in the closed-source cloud repo — out of scope here.
This proves the CLI's capture -> submit wiring for `optimize` and the
fetch -> write-overrides wiring for `apply`.
"""
import json

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import praximetry
from praximetry.cli import app as cli_app

runner = CliRunner()

VALID_KEY = "px_live_stub_key"

EXAMPLES = [
    {"id": "e1", "stage": "checkout", "input": "buy 2 widgets"},
]


def _stub_app(*, winner: dict | None = "missing", optimize_result: dict | None = None) -> FastAPI:
    """`winner="missing"` (the default) means /api/optimize/winner 404s —
    no completed run at all. Pass an explicit dict (possibly with None
    fields) or None to mean "run completed, no winner found"."""
    app = FastAPI()
    received = {}

    @app.get("/api/eval/corpus")
    def corpus(stage: str | None = None, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        return [e for e in EXAMPLES if not stage or e["stage"] == stage]

    @app.post("/api/optimize/captures")
    def optimize_captures(body: dict, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        received["body"] = body
        return optimize_result or {
            "stage": body["stage"], "examples": len(body["captured_request"]),
            "winner": {"model": "amazon.nova-pro-v1:0"}, "savings_pct": 0.34,
            "truncated": False, "errors": [],
        }

    @app.get("/api/optimize/winner")
    def optimize_winner(stage: str, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        if winner == "missing":
            raise HTTPException(status_code=404, detail="no completed run")
        if winner is None:
            return {"stage": stage, "model": None, "transforms": None,
                    "experiment_id": None, "savings_pct": None}
        return winner

    app.state.received = received
    return app


def _patch_client(monkeypatch, http):
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod
    monkeypatch.setattr(hosted_mod, "client_from_env",
                        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http))


def test_optimize_submits_a_capture_and_prints_the_winner(monkeypatch, fake_llm):
    http = TestClient(_stub_app())
    _patch_client(monkeypatch, http)

    @praximetry.stage("checkout")
    def checkout(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["optimize", "--stage", "checkout"])

    assert result.exit_code == 0, result.output
    assert "winner found" in result.output
    body = http.app.state.received["body"]
    assert body["stage"] == "checkout"
    assert body["captured_request"]["stage"] == "checkout"
    assert body["captured_request"]["model"] == "gpt-4o"


def test_optimize_passes_flags_through(monkeypatch, fake_llm):
    http = TestClient(_stub_app())
    _patch_client(monkeypatch, http)

    @praximetry.stage("checkout")
    def checkout(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, [
        "optimize", "--stage", "checkout",
        "--model", "amazon.nova-pro-v1:0", "--model", "gpt-4o-mini",
        "--transform", "compact", "--quality-tolerance", "0.05", "--max-trials", "3",
    ])

    assert result.exit_code == 0, result.output
    body = http.app.state.received["body"]
    assert body["candidate_models"] == ["amazon.nova-pro-v1:0", "gpt-4o-mini"]
    assert body["transforms"] == ["compact"]
    assert body["quality_tolerance"] == 0.05
    assert body["max_trials"] == 3


def test_optimize_no_winner_is_reported_but_not_an_error(monkeypatch, fake_llm):
    http = TestClient(_stub_app(optimize_result={
        "stage": "checkout", "examples": 1, "winner": None, "truncated": False, "errors": [],
    }))
    _patch_client(monkeypatch, http)

    @praximetry.stage("checkout")
    def checkout(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["optimize", "--stage", "checkout"])

    assert result.exit_code == 0, result.output
    assert "no winner found" in result.output


def test_optimize_empty_corpus_is_an_error(monkeypatch):
    http = TestClient(_stub_app())
    _patch_client(monkeypatch, http)

    result = runner.invoke(cli_app, ["optimize", "--stage", "no-such-stage"])

    assert result.exit_code != 0, result.output
    assert "No golden examples" in result.output


def test_optimize_capture_failure_is_an_error(monkeypatch):
    http = TestClient(_stub_app())
    _patch_client(monkeypatch, http)

    @praximetry.stage("checkout")
    def checkout(text):
        return "never calls record_call"

    result = runner.invoke(cli_app, ["optimize", "--stage", "checkout"])

    assert result.exit_code != 0, result.output
    assert "nothing to capture" in result.output


def test_optimize_without_an_api_key_is_an_error(monkeypatch):
    monkeypatch.delenv("PRAXIMETRY_API_KEY", raising=False)

    result = runner.invoke(cli_app, ["optimize", "--stage", "checkout"])

    assert result.exit_code != 0, result.output
    assert "PRAXIMETRY_API_KEY" in result.output


def test_apply_writes_the_winner_to_overrides_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    http = TestClient(_stub_app(winner={
        "stage": "checkout", "model": "amazon.nova-pro-v1:0", "transforms": ["compact"],
        "experiment_id": "exp1", "savings_pct": 0.34,
    }))
    _patch_client(monkeypatch, http)

    result = runner.invoke(cli_app, ["apply", "--stage", "checkout"])

    assert result.exit_code == 0, result.output
    overrides = json.loads((tmp_path / ".praximetry" / "overrides.json").read_text())
    assert overrides["stages"]["checkout"]["model"] == "amazon.nova-pro-v1:0"
    assert overrides["stages"]["checkout"]["transforms"] == ["compact"]
    assert overrides["stages"]["checkout"]["experiment_id"] == "exp1"
    assert overrides["stages"]["checkout"]["savings_pct"] == 0.34


def test_apply_preserves_other_stages_in_overrides_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".praximetry").mkdir()
    (tmp_path / ".praximetry" / "overrides.json").write_text(json.dumps({
        "stages": {"other_stage": {"model": "existing", "transforms": []}}
    }))
    http = TestClient(_stub_app(winner={
        "stage": "checkout", "model": "amazon.nova-pro-v1:0", "transforms": [],
        "experiment_id": "exp1", "savings_pct": 0.1,
    }))
    _patch_client(monkeypatch, http)

    result = runner.invoke(cli_app, ["apply", "--stage", "checkout"])

    assert result.exit_code == 0, result.output
    overrides = json.loads((tmp_path / ".praximetry" / "overrides.json").read_text())
    assert overrides["stages"]["other_stage"]["model"] == "existing"
    assert overrides["stages"]["checkout"]["model"] == "amazon.nova-pro-v1:0"


def test_apply_no_winner_exits_zero_and_does_not_write(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    http = TestClient(_stub_app(winner=None))
    _patch_client(monkeypatch, http)

    result = runner.invoke(cli_app, ["apply", "--stage", "checkout"])

    assert result.exit_code == 0, result.output
    assert "nothing to apply" in result.output
    assert not (tmp_path / ".praximetry" / "overrides.json").exists()


def test_apply_no_run_yet_is_an_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    http = TestClient(_stub_app(winner="missing"))
    _patch_client(monkeypatch, http)

    result = runner.invoke(cli_app, ["apply", "--stage", "checkout"])

    assert result.exit_code == 1, result.output
    assert "no optimize run found" in result.output
    assert not (tmp_path / ".praximetry" / "overrides.json").exists()


def test_apply_without_an_api_key_is_an_error(monkeypatch):
    monkeypatch.delenv("PRAXIMETRY_API_KEY", raising=False)

    result = runner.invoke(cli_app, ["apply", "--stage", "checkout"])

    assert result.exit_code == 2, result.output
    assert "PRAXIMETRY_API_KEY" in result.output
