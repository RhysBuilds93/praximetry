"""`praximetry eval` CLI command, end to end against a stub hosted API.

Corpus fetch and capture push go through a small stub FastAPI app (the real
hosted server lives in the closed-source cloud repo and does real scoring —
out of scope here). This proves the CLI's capture -> push -> gate wiring.
"""
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import praximetry
from praximetry.cli import app as cli_app

runner = CliRunner()

VALID_KEY = "px_live_stub_key"

EXAMPLES = [
    {"id": "e1", "stage": "plan_action", "input": "track my order"},
    {"id": "e2", "stage": "plan_action", "input": "cancel my order"},
]


def _stub_app(score_by_example: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/api/eval/corpus")
    def corpus(stage: str | None = None, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        return [e for e in EXAMPLES if not stage or e["stage"] == stage]

    @app.post("/api/eval/captures")
    def captures(body: dict, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        caps = body["captures"]
        scores = [score_by_example.get(c["example_id"], 1.0) for c in caps]
        quality = sum(scores) / len(scores) if scores else 0.0
        return {
            "status": "saved", "stage": body["stage"], "count": len(caps),
            "quality": quality, "pass_rate": quality, "cost_usd": 0.01 * len(caps),
            "skipped": [],
        }

    return app


def test_eval_passes_when_quality_meets_fail_under(monkeypatch, fake_llm):
    http = TestClient(_stub_app(score_by_example={"e1": 1.0, "e2": 1.0}))
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod
    monkeypatch.setattr(hosted_mod, "client_from_env",
                        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http))

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 0, result.output
    assert "quality=1.00" in result.output
    assert "PASS" in result.output


def test_eval_fails_under_threshold(monkeypatch, fake_llm):
    http = TestClient(_stub_app(score_by_example={"e1": 0.0, "e2": 0.0}))
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod
    monkeypatch.setattr(hosted_mod, "client_from_env",
                        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http))

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_eval_partial_skip_still_pushes_captured_examples(monkeypatch, fake_llm):
    http = TestClient(_stub_app(score_by_example={"e1": 1.0}))
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod
    monkeypatch.setattr(hosted_mod, "client_from_env",
                        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http))

    @praximetry.stage("plan_action")
    def plan_action(text):
        if text == "cancel my order":
            return "no llm call for this one"  # never calls record_call
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert "[SKIP] plan_action/e2" in result.output
    assert "examples=1" in result.output


def test_eval_empty_corpus_is_unusable(monkeypatch):
    http = TestClient(_stub_app(score_by_example={}))
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod
    monkeypatch.setattr(hosted_mod, "client_from_env",
                        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http))

    result = runner.invoke(cli_app, ["eval", "--stage", "no-such-stage"])

    assert result.exit_code == 2, result.output
    assert "No golden examples" in result.output


def test_eval_without_an_api_key_is_unusable(monkeypatch):
    monkeypatch.delenv("PRAXIMETRY_API_KEY", raising=False)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 2, result.output
    assert "PRAXIMETRY_API_KEY" in result.output
