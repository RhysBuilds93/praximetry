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

PROJECT_EXAMPLES = [
    {"id": "e1", "stage": "plan_action", "input": "track my order"},
    {"id": "e2", "stage": "confirm_action", "input": "confirm order"},
]


def _stub_app(
    score_by_example: dict, *, examples: list | None = None, eval_config: dict | None = None
) -> FastAPI:
    app = FastAPI()
    corpus_examples = examples if examples is not None else EXAMPLES
    batches: dict[str, list[dict]] = {}
    saved_config: dict = dict(eval_config or {})

    def _score(caps: list[dict]) -> dict:
        scores = [score_by_example.get(c["example_id"], 1.0) for c in caps]
        quality = sum(scores) / len(scores) if scores else 0.0
        return {
            "count": len(caps),
            "quality": quality,
            "pass_rate": quality,
            "cost_usd": 0.01 * len(caps),
        }

    @app.get("/api/eval/corpus")
    def corpus(
        stage: str | None = None,
        project: str | None = None,
        source: str | None = None,
        authorization: str = Header(None),
    ):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        return [
            e
            for e in corpus_examples
            if (not stage or e["stage"] == stage)
            and (not project or e.get("project", "proj") == project)
        ]

    @app.post("/api/eval/captures")
    def captures(body: dict, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        caps = body["captures"]
        experiment_id = body.get("experiment_id") or "generated-experiment-id"
        scored = _score(caps)
        batches.setdefault(experiment_id, {})[body["stage"]] = scored
        return {
            "status": "saved",
            "stage": body["stage"],
            "experiment_id": experiment_id,
            "skipped": [],
            **scored,
        }

    @app.get("/api/eval/results")
    def results(
        stage: str | None = None,
        project: str | None = None,
        experiment_id: str | None = None,
        authorization: str = Header(None),
    ):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        if not stage and not project:
            raise HTTPException(status_code=400, detail="stage or project required")
        exp_id = experiment_id or next(reversed(batches), "generated-experiment-id")
        batch = batches.get(exp_id, {})
        if stage:
            scored = batch.get(
                stage, {"count": 0, "quality": 0.0, "pass_rate": 0.0, "cost_usd": 0.0}
            )
            return {"stage": stage, "experiment_id": exp_id, **scored}
        stages = {
            name: {
                "quality": s["quality"],
                "pass_rate": s["pass_rate"],
                "count": s["count"],
                "experiment_id": exp_id,
            }
            for name, s in batch.items()
        }
        agg = sum(s["quality"] for s in stages.values()) / len(stages) if stages else 0.0
        return {"project": project, "stages": stages, "aggregate_quality": agg}

    @app.get("/api/eval/config")
    def get_config(project: str, stage: str | None = None, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        fail_under = saved_config.get((project, stage), saved_config.get((project, None), 0.8))
        return {"project": project, "stage": stage, "fail_under": fail_under}

    return app


def test_eval_passes_when_quality_meets_fail_under(monkeypatch, fake_llm):
    http = TestClient(_stub_app(score_by_example={"e1": 1.0, "e2": 1.0}))
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )

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

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )

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

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )

    @praximetry.stage("plan_action")
    def plan_action(text):
        if text == "cancel my order":
            return "no llm call for this one"
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert "[SKIP] plan_action/e2" in result.output
    assert "examples=1" in result.output


def test_eval_empty_corpus_is_unusable(monkeypatch):
    http = TestClient(_stub_app(score_by_example={}))
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )

    result = runner.invoke(cli_app, ["eval", "--stage", "no-such-stage"])

    assert result.exit_code == 2, result.output
    assert "No golden examples" in result.output


def test_eval_without_an_api_key_is_unusable(monkeypatch):
    monkeypatch.delenv("PRAXIMETRY_API_KEY", raising=False)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 2, result.output
    assert "PRAXIMETRY_API_KEY" in result.output


def _register_project_stages(fake_llm):
    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    @praximetry.stage("confirm_action")
    def confirm_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)


def test_eval_neither_stage_nor_project_is_unusable(monkeypatch):
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)

    result = runner.invoke(cli_app, ["eval"])

    assert result.exit_code == 2, result.output
    assert "--stage" in result.output and "--project" in result.output


def test_eval_project_alone_gates_on_aggregate_quality(monkeypatch, fake_llm):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 0.5},
            examples=PROJECT_EXAMPLES,
        )
    )
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", "--project", "proj", "--fail-under", "0.5"])

    assert result.exit_code == 0, result.output
    assert "plan_action: quality=1.00" in result.output
    assert "confirm_action: quality=0.50" in result.output
    assert "aggregate quality=0.75" in result.output
    assert "PASS" in result.output


def test_eval_project_alone_fails_below_aggregate_threshold(monkeypatch, fake_llm):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 0.5},
            examples=PROJECT_EXAMPLES,
        )
    )
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", "--project", "proj", "--fail-under", "0.9"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_eval_stage_and_project_together_narrows_to_one_stage(monkeypatch, fake_llm):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 0.5},
            examples=PROJECT_EXAMPLES,
        )
    )
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )
    _register_project_stages(fake_llm)

    result = runner.invoke(
        cli_app, ["eval", "--stage", "plan_action", "--project", "proj", "--fail-under", "0.9"]
    )

    assert result.exit_code == 0, result.output
    assert "quality=1.00" in result.output
    assert "examples=1" in result.output
    assert "PASS" in result.output


def test_eval_explicit_fail_under_overrides_fetched_config(monkeypatch, fake_llm):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 0.5},
            examples=PROJECT_EXAMPLES,
            eval_config={("proj", None): 0.99},
        )
    )
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", "--project", "proj", "--fail-under", "0.5"])

    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_eval_project_without_fail_under_falls_back_to_fetched_config(monkeypatch, fake_llm):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 0.5},
            examples=PROJECT_EXAMPLES,
            eval_config={("proj", None): 0.9},
        )
    )
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", "--project", "proj"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_eval_stage_and_project_without_fail_under_uses_stage_override_config(
    monkeypatch, fake_llm
):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 0.5},
            examples=PROJECT_EXAMPLES,
            eval_config={
                ("proj", None): 0.9,
                (
                    "proj",
                    "confirm_action",
                ): 0.99,
            },
        )
    )
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", "--stage", "confirm_action", "--project", "proj"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_eval_stage_only_without_fail_under_keeps_default_point_nine(monkeypatch, fake_llm):
    """No --project: the pre-existing default (0.9) applies unchanged, and the
    config endpoint is never consulted."""
    http = TestClient(_stub_app(score_by_example={"e1": 0.85, "e2": 0.85}))
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
