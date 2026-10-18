"""`praximetry eval` CLI command, end to end against a stub hosted API.

Corpus fetch and capture push go through a small stub FastAPI app (the real
hosted server lives in the closed-source cloud repo and does real scoring —
out of scope here). This proves the CLI's capture -> push -> gate wiring.
"""

import pytest
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
    score_by_example: dict,
    *,
    examples: list | None = None,
    eval_config: dict | None = None,
    eval_default: dict | None = None,
    eval_scorers: dict | None = None,
    response_by_stage: dict[str, dict] | None = None,
) -> FastAPI:
    app = FastAPI()
    corpus_examples = examples if examples is not None else EXAMPLES
    batches: dict[str, list[dict]] = {}
    saved_config: dict = dict(eval_config or {})
    saved_default: dict | None = dict(eval_default) if eval_default else None
    saved_scorers: dict = dict(eval_scorers or {})
    captures_seen: list[dict] = []
    app.state.captures_seen = captures_seen

    def _scorers_for(project: str | None, stage: str | None):
        return saved_scorers.get((project, stage), saved_scorers.get((project, None)))

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
        captures_seen.append(body)
        caps = body["captures"]
        experiment_id = body.get("experiment_id") or "generated-experiment-id"
        scored = _score(caps)
        scored.update((response_by_stage or {}).get(body["stage"], {}))
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
        return {
            "project": project,
            "stage": stage,
            "fail_under": fail_under,
            "scorers": _scorers_for(project, stage),
        }

    @app.get("/api/eval/default")
    def get_default(authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        if saved_default is None:
            raise HTTPException(status_code=404, detail="no default eval saved")
        fail_under = saved_config.get(
            (saved_default["project"], saved_default.get("stage")),
            saved_config.get((saved_default["project"], None), 0.8),
        )
        return {
            **saved_default,
            "fail_under": fail_under,
            "scorers": _scorers_for(saved_default["project"], saved_default.get("stage")),
        }

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
            return "no llm call for this one"  # never calls record_call
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


def test_eval_no_args_and_no_default_is_unusable(monkeypatch):
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    http = TestClient(_stub_app(score_by_example={}))
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )

    result = runner.invoke(cli_app, ["eval"])

    assert result.exit_code == 2, result.output
    assert "no default saved" in result.output


def test_eval_no_args_uses_saved_default(monkeypatch, fake_llm):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 1.0},
            eval_default={"project": "proj", "stage": "plan_action"},
            eval_config={("proj", "plan_action"): 0.9},
        )
    )
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

    result = runner.invoke(cli_app, ["eval"])

    assert result.exit_code == 0, result.output
    assert "using default: project=proj stage=plan_action" in result.output
    assert "PASS" in result.output


def test_eval_config_file_supplies_project_and_threshold(monkeypatch, fake_llm, tmp_path):
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

    cfg = tmp_path / "eval.json"
    cfg.write_text('{"project": "proj", "fail_under": 0.5}')

    result = runner.invoke(cli_app, ["eval", "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert "aggregate quality=0.75" in result.output
    assert "PASS" in result.output


def test_eval_config_file_overridden_by_cli_flag(monkeypatch, fake_llm, tmp_path):
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

    cfg = tmp_path / "eval.json"
    cfg.write_text('{"project": "proj", "fail_under": 0.5}')

    result = runner.invoke(cli_app, ["eval", "--config", str(cfg), "--fail-under", "0.9"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


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
            eval_config={("proj", None): 0.99},  # would fail if used
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
            eval_config={("proj", None): 0.9},  # aggregate 0.75 < 0.9 -> FAIL
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
                ): 0.99,  # stage override -> confirm_action (0.5) should FAIL
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

    assert result.exit_code == 1, result.output  # 0.85 < 0.9 default
    assert "FAIL" in result.output


def _patch_client(monkeypatch, http):
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )


def test_eval_scorer_flag_reaches_capture_submission(monkeypatch, fake_llm):
    http = TestClient(_stub_app(score_by_example={"e1": 1.0, "e2": 1.0}))
    _patch_client(monkeypatch, http)

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(
        cli_app,
        ["eval", "--stage", "plan_action", "--scorer", "faithfulness", "--scorer", "toxicity"],
    )

    assert result.exit_code == 0, result.output
    assert http.app.state.captures_seen[0]["scorers"] == ["faithfulness", "toxicity"]


def test_eval_scorer_flag_overrides_config_file(monkeypatch, fake_llm, tmp_path):
    http = TestClient(_stub_app(score_by_example={"e1": 1.0, "e2": 1.0}))
    _patch_client(monkeypatch, http)

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    cfg = tmp_path / "eval.json"
    cfg.write_text('{"stage": "plan_action", "scorers": ["from_config"]}')

    result = runner.invoke(cli_app, ["eval", "--config", str(cfg), "--scorer", "from_flag"])

    assert result.exit_code == 0, result.output
    assert http.app.state.captures_seen[0]["scorers"] == ["from_flag"]


def test_eval_config_file_scorers_used_when_no_flag(monkeypatch, fake_llm, tmp_path):
    http = TestClient(_stub_app(score_by_example={"e1": 1.0, "e2": 1.0}))
    _patch_client(monkeypatch, http)

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    cfg = tmp_path / "eval.json"
    cfg.write_text('{"stage": "plan_action", "scorers": ["from_config"]}')

    result = runner.invoke(cli_app, ["eval", "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert http.app.state.captures_seen[0]["scorers"] == ["from_config"]


def test_eval_config_file_empty_scorers_list_overrides_hosted_default(
    monkeypatch, fake_llm, tmp_path
):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 1.0},
            eval_scorers={("proj", None): ["would_win_without_config"]},
        )
    )
    _patch_client(monkeypatch, http)

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    cfg = tmp_path / "eval.json"
    cfg.write_text('{"stage": "plan_action", "scorers": []}')

    result = runner.invoke(cli_app, ["eval", "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert http.app.state.captures_seen[0]["scorers"] == []


def test_eval_project_run_uses_hosted_config_scorers(monkeypatch, fake_llm):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 0.5},
            examples=PROJECT_EXAMPLES,
            eval_scorers={("proj", None): ["hosted_a", "hosted_b"]},
        )
    )
    _patch_client(monkeypatch, http)
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", "--project", "proj", "--fail-under", "0.5"])

    assert result.exit_code == 0, result.output
    assert all(b["scorers"] == ["hosted_a", "hosted_b"] for b in http.app.state.captures_seen)


def test_eval_no_args_default_run_uses_hosted_default_scorers(monkeypatch, fake_llm):
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 1.0, "e2": 1.0},
            eval_default={"project": "proj", "stage": "plan_action"},
            eval_scorers={("proj", "plan_action"): ["hosted_default_scorer"]},
        )
    )
    _patch_client(monkeypatch, http)

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval"])

    assert result.exit_code == 0, result.output
    assert http.app.state.captures_seen[0]["scorers"] == ["hosted_default_scorer"]


def test_eval_no_scorer_anywhere_omits_the_field(monkeypatch, fake_llm):
    http = TestClient(_stub_app(score_by_example={"e1": 1.0, "e2": 1.0}))
    _patch_client(monkeypatch, http)

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 0, result.output
    assert "scorers" not in http.app.state.captures_seen[0]


def _use_stub(monkeypatch: pytest.MonkeyPatch, http: TestClient) -> None:
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setattr(
        hosted_mod,
        "client_from_env",
        lambda client=None: hosted_mod.CloudClient("", VALID_KEY, client=http),
    )


def test_eval_skips_and_counts_stage_level_examples(monkeypatch, fake_llm):
    examples = [
        {"id": "e1", "stage": "plan_action", "input": "track my order"},
        {"id": "e2", "stage": "plan_action", "input": "cancel my order", "level": "stage"},
        {"id": "e3", "stage": "plan_action", "input": "refund my order", "level": "stage"},
    ]
    stub = _stub_app(score_by_example={}, examples=examples)
    _use_stub(monkeypatch, TestClient(stub))

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 0, result.output
    assert "skipped 2 stage-level example(s)" in result.output
    assert "examples=1" in result.output
    pushed = [c["example_id"] for body in stub.state.captures_seen for c in body["captures"]]
    assert pushed == ["e1"]


def test_eval_all_stage_level_corpus_is_unusable(monkeypatch):
    examples = [{"id": "e1", "stage": "plan_action", "input": "x", "level": "stage"}]
    stub = _stub_app(score_by_example={}, examples=examples)
    _use_stub(monkeypatch, TestClient(stub))

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action", "--fail-under", "0"])

    assert result.exit_code == 2, result.output
    assert "skipped 1 stage-level example(s)" in result.output
    assert "nothing to evaluate" in result.output
    assert stub.state.captures_seen == []


def test_eval_unreachable_api_is_unusable(monkeypatch):
    import httpx
    import praximetry.eval.hosted as hosted_mod

    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)

    def unreachable(client=None):
        request = httpx.Request("GET", "https://example.invalid")
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr(hosted_mod, "client_from_env", unreachable)
    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 2, result.output
    assert "offline" in result.output


def test_eval_bad_module_is_unusable(monkeypatch):
    http = TestClient(_stub_app(score_by_example={}))
    _use_stub(monkeypatch, http)

    result = runner.invoke(
        cli_app, ["eval", "--stage", "plan_action", "--module", "missing_eval_test_module"]
    )

    assert result.exit_code == 2, result.output
    assert "missing_eval_test_module" in result.output
    assert http.app.state.captures_seen == []


def test_eval_stage_exception_is_unusable(monkeypatch):
    examples = [{"id": "e1", "stage": "failing_stage", "input": "x"}]
    http = TestClient(_stub_app(score_by_example={}, examples=examples))
    _use_stub(monkeypatch, http)

    @praximetry.stage("failing_stage")
    def failing_stage(text):
        raise ValueError("stage failed")

    result = runner.invoke(cli_app, ["eval", "--stage", "failing_stage"])

    assert result.exit_code == 2, result.output
    assert "stage failed" in result.output
    assert http.app.state.captures_seen == []


def test_eval_non_numeric_threshold_from_config_is_unusable(monkeypatch, fake_llm, tmp_path):
    http = TestClient(_stub_app(score_by_example={"e1": 1.0, "e2": 1.0}))
    _use_stub(monkeypatch, http)
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        '{"stage": "plan_action", "fail_under": "not-a-number"}',
        encoding="utf-8",
    )

    @praximetry.stage("plan_action")
    def plan_action(text):
        return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}], expected_key=text)

    result = runner.invoke(cli_app, ["eval", "--config", str(config_path)])

    assert result.exit_code == 2, result.output
    assert "fail_under must be a number" in result.output


def _regression(*, regressed: bool = False, example_id: str = "e1") -> dict:
    return {
        "baseline": {
            "experiment_id": "baseline-exp",
            "git_branch": "main",
            "git_commit": "baseline-sha",
            "created_at": "2026-10-01T00:00:00Z",
        },
        "overlap": 2,
        "added": ["e3"],
        "removed": ["e0"],
        "baseline_quality": 0.95,
        "quality": 0.975,
        "aggregate_delta": 0.025,
        "regressed": (
            [{"example_id": example_id, "baseline_score": 1.0, "score": 0.95}]
            if regressed
            else []
        ),
        "tolerance": 0.02,
    }


@pytest.mark.parametrize(
    "regression,fail_on_regression,exit_code",
    [
        (None, False, 0),
        (None, True, 0),
        (_regression(), False, 0),
        (_regression(), True, 0),
        (_regression(regressed=True), False, 0),
        (_regression(regressed=True), True, 1),
    ],
)
def test_eval_reports_regressions_and_only_gates_when_opted_in(
    monkeypatch, fake_llm, regression, fail_on_regression, exit_code
):
    response = {"regression": regression} if regression is not None else {}
    http = TestClient(
        _stub_app(
            score_by_example={"e1": 0.95, "e2": 1.0},
            response_by_stage={"plan_action": response},
        )
    )
    _patch_client(monkeypatch, http)
    _register_project_stages(fake_llm)
    args = ["eval", "--stage", "plan_action", "--fail-under", "0.9"]
    if fail_on_regression:
        args.append("--fail-on-regression")

    result = runner.invoke(cli_app, args)

    assert result.exit_code == exit_code, result.output
    if regression is not None:
        assert "regression plan_action:" in result.output
        assert "baseline=baseline-exp" in result.output
        assert "overlap=2" in result.output
        assert "added=1 removed=1" in result.output
        assert "delta=+0.025" in result.output
    elif fail_on_regression:
        assert "regression plan_action: no baseline" in result.output
    if regression and regression["regressed"]:
        assert "WARNING" in result.output
        assert "plan_action: 1 example(s) regressed" in result.output
        assert "e1 (1.000 -> 0.950)" in result.output
    else:
        assert "WARNING" not in result.output
    if exit_code == 0:
        assert "PASS" in result.output
    else:
        assert "FAIL" in result.output
        assert "--fail-on-regression" in result.output
        assert "PASS" not in result.output


@pytest.mark.parametrize("scope", [["--stage", "plan_action"], ["--project", "proj"]])
@pytest.mark.parametrize("fail_on_regression", [False, True])
@pytest.mark.parametrize("quality,exit_code", [(1.0, 0), (0.5, 1)])
def test_eval_without_comparable_examples_only_gates_current_quality(
    monkeypatch, fake_llm, scope, fail_on_regression, quality, exit_code
):
    regression = {
        **_regression(),
        "overlap": 0,
        "added": ["e1", "e2"],
        "removed": ["old-example"],
        "baseline_quality": None,
        "quality": None,
        "aggregate_delta": None,
        "regressed": [],
    }
    http = TestClient(
        _stub_app(
            score_by_example={"e1": quality, "e2": quality},
            response_by_stage={"plan_action": {"regression": regression}},
        )
    )
    _patch_client(monkeypatch, http)
    _register_project_stages(fake_llm)
    args = ["eval", *scope, "--fail-under", "0.9"]
    if fail_on_regression:
        args.append("--fail-on-regression")

    result = runner.invoke(cli_app, args)

    assert result.exit_code == exit_code, result.output
    assert "baseline=baseline-exp" in result.output
    assert "overlap=0" in result.output
    assert "added=2 removed=1" in result.output
    assert "quality=n/a->n/a" in result.output
    assert "delta=n/a regressed=0" in result.output
    assert f"quality={quality:.2f}" in result.output
    assert "WARNING" not in result.output
    if exit_code == 0:
        assert "PASS" in result.output
    else:
        assert "FAIL: quality" in result.output


@pytest.mark.parametrize("regressed_stage", [None, "plan_action", "confirm_action"])
@pytest.mark.parametrize("fail_on_regression", [False, True])
def test_eval_project_aggregates_regressions_from_any_stage(
    monkeypatch, fake_llm, regressed_stage, fail_on_regression
):
    http = TestClient(
        _stub_app(
            score_by_example={},
            examples=PROJECT_EXAMPLES,
            response_by_stage=(
                {
                    regressed_stage: {
                        "regression": _regression(
                            regressed=True,
                            example_id="e1" if regressed_stage == "plan_action" else "e2",
                        )
                    }
                }
                if regressed_stage is not None
                else {}
            ),
        )
    )
    _patch_client(monkeypatch, http)
    _register_project_stages(fake_llm)
    args = ["eval", "--project", "proj", "--fail-under", "0.9"]
    if fail_on_regression:
        args.append("--fail-on-regression")

    result = runner.invoke(cli_app, args)

    assert result.exit_code == (1 if fail_on_regression and regressed_stage else 0), result.output
    assert "aggregate quality=1.00" in result.output
    if regressed_stage is not None:
        assert f"regression {regressed_stage}:" in result.output
        assert f"WARNING: {regressed_stage}: 1 example(s) regressed" in result.output
    else:
        assert "WARNING" not in result.output
        if fail_on_regression:
            assert "regression plan_action: no baseline" in result.output
            assert "regression confirm_action: no baseline" in result.output


@pytest.mark.parametrize("scope", [["--stage", "plan_action"], ["--project", "proj"]])
def test_eval_missing_baseline_does_not_bypass_quality_threshold(monkeypatch, fake_llm, scope):
    http = TestClient(
        _stub_app(score_by_example={"e1": 0.0, "e2": 0.0}, examples=PROJECT_EXAMPLES)
    )
    _patch_client(monkeypatch, http)
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", *scope, "--fail-on-regression"])

    assert result.exit_code == 1, result.output
    assert "no baseline" in result.output
    assert "FAIL: quality" in result.output


@pytest.mark.parametrize("scope", [["--stage", "plan_action"], ["--project", "proj"]])
def test_eval_regression_option_preserves_unusable_exit_code(monkeypatch, scope):
    http = TestClient(_stub_app(score_by_example={}, examples=[]))
    _patch_client(monkeypatch, http)

    result = runner.invoke(cli_app, ["eval", *scope, "--fail-on-regression"])

    assert result.exit_code == 2, result.output
    assert "No golden examples" in result.output


@pytest.mark.parametrize("scope", [["--stage", "plan_action"], ["--project", "proj"]])
@pytest.mark.parametrize("unscored", [0, 2])
def test_eval_reports_nonzero_unscored_counts_per_stage(
    monkeypatch, fake_llm, scope, unscored
):
    http = TestClient(
        _stub_app(
            score_by_example={},
            examples=PROJECT_EXAMPLES,
            response_by_stage={
                "plan_action": {"unscored": unscored},
                "confirm_action": {"unscored": 0},
            },
        )
    )
    _patch_client(monkeypatch, http)
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", *scope])

    assert result.exit_code == 0, result.output
    if unscored:
        assert f"unscored in plan_action: {unscored}" in result.output
    else:
        assert "unscored" not in result.output
    assert "unscored in confirm_action" not in result.output


@pytest.mark.parametrize("override", [False, True])
def test_eval_sends_ci_git_metadata_and_explicit_overrides_for_every_stage(
    monkeypatch, fake_llm, tmp_path, override
):
    monkeypatch.chdir(tmp_path)
    event = tmp_path / "github-event.json"
    event.write_text('{"repository": {"default_branch": "main"}}')
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/pr")
    monkeypatch.setenv("GITHUB_REF_NAME", "123/merge")
    monkeypatch.setenv("GITHUB_SHA", "ci-sha")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    http = TestClient(_stub_app(score_by_example={}, examples=PROJECT_EXAMPLES))
    _patch_client(monkeypatch, http)
    _register_project_stages(fake_llm)
    args = ["eval", "--project", "proj"]
    expected = {
        "git_branch": "feature/pr",
        "git_commit": "ci-sha",
        "default_branch": "main",
    }
    if override:
        args += ["--branch", "manual", "--commit", "manual-sha", "--default-branch", "trunk"]
        expected = {
            "git_branch": "manual",
            "git_commit": "manual-sha",
            "default_branch": "trunk",
        }

    result = runner.invoke(cli_app, args)

    assert result.exit_code == 0, result.output
    assert len(http.app.state.captures_seen) == 2
    for body in http.app.state.captures_seen:
        assert {key: body[key] for key in expected} == expected
    assert len({body["experiment_id"] for body in http.app.state.captures_seen}) == 1


def test_eval_without_git_still_evaluates_and_omits_unknown_metadata(
    monkeypatch, fake_llm, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    for name in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME", "GITHUB_SHA", "GITHUB_EVENT_PATH"):
        monkeypatch.delenv(name, raising=False)
    http = TestClient(_stub_app(score_by_example={}))
    _patch_client(monkeypatch, http)
    _register_project_stages(fake_llm)

    result = runner.invoke(cli_app, ["eval", "--stage", "plan_action"])

    assert result.exit_code == 0, result.output
    body = http.app.state.captures_seen[0]
    assert "git_branch" not in body
    assert "git_commit" not in body
    assert "default_branch" not in body


@pytest.mark.parametrize("contents", [None, "{not-json}"])
def test_eval_config_read_errors_are_actionable_and_unusable(monkeypatch, tmp_path, contents):
    monkeypatch.delenv("PRAXIMETRY_API_KEY", raising=False)
    config_path = tmp_path / "[broken].json"
    if contents is not None:
        config_path.write_text(contents)

    result = runner.invoke(cli_app, ["eval", "--config", str(config_path)])

    assert result.exit_code == 2, result.output
    assert "error:" in result.output
    assert "[broken].json" in result.output
    assert ("could not read --config" if contents is None else "not valid JSON") in result.output
