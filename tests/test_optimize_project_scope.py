import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import praximetry
from praximetry import config
from praximetry.cli import app as cli_app
from praximetry.eval import hosted


@pytest.mark.parametrize("override", [False, True])
def test_optimize_and_apply_use_the_same_project_for_shared_stage(
    monkeypatch, tmp_path, fake_llm, override
):
    monkeypatch.chdir(tmp_path)
    config.get_config().project = "orders"
    chosen = "support" if override else "orders"
    models = {"orders": "gpt-4o-mini", "support": "amazon.nova-pro-v1:0"}
    api = FastAPI()
    optimized = set()

    @api.get("/api/eval/corpus")
    def corpus(stage: str, project: str | None = None):
        if project not in models:
            raise HTTPException(409, "stage belongs to multiple projects")
        return [{"id": project, "stage": stage, "input": project, "expected": project}]

    @api.post("/api/optimize/captures")
    def optimize(body: dict):
        project = body.get("project")
        if project not in models or body["captured_request"]["example_id"] != project:
            raise HTTPException(409, "capture and project do not match")
        optimized.add(project)
        return {"stage": body["stage"], "examples": 1, "winner": {"model": models[project]}}

    @api.get("/api/optimize/winner")
    def winner(stage: str, project: str | None = None):
        if project not in optimized:
            raise HTTPException(404, "no optimization in this project")
        return {"stage": stage, "model": models[project], "transforms": [], "experiment_id": project}

    with TestClient(api) as http:
        monkeypatch.setattr(hosted, "client_from_env", lambda: hosted.CloudClient("", "test", client=http))

        @praximetry.stage("shared")
        def shared(text):
            return fake_llm.chat("gpt-4o", [{"role": "user", "content": text}])

        project_args = ["--project", chosen] if override else []
        runner = CliRunner()
        result = runner.invoke(cli_app, ["optimize", "--stage", "shared", *project_args])
        assert result.exit_code == 0, result.output
        applied = runner.invoke(cli_app, ["apply", "--stage", "shared", *project_args])
        assert applied.exit_code == 0, applied.output

    policy = json.loads((tmp_path / ".praximetry" / "overrides.json").read_text())
    assert policy["stages"]["shared"]["model"] == models[chosen]
    assert policy["stages"]["shared"]["experiment_id"] == chosen
