"""CloudClient: the thin HTTP wrapper the `eval` CLI command uses to fetch the
hosted corpus and push captured request shapes. Exercised against a small
stub FastAPI app rather than the real hosted server (that lives in the
closed-source cloud repo) — this proves the request shape, auth header, and
error surfacing CloudClient produces.
"""
import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from praximetry.eval.capture import CapturedRequest
from praximetry.eval.hosted import CloudClient, CloudError

VALID_KEY = "px_live_stub_key"


def _stub_app() -> FastAPI:
    app = FastAPI()
    received = {}

    @app.post("/api/eval/captures")
    def captures(body: dict, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        received["body"] = body
        return {"count": len(body["captures"]), "stage": body["stage"],
                "experiment_id": body.get("experiment_id") or "generated-id"}

    @app.get("/api/eval/corpus")
    def corpus(stage: str | None = None, project: str | None = None,
               source: str | None = None, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        received["corpus_stage"] = stage
        received["corpus_project"] = project
        received["corpus_source"] = source
        return [{"id": "e1", "stage": stage or "route", "input": "hi"}]

    @app.get("/api/eval/results")
    def results(stage: str | None = None, project: str | None = None,
                experiment_id: str | None = None, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        if not stage and not project:
            raise HTTPException(status_code=400, detail="stage or project required")
        received["results_query"] = {"stage": stage, "project": project,
                                      "experiment_id": experiment_id}
        if project:
            return {"project": project,
                    "stages": {"plan_action": {"quality": 0.9, "pass_rate": 0.9,
                                               "count": 2, "experiment_id": experiment_id}},
                    "aggregate_quality": 0.9}
        return {"stage": stage, "experiment_id": experiment_id or "generated-id",
                "count": 2, "quality": 0.9, "pass_rate": 0.9, "cost_usd": 0.02}

    @app.get("/api/eval/config")
    def get_config(project: str, stage: str | None = None, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        received["config_query"] = {"project": project, "stage": stage}
        return {"project": project, "stage": stage, "fail_under": 0.8}

    @app.post("/api/eval/config")
    def post_config(body: dict, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        received["config_body"] = body
        return body

    @app.post("/api/optimize/captures")
    def optimize_captures(body: dict, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        received["optimize_body"] = body
        return {
            "stage": body["stage"], "baseline": {"model": "gpt-4o", "cost_usd": 0.02},
            "trials": 3, "winner": {"model": "amazon.nova-pro-v1:0"},
            "examples": 1, "truncated": False, "errors": [],
        }

    @app.get("/api/optimize/winner")
    def optimize_winner(stage: str, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        received["winner_stage"] = stage
        if stage == "no-run":
            raise HTTPException(status_code=404, detail="no completed run")
        return {
            "stage": stage, "model": "amazon.nova-pro-v1:0", "transforms": ["compact"],
            "experiment_id": "exp1", "savings_pct": 0.34,
        }

    app.state.received = received
    return app


@pytest.fixture
def stub():
    app = _stub_app()
    return app, TestClient(app)


CAPTURES = [
    CapturedRequest(example_id="e1", stage="plan_action", provider="fake", model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[{"name": "get_order", "args": {"order_id": "str"}}]),
    CapturedRequest(example_id="e2", stage="plan_action", provider="fake", model="gpt-4o",
                    messages=[{"role": "user", "content": "bye"}]),
]


def test_push_captures_posts_stage_and_captures(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    result = client.push_captures("plan_action", CAPTURES)

    assert result["count"] == 2
    assert result["stage"] == "plan_action"
    body = app.state.received["body"]
    assert body["stage"] == "plan_action"
    assert len(body["captures"]) == 2
    assert body["captures"][0]["example_id"] == "e1"
    assert body["captures"][0]["tools"] == [{"name": "get_order", "args": {"order_id": "str"}}]
    assert body["captures"][1]["tools"] == []


def test_push_captures_sends_the_bearer_auth_header(stub):
    _app, http = stub
    client = CloudClient("", "wrong_key", client=http)

    with pytest.raises(CloudError, match="401"):
        client.push_captures("plan_action", CAPTURES)


def test_fetch_corpus_returns_a_dataset(stub):
    _app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    dataset = client.fetch_corpus("plan_action")

    assert dataset.examples[0].id == "e1"
    assert dataset.examples[0].stage == "plan_action"


def test_push_optimize_capture_posts_stage_and_capture(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)
    capture = CapturedRequest(example_id="e1", stage="checkout", provider="anthropic",
                              model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}],
                              tools=[])

    result = client.push_optimize_capture(
        "checkout", capture, candidate_models=["amazon.nova-pro-v1:0"],
        transforms=["compact"], quality_tolerance=0.02, max_trials=8,
    )

    assert result["winner"] == {"model": "amazon.nova-pro-v1:0"}
    body = app.state.received["optimize_body"]
    assert body["stage"] == "checkout"
    assert body["captured_request"]["example_id"] == "e1"
    assert body["captured_request"]["model"] == "claude-sonnet-5"
    assert body["candidate_models"] == ["amazon.nova-pro-v1:0"]
    assert body["transforms"] == ["compact"]
    assert body["quality_tolerance"] == 0.02
    assert body["max_trials"] == 8


def test_push_optimize_capture_defaults(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)
    capture = CapturedRequest(example_id="e1", stage="checkout", provider="anthropic",
                              model="claude-sonnet-5")

    client.push_optimize_capture("checkout", capture)

    body = app.state.received["optimize_body"]
    assert body["candidate_models"] == []
    assert body["transforms"] == []
    assert body["quality_tolerance"] == 0.02
    assert body["max_trials"] == 8


def test_fetch_winner_returns_the_winner(stub):
    _app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    winner = client.fetch_winner("checkout")

    assert winner == {
        "stage": "checkout", "model": "amazon.nova-pro-v1:0", "transforms": ["compact"],
        "experiment_id": "exp1", "savings_pct": 0.34,
    }


def test_fetch_winner_returns_none_on_404(stub):
    _app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    assert client.fetch_winner("no-run") is None


def test_fetch_winner_raises_on_401(stub):
    _app, http = stub
    client = CloudClient("", "wrong_key", client=http)

    with pytest.raises(CloudError, match="401"):
        client.fetch_winner("checkout")


def test_fetch_corpus_passes_project_and_source(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    client.fetch_corpus(stage="plan_action", source="expert", project="proj")

    assert app.state.received["corpus_stage"] == "plan_action"
    assert app.state.received["corpus_project"] == "proj"
    assert app.state.received["corpus_source"] == "expert"


def test_push_captures_threads_experiment_id(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    result = client.push_captures("plan_action", CAPTURES, experiment_id="exp-123")

    assert result["experiment_id"] == "exp-123"
    assert app.state.received["body"]["experiment_id"] == "exp-123"


def test_fetch_results_requires_stage_or_project(stub):
    _app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    with pytest.raises(CloudError, match="stage or project"):
        client.fetch_results()


def test_fetch_results_by_stage(stub):
    _app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    result = client.fetch_results(stage="plan_action", experiment_id="exp-123")

    assert result == {"stage": "plan_action", "experiment_id": "exp-123",
                       "count": 2, "quality": 0.9, "pass_rate": 0.9, "cost_usd": 0.02}


def test_fetch_results_by_project(stub):
    _app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    result = client.fetch_results(project="proj", experiment_id="exp-123")

    assert result["aggregate_quality"] == 0.9
    assert "plan_action" in result["stages"]


def test_fetch_eval_config(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    result = client.fetch_eval_config(project="proj", stage="plan_action")

    assert result == {"project": "proj", "stage": "plan_action", "fail_under": 0.8}
    assert app.state.received["config_query"] == {"project": "proj", "stage": "plan_action"}


def test_save_eval_config(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)

    client.save_eval_config(project="proj", fail_under=0.95, stage="plan_action")

    assert app.state.received["config_body"] == {
        "project": "proj", "stage": "plan_action", "fail_under": 0.95,
    }


def test_client_from_env_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("PRAXIMETRY_API_KEY", raising=False)

    from praximetry.eval.hosted import client_from_env

    with pytest.raises(CloudError, match="PRAXIMETRY_API_KEY"):
        client_from_env()
