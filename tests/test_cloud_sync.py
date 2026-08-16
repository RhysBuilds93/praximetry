"""Background sync of locally-recorded calls to the hosted /api/traces route.

Exercised against a stub FastAPI app + real TestClient (same pattern as
test_eval_hosted.py) rather than mocking httpx — proves the request shape,
auth header, and graceful-degradation behavior cloud_sync actually produces.
"""
import logging

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

import praximetry
from praximetry import cloud_sync, runtime
from praximetry.eval.hosted import CloudClient
from praximetry.models import Call, Run
from praximetry.store import get_store

VALID_KEY = "px_live_stub_key"


def _stub_app() -> FastAPI:
    app = FastAPI()
    received = {"traces": []}

    @app.post("/api/traces")
    def traces(body: dict, authorization: str = Header(None)):
        if authorization != f"Bearer {VALID_KEY}":
            raise HTTPException(status_code=401, detail="bad key")
        received["traces"].append(body)
        return {"ok": True}

    app.state.received = received
    return app


@pytest.fixture
def stub():
    app = _stub_app()
    return app, TestClient(app)


@pytest.fixture(autouse=True)
def _cloud_sync_isolation():
    cloud_sync.reset()
    yield
    cloud_sync.reset()


def _make_run_and_call(stage="plan_action", messages=None):
    run = Run(project="test")
    call = Call(
        run_id=run.id, stage=stage, provider="fake", model="gpt-4o",
        messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
        output_text="ok",
    )
    return run, call


def test_happy_path_pushes_run_and_calls(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)
    cloud_sync.start(client)

    run, call = _make_run_and_call()
    cloud_sync.note_run(run)
    cloud_sync.enqueue(call)
    cloud_sync.flush_now()

    assert len(app.state.received["traces"]) == 1
    body = app.state.received["traces"][0]
    assert body["run"]["id"] == run.id
    assert body["calls"][0]["id"] == call.id
    assert body["calls"][0]["messages"] == [{"role": "user", "content": "hi"}]


def test_default_init_without_api_key_does_not_start_cloud_sync(monkeypatch):
    monkeypatch.delenv("PRAXIMETRY_API_KEY", raising=False)

    praximetry.init(project="test")

    assert cloud_sync.is_running() is False


def test_queue_full_drops_and_logs_without_raising(stub, caplog):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)
    cloud_sync.start(client, maxsize=1)

    run, call1 = _make_run_and_call()
    _, call2 = _make_run_and_call()
    call2.run_id = run.id

    with caplog.at_level(logging.WARNING, logger="praximetry.cloud_sync"):
        cloud_sync.enqueue(call1)
        cloud_sync.enqueue(call2)  # queue full -> dropped, must not raise

    assert any("dropped" in r.message.lower() for r in caplog.records)


def test_network_down_does_not_raise_and_worker_survives():
    client = CloudClient("http://127.0.0.1:1", "some-key")
    cloud_sync.start(client)

    run, call = _make_run_and_call()
    cloud_sync.note_run(run)
    cloud_sync.enqueue(call)

    # Must not raise even though the server is unreachable.
    cloud_sync.flush_now()

    assert cloud_sync.is_running() is True


def test_worker_recovers_after_network_failure(stub):
    dead_client = CloudClient("http://127.0.0.1:1", "some-key")
    cloud_sync.start(dead_client)

    run, call1 = _make_run_and_call()
    cloud_sync.note_run(run)
    cloud_sync.enqueue(call1)
    cloud_sync.flush_now()  # fails silently, dead client

    app, http = stub
    working_client = CloudClient("", VALID_KEY, client=http)
    cloud_sync.start(working_client)  # swap in a working client

    _, call2 = _make_run_and_call()
    call2.run_id = run.id
    cloud_sync.enqueue(call2)
    cloud_sync.flush_now()

    assert len(app.state.received["traces"]) == 1
    assert app.state.received["traces"][0]["calls"][0]["id"] == call2.id


def test_redaction_hook_strips_messages_before_push_but_not_locally(stub, monkeypatch):
    app, http = stub
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)
    client = CloudClient("", VALID_KEY, client=http)
    cloud_sync.start(client)

    def strip_messages(call: Call) -> Call:
        return call.model_copy(update={"messages": []})

    cloud_sync.set_redaction_hook(strip_messages)

    run = runtime.current_run()
    runtime.record_call(
        provider="fake", model="gpt-4o",
        messages=[{"role": "user", "content": "secret"}],
        output_text="ok",
    )
    cloud_sync.note_run(run)
    cloud_sync.flush_now()

    body = app.state.received["traces"][0]
    assert body["calls"][0]["messages"] == []

    stored = get_store().calls(run_id=run.id)
    assert stored[0].messages == [{"role": "user", "content": "secret"}]


def test_record_call_enqueues_when_cloud_sync_running(stub):
    app, http = stub
    client = CloudClient("", VALID_KEY, client=http)
    cloud_sync.start(client)

    run = runtime.current_run()
    cloud_sync.note_run(run)
    runtime.record_call(
        provider="fake", model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        output_text="ok",
    )
    cloud_sync.flush_now()

    assert len(app.state.received["traces"]) == 1


def test_init_cloud_true_without_api_key_raises(monkeypatch):
    monkeypatch.delenv("PRAXIMETRY_API_KEY", raising=False)

    from praximetry.eval.hosted import CloudError

    with pytest.raises(CloudError):
        praximetry.init(project="test", cloud=True)


def test_init_cloud_false_never_starts_even_with_api_key(monkeypatch):
    monkeypatch.setenv("PRAXIMETRY_API_KEY", VALID_KEY)

    praximetry.init(project="test", cloud=False)

    assert cloud_sync.is_running() is False
