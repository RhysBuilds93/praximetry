import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

import praximetry as px
from praximetry.config import get_config
from praximetry.instrument import patch as P
from praximetry.instrument.adapters import ADAPTERS
from praximetry.otel import map_span
from praximetry.overrides import load_override, validate_override
from praximetry.runtime import override_context, set_policy_hook
from praximetry.store import get_store


def write_policy(tmp_path, monkeypatch, policy):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".praximetry").mkdir(exist_ok=True)
    (tmp_path / ".praximetry" / "overrides.json").write_text(json.dumps(policy))


def patched_client():
    seen = []

    def original(self, **kwargs):
        seen.append(kwargs)
        return NS(choices=[NS(message=NS(content="ok"))], usage=NS(prompt_tokens=2, completion_tokens=1))

    return P._instrument(original, "openai", ADAPTERS["openai"], False), seen


def test_exact_project_scoping_and_legacy_non_application(tmp_path, monkeypatch):
    write_policy(tmp_path, monkeypatch, {"stages": {"work": {"model": "legacy"}}, "projects": {"test": {"work": {"model": "new"}}, "other": {"work": {"model": "other"}}}})
    assert load_override("work") == {"model": "new"}
    assert load_override("wor") is None
    get_config().project = "other"
    assert load_override("work") == {"model": "other"}
    get_config().project = "missing"
    assert load_override("work") is None


def test_sdk_selects_original_identity_and_records_outbound_tools(tmp_path, monkeypatch):
    client, seen = patched_client()
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    def call():
        return client(None, model="original", messages=[{"role": "system", "content": "original prompt"}], tools=tools)

    call()
    original = get_store().calls()[0]
    write_policy(tmp_path, monkeypatch, {"projects": {"test": {original.target: {"model": "replacement", "transforms": ["llm_rewrite"], "rewritten_system_prompt": "new prompt"}}}})
    call()
    latest = get_store().calls()[0]
    assert seen[-1]["model"] == "replacement"
    assert seen[-1]["messages"][0]["content"] == "new prompt"
    assert latest.stage == original.stage
    assert latest.derived_label == original.derived_label
    assert latest.tools == tools
    assert latest.tool_calls == []
    assert latest.metadata["request_complete"] is True
    assert latest.target == original.target

    def changed_prompt():
        return client(None, model="original", messages=[{"role": "system", "content": "changed"}])

    changed_prompt()
    assert seen[-1]["model"] == "original"


def test_otel_never_loads_sdk_overrides(monkeypatch):
    def forbidden(*args):
        raise AssertionError("OTel must not load overrides")

    monkeypatch.setattr("praximetry.instrument.patch.load_override", forbidden)
    attrs = {"gen_ai.request.model": "m", "gen_ai.prompt": "instruction"}
    assert map_span("node", attrs).derived_label.startswith("otel:")


def test_inflight_empty_override_and_policy_hook_take_precedence(tmp_path, monkeypatch):
    write_policy(tmp_path, monkeypatch, {"projects": {"test": {"work": {"model": "stored"}}}})
    client, seen = patched_client()
    with px.label("work"), override_context():
        client(None, model="original", messages=[])
    assert seen[-1]["model"] == "original"
    with px.label("work"), override_context(model="inflight"):
        client(None, model="original", messages=[])
    assert seen[-1]["model"] == "inflight"

    @contextmanager
    def hook(name):
        yield

    set_policy_hook(hook)
    try:
        with px.label("work"):
            client(None, model="original", messages=[])
        assert seen[-1]["model"] == "original"
    finally:
        set_policy_hook(None)


@pytest.mark.parametrize("anthropic", [False, True])
@pytest.mark.parametrize("matching", [False, True])
def test_materialized_prompt_hash_guard(tmp_path, monkeypatch, anthropic, matching):
    original = "old"
    guard = hashlib.sha256((original if matching else "different").encode()).hexdigest()
    write_policy(tmp_path, monkeypatch, {"projects": {"test": {"work": {"transforms": ["llm_rewrite"], "rewritten_system_prompt": "new", "original_system_prompt_hash": guard}}}})
    kwargs = {"model": "m", "messages": [] if anthropic else [{"role": "system", "content": original}]}
    if anthropic:
        kwargs["system"] = original
    out = P._apply_overrides(kwargs, "messages", "work")
    prompt = out["system"] if anthropic else out["messages"][0]["content"]
    assert prompt == ("new" if matching else "old")


def test_unknown_transform_rejected_by_validator_but_capture_remains_safe(tmp_path, monkeypatch, caplog):
    write_policy(tmp_path, monkeypatch, {"projects": {"test": {"work": {"transforms": ["compact"]}}}})
    client, seen = patched_client()
    with px.label("work"):
        client(None, model="m", messages=[])
    assert seen[-1]["model"] == "m"
    assert "could not apply stored override" in caplog.text
    with pytest.raises(ValueError, match="Unsupported"):
        validate_override({"transforms": ["compact"]})
    with pytest.raises(ValueError, match="materialized"):
        validate_override({"transforms": ["llm_rewrite"]})


def test_message_iterator_reaches_outbound_intact():
    client, seen = patched_client()
    messages = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    client(None, model="m", messages=iter(messages))
    assert seen[-1]["messages"] == messages
    assert get_store().calls()[0].messages == messages


@pytest.mark.parametrize("typed", [False, True])
def test_gemini_config_tools_are_captured(typed):
    from pydantic import BaseModel

    class Config(BaseModel):
        tools: list[dict]
        system_instruction: str

    tools = [{"function_declarations": [{"name": "lookup"}]}]
    config = {"tools": tools, "system_instruction": "instruction"}
    if typed:
        config = Config(**config)
    seen = []

    def original(self, **kwargs):
        seen.append(kwargs)
        return NS(candidates=[], usage_metadata=None)

    client = P._instrument(original, "gemini", ADAPTERS["gemini"], False, "contents")
    client(None, model="m", contents="question", config=config)
    call = get_store().calls()[0]
    assert call.tools == tools
    assert call.metadata["request_complete"] is True
    assert call.messages[0] == {"role": "system", "content": "instruction"}
    assert seen[-1]["config"] == config


def test_unsupported_gemini_config_is_incomplete():
    client = P._instrument(lambda self, **kwargs: NS(candidates=[], usage_metadata=None),
                           "gemini", ADAPTERS["gemini"], False, "contents")
    client(None, model="m", contents="question", config=NS(tools=[object()]))
    assert get_store().calls()[0].metadata["request_complete"] is False


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("matching", [False, True])
def test_gemini_rewrite_preserves_original_identity_and_config(tmp_path, monkeypatch, typed, matching):
    from pydantic import BaseModel

    class Config(BaseModel):
        system_instruction: str
        tools: list[dict]
        temperature: float

    tools = [{"function_declarations": [{"name": "lookup"}]}]
    initial = {"system_instruction": "original", "tools": tools, "temperature": 0.2}
    config = Config(**initial) if typed else dict(initial)
    seen = []

    def original(self, **kwargs):
        seen.append(kwargs)
        return NS(candidates=[], usage_metadata=None)

    client = P._instrument(original, "gemini", ADAPTERS["gemini"], False, "contents")

    def invoke():
        with px.label("gemini-work"):
            client(None, model="m", contents="question", config=config)

    invoke()
    before = get_store().calls()[0]
    guard = hashlib.sha256(("original" if matching else "different").encode()).hexdigest()
    write_policy(tmp_path, monkeypatch, {"projects": {"test": {"gemini-work": {
        "transforms": ["llm_rewrite"], "rewritten_system_prompt": "rewritten",
        "original_system_prompt_hash": guard,
    }}}})
    invoke()
    after = get_store().calls()[0]
    outbound = seen[-1]["config"]
    outbound_dict = outbound.model_dump() if typed else outbound
    expected = "rewritten" if matching else "original"
    assert outbound_dict == {**initial, "system_instruction": expected}
    assert (config.model_dump() if typed else config) == initial
    assert after.messages[0] == {"role": "system", "content": expected}
    assert after.tools == tools
    assert after.derived_label == before.derived_label
    assert after.target == before.target
    if matching:
        assert outbound is not config
