import asyncio
import sqlite3

import praximetry as px
from praximetry import runtime
from praximetry.models import Call, Run
from praximetry.otel import map_span
from praximetry.store import Store, get_store


def test_distinct_sources_and_shared_helper(fake_llm):
    def first():
        return runtime.record_call(model="m", messages=[])

    def second():
        return runtime.record_call(model="m", messages=[])

    assert first().target != second().target
    fake_llm.chat("m", [{"role": "user", "content": "one"}])
    fake_llm.chat("m", [{"role": "user", "content": "two"}])
    calls = get_store().calls()
    helper_calls = [c for c in calls if c.provider == "fake"]
    assert len({c.derived_label for c in helper_calls}) == 1
    assert all("FakeLLM.chat" in c.derived_label for c in helper_calls)


def test_nested_labels_restore_and_do_not_register():
    async def worker(name):
        with px.label(name):
            await asyncio.sleep(0)
            assert runtime.effective_stage() == name
            assert runtime.current_stage() == "outer>" + name
            return px.record_call(model="m")

    async def run():
        with px.label("outer"):
            calls = await asyncio.gather(worker("a"), worker("b"))
            assert runtime.effective_stage() == "outer"
        assert runtime.effective_stage() is None
        return calls

    calls = asyncio.run(run())
    assert [c.stage for c in calls] == ["outer>a", "outer>b"]
    assert [c.target for c in calls] == ["a", "b"]
    assert all(c.derived_label.startswith("sdk:") for c in calls)
    assert runtime.STAGE_REGISTRY == {}


def test_tools_and_derived_label_persist_and_migrate(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE calls (id TEXT PRIMARY KEY, run_id TEXT, stage TEXT)")
        conn.execute("INSERT INTO calls VALUES ('old', 'r', 'legacy')")
        conn.commit()
    finally:
        conn.close()
    store = Store(path)
    try:
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        call = Call(
            run_id="r",
            stage="chosen",
            target="scope",
            derived_label="sdk:identity",
            tools=tools,
        )
        store.save_call(call)
        loaded = {c.id: c for c in store.calls()}
        assert loaded[call.id].tools == tools
        assert loaded[call.id].derived_label == "sdk:identity"
        assert loaded[call.id].target == "scope"
        assert loaded["old"].derived_label is None
        assert loaded["old"].target is None
        assert loaded["old"].tools == []
    finally:
        store.close()


def test_otel_identity_uses_evidence():
    attrs = {"gen_ai.request.model": "m", "gen_ai.prompt": "instruction", "gen_ai.operation.name": "chat"}
    first = map_span("framework-node", attrs)
    second = map_span("other-node", attrs)
    assert first.derived_label == second.derived_label
    assert first.stage == "chat"
    assert first.derived_label.startswith("otel:chat:")
    assert "test_labeling" not in first.derived_label
    with px.label("explicit"):
        labeled = map_span("node", attrs)
    assert labeled.stage == "chat"
    assert labeled.target == "explicit"
    assert labeled.derived_label == first.derived_label

def test_project_scoped_aggregates_keep_same_named_labels_apart():
    store = get_store()
    for project, cost in (("one", 1.0), ("two", 2.0)):
        run = Run(project=project)
        store.save_run(run)
        store.save_call(Call(run_id=run.id, stage="same", cost_usd=cost, input_tokens=1))
    assert store.stage_summary()[0]["n"] == 2
    assert len(store.stage_summary(project="one")) == 1
    assert store.stage_summary(project="one")[0]["cost"] == 1.0
    assert store.totals(project="two")["cost"] == 2.0
    assert store.totals()["cost"] == 3.0


def test_otel_user_only_prompts_do_not_invent_instructions():
    attrs = {"gen_ai.request.model": "m", "gen_ai.operation.name": "chat"}
    first = map_span("node", {**attrs, "gen_ai.prompt": "user question one"})
    second = map_span("node", {**attrs, "gen_ai.prompt": "user question two"})
    assert first.derived_label == second.derived_label
    assert first.messages == [{"role": "prompt", "content": "user question one"}]
    structured = map_span("node", {**attrs, "gen_ai.prompt": '[{"role":"user","content":"hello"}]'})
    assert structured.messages == [{"role": "user", "content": "hello"}]
    assert structured.derived_label == first.derived_label


def test_langchain_flattened_callbacks_are_not_replay_complete():
    from uuid import uuid4
    from langchain_core.outputs import Generation, LLMResult
    from praximetry.instrument.langchain import LangChainCallbackCapture

    capture = LangChainCallbackCapture()
    run_id = uuid4()
    capture.on_llm_start({}, ["flattened prompt"], run_id=run_id)
    capture.on_llm_end(LLMResult(generations=[[Generation(text="ok")]]), run_id=run_id)
    assert get_store().calls()[0].metadata["request_complete"] is False
