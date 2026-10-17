import asyncio
from types import SimpleNamespace

import pytest

from praximetry.instrument import patch
from praximetry.instrument.adapters import ADAPTERS
from praximetry.instrument.providers import PatchTarget, ProviderSpec
from praximetry.store import get_store


def test_missing_required_target_leaves_every_sdk_method_unchanged(monkeypatch):
    def create(self, **kwargs):
        return "original"

    sync = type("Sync", (), {"create": create})
    asynchronous = type("Async", (), {})
    spec = ProviderSpec(
        "missing-target", lambda: (sync, asynchronous),
        [PatchTarget("create"), PatchTarget("create", is_async=True)], ADAPTERS["openai"],
    )
    monkeypatch.setattr(patch, "_patched", set())

    for _ in range(2):
        with pytest.raises(AttributeError):
            patch._patch(spec)
        assert sync.create is create
        assert sync().create(model="gpt-4o", messages=[]) == "original"
    assert get_store().calls() == []
    assert spec.name not in patch._patched


def test_failed_assignment_rolls_back_prior_targets_and_allows_retry(monkeypatch):
    def original(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    class Host(SimpleNamespace):
        reject_patch = True

        def __setattr__(self, name, value):
            if name == "second" and self.reject_patch and value is not original:
                raise RuntimeError("read-only target")
            super().__setattr__(name, value)

    host = Host(first=original, second=original)
    spec = ProviderSpec(
        "assignment-failure", lambda: (host, host),
        [PatchTarget("first", self_less=True), PatchTarget("second", self_less=True)],
        ADAPTERS["openai"],
    )
    monkeypatch.setattr(patch, "_patched", set())

    with pytest.raises(RuntimeError, match="read-only target"):
        patch._patch(spec)
    assert host.first is original
    assert host.second is original
    assert spec.name not in patch._patched

    host.reject_patch = False
    assert patch._patch(spec)
    response = host.first(model="gpt-4o", messages=[])
    assert response.choices[0].message.content == "answer"
    [call] = get_store().calls()
    assert call.output_text == "answer"
    assert (call.input_tokens, call.output_tokens) == (3, 2)


@pytest.mark.parametrize("self_less", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
def test_retry_after_provider_state_reset_records_each_completion_once(
    monkeypatch, self_less, is_async
):
    def response():
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="one answer"))],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3),
        )

    async def async_create(*args, **kwargs):
        return response()

    if self_less:
        host = SimpleNamespace(create=async_create if is_async else lambda **kwargs: response())
    else:
        host = type(
            "Client", (), {"create": async_create if is_async else lambda self, **kwargs: response()}
        )
    spec = ProviderSpec(
        "retry-once", lambda: (host, host),
        [PatchTarget("create", self_less=self_less, is_async=is_async)], ADAPTERS["openai"],
    )
    monkeypatch.setattr(patch, "_patched", set())
    assert patch._patch(spec)
    patch._patched.clear()
    assert patch._patch(spec)
    assert patch._patch(spec)

    client = host if self_less else host()
    result = client.create(model="gpt-4o", messages=[])
    if is_async:
        asyncio.run(result)
    [call] = get_store().calls()
    assert call.output_text == "one answer"
    assert (call.input_tokens, call.output_tokens) == (4, 3)


def test_a_self_less_module_function_still_records_positional_model_and_messages(monkeypatch):
    def completion(model, messages):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="positional"))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )

    host = SimpleNamespace(completion=completion)
    spec = ProviderSpec(
        "positional", lambda: (host, host),
        [PatchTarget("completion", self_less=True)], ADAPTERS["openai"],
    )
    monkeypatch.setattr(patch, "_patched", set())
    assert patch._patch(spec)

    host.completion("gpt-4o", [{"role": "user", "content": "hi"}])

    [call] = get_store().calls()
    assert call.model == "gpt-4o"
    assert call.messages == [{"role": "user", "content": "hi"}]
    assert call.output_text == "positional"
