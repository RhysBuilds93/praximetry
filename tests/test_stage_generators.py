"""@stage on generator functions: stage attribution and policy override hold during iteration."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

import pytest

from praximetry import stage
from praximetry.runtime import current_stage, get_overrides, override_context, set_policy_hook


@pytest.fixture(autouse=True)
def winner_policy() -> Iterator[None]:
    @contextmanager
    def hook(stage_name: str) -> Iterator[None]:
        with override_context(model="winner-model"):
            yield

    set_policy_hook(hook)
    yield
    set_policy_hook(None)


def observe() -> tuple[str | None, str | None]:
    overrides = get_overrides()
    return current_stage(), overrides["model"] if overrides else None


def test_sync_generator_sees_stage_and_override_and_restores():
    seen = []

    @stage("gen_stage")
    def gen() -> Iterator[int]:
        seen.append(observe())
        yield 1
        seen.append(observe())
        yield 2

    for _ in gen():
        assert observe() == (None, None)
    assert seen == [("gen_stage", "winner-model")] * 2
    assert observe() == (None, None)


def test_sync_generator_break_closes_inside_context():
    seen = []

    @stage("gen_stage")
    def gen() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            seen.append(observe())

    for _ in gen():
        break
    assert seen == [("gen_stage", "winner-model")]
    assert observe() == (None, None)


def test_sync_generator_close_before_start_does_not_run_body():
    seen = []

    @stage("gen_stage")
    def gen() -> Iterator[int]:
        seen.append(observe())
        yield 1

    gen().close()
    assert seen == []


def test_async_generator_sees_stage_and_override_and_restores():
    async def scenario() -> None:
        seen = []

        @stage("agen_stage")
        async def agen() -> AsyncIterator[int]:
            seen.append(observe())
            yield 1
            seen.append(observe())
            yield 2

        async for _ in agen():
            assert observe() == (None, None)
        assert seen == [("agen_stage", "winner-model")] * 2
        assert observe() == (None, None)

    asyncio.run(scenario())


def test_async_generator_aclose_runs_cleanup_inside_context():
    async def scenario() -> None:
        seen = []

        @stage("agen_stage")
        async def agen() -> AsyncIterator[int]:
            try:
                yield 1
                yield 2
            finally:
                seen.append(observe())

        gen = agen()
        await gen.__anext__()
        await gen.aclose()
        assert seen == [("agen_stage", "winner-model")]
        assert observe() == (None, None)

    asyncio.run(scenario())


def test_async_generator_break_out_of_loop():
    async def scenario() -> None:
        seen = []

        @stage("agen_stage")
        async def agen() -> AsyncIterator[int]:
            try:
                yield 1
                yield 2
            finally:
                seen.append(observe())

        gen = agen()
        async for _ in gen:
            break
        await gen.aclose()
        assert seen == [("agen_stage", "winner-model")]
        assert observe() == (None, None)

    asyncio.run(scenario())
