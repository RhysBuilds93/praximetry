from praximetry.instrument.adapters import ADAPTERS, OpenAIAdapter


def test_openai_adapter_registered():
    assert isinstance(ADAPTERS["openai"], OpenAIAdapter)


def test_openai_get_messages():
    adapter = OpenAIAdapter()
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    assert adapter.get_messages(kwargs) == [{"role": "user", "content": "hi"}]


def test_openai_parse_response_plain_text():
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    resp = ChatCompletion(
        id="x", created=0, model="gpt-4o", object="chat.completion",
        choices=[Choice(finish_reason="stop", index=0,
                        message=ChatCompletionMessage(role="assistant", content="hello world"))],
        usage=CompletionUsage(prompt_tokens=11, completion_tokens=3, total_tokens=14),
    )
    out = OpenAIAdapter().parse_response(resp, "gpt-4o")
    assert out.output_text == "hello world"
    assert out.reasoning_text == ""
    assert out.tokens_in == 11 and out.tokens_out == 3


def test_openai_parse_response_splits_gpt_oss_reasoning():
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    text = "<reasoning>thinking about it</reasoning>final answer"
    resp = ChatCompletion(
        id="x", created=0, model="openai.gpt-oss-20b-1:0", object="chat.completion",
        choices=[Choice(finish_reason="stop", index=0,
                        message=ChatCompletionMessage(role="assistant", content=text))],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )
    out = OpenAIAdapter().parse_response(resp, "openai.gpt-oss-20b-1:0")
    assert out.output_text == "final answer"
    assert out.reasoning_text == "thinking about it"


def test_openai_parse_response_tool_calls():
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall, Function,
    )

    resp = ChatCompletion(
        id="x", created=0, model="gpt-4o", object="chat.completion",
        choices=[Choice(finish_reason="tool_calls", index=0, message=ChatCompletionMessage(
            role="assistant", content=None,
            tool_calls=[ChatCompletionMessageToolCall(
                id="call_1", type="function",
                function=Function(name="lookup", arguments='{"q": "weather"}'),
            )],
        ))],
        usage=CompletionUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
    )
    out = OpenAIAdapter().parse_response(resp, "gpt-4o")
    assert out.output_text == ""
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "lookup"
    assert out.tool_calls[0].arguments == {"q": "weather"}


def test_openai_accumulate_and_finalize():
    from types import SimpleNamespace as NS

    adapter = OpenAIAdapter()
    state: dict = {}
    chunks = [
        NS(choices=[NS(delta=NS(content="he", tool_calls=None))], usage=None),
        NS(choices=[NS(delta=NS(content="llo", tool_calls=None))], usage=None),
        NS(choices=[], usage=NS(prompt_tokens=4, completion_tokens=2)),
    ]
    for chunk in chunks:
        adapter.accumulate(chunk, state)
    out = adapter.finalize_stream(state)
    assert out.output_text == "hello"
    assert out.tokens_in == 4 and out.tokens_out == 2
