from praximetry.instrument.adapters import ADAPTERS, AnthropicAdapter, GeminiAdapter, OpenAIAdapter


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
        id="x",
        created=0,
        model="gpt-4o",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello world"),
            )
        ],
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
        id="x",
        created=0,
        model="openai.gpt-oss-20b-1:0",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
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
        ChatCompletionMessageToolCall,
        Function,
    )

    resp = ChatCompletion(
        id="x",
        created=0,
        model="gpt-4o",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="lookup", arguments='{"q": "weather"}'),
                        )
                    ],
                ),
            )
        ],
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


def test_anthropic_adapter_registered():
    assert isinstance(ADAPTERS["anthropic"], AnthropicAdapter)


def test_anthropic_parse_response_text_only():
    from anthropic.types import Message, TextBlock, Usage

    msg = Message(
        id="x",
        model="claude-sonnet-5",
        role="assistant",
        type="message",
        stop_reason="end_turn",
        stop_sequence=None,
        content=[TextBlock(type="text", text="hi there")],
        usage=Usage(input_tokens=7, output_tokens=2),
    )
    out = AnthropicAdapter().parse_response(msg, "claude-sonnet-5")
    assert out.output_text == "hi there"
    assert out.reasoning_text == ""


def test_anthropic_parse_response_thinking_block():
    from anthropic.types import Message, TextBlock, ThinkingBlock, Usage

    msg = Message(
        id="x",
        model="claude-sonnet-5",
        role="assistant",
        type="message",
        stop_reason="end_turn",
        stop_sequence=None,
        content=[
            ThinkingBlock(type="thinking", thinking="let me work through this", signature="sig"),
            TextBlock(type="text", text="the answer is 4"),
        ],
        usage=Usage(input_tokens=7, output_tokens=9),
    )
    out = AnthropicAdapter().parse_response(msg, "claude-sonnet-5")
    assert out.output_text == "the answer is 4"
    assert out.reasoning_text == "let me work through this"


def test_anthropic_parse_response_tool_use():
    from anthropic.types import Message, ToolUseBlock, Usage

    msg = Message(
        id="x",
        model="claude-sonnet-5",
        role="assistant",
        type="message",
        stop_reason="tool_use",
        stop_sequence=None,
        content=[ToolUseBlock(type="tool_use", id="tu_1", name="lookup", input={"q": "weather"})],
        usage=Usage(input_tokens=7, output_tokens=5),
    )
    out = AnthropicAdapter().parse_response(msg, "claude-sonnet-5")
    assert out.output_text == ""
    assert out.tool_calls[0].name == "lookup"
    assert out.tool_calls[0].arguments == {"q": "weather"}


def test_gemini_adapter_registered():
    assert isinstance(ADAPTERS["gemini"], GeminiAdapter)


def test_gemini_parse_response_text():
    from google.genai import types

    resp = types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=4
        ),
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text="answer")]))
        ],
    )
    out = GeminiAdapter().parse_response(resp, "gemini-2.0-flash")
    assert out.output_text == "answer"
    assert out.tokens_in == 10 and out.tokens_out == 4


def test_gemini_parse_response_function_call():
    from google.genai import types

    resp = types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=6, candidates_token_count=3
        ),
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(name="lookup", args={"q": "weather"})
                        )
                    ],
                )
            )
        ],
    )
    out = GeminiAdapter().parse_response(resp, "gemini-2.0-flash")
    assert out.tool_calls[0].name == "lookup"
    assert out.tool_calls[0].arguments == {"q": "weather"}


def test_gemini_parse_response_separates_thought_parts():
    from google.genai import types

    resp = types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=4
        ),
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="let me think", thought=True),
                        types.Part(text="the answer"),
                    ],
                )
            )
        ],
    )
    out = GeminiAdapter().parse_response(resp, "gemini-2.5-flash")
    assert out.output_text == "the answer"
    assert out.reasoning_text == "let me think"
