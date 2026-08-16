from praximetry.instrument.output import ContentPart, NormalizedOutput, ToolCall


def test_normalized_output_defaults():
    out = NormalizedOutput()
    assert out.output_text == ""
    assert out.reasoning_text == ""
    assert out.tool_calls == []
    assert out.structured_output is None
    assert out.content_parts == []
    assert out.tokens_in == 0
    assert out.tokens_out == 0


def test_normalized_output_with_tool_call():
    out = NormalizedOutput(
        output_text="",
        tool_calls=[ToolCall(id="t1", name="lookup", arguments={"q": "x"})],
    )
    assert out.tool_calls[0].name == "lookup"
    assert out.tool_calls[0].arguments == {"q": "x"}


def test_content_part():
    part = ContentPart(type="image", data="base64...")
    assert part.type == "image"
