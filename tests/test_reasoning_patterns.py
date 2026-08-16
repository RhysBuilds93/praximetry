from praximetry.instrument.reasoning_patterns import split_embedded_reasoning


def test_strips_gpt_oss_reasoning_prefix():
    text = "<reasoning>because the message asks for a color, answer red</reasoning>red"
    output, reasoning = split_embedded_reasoning(text, "openai.gpt-oss-20b-1:0")
    assert output == "red"
    assert reasoning == "because the message asks for a color, answer red"


def test_no_match_is_noop():
    output, reasoning = split_embedded_reasoning("plain answer", "openai.gpt-oss-20b-1:0")
    assert output == "plain answer"
    assert reasoning == ""


def test_unknown_model_is_noop():
    text = "<reasoning>trace</reasoning>answer"
    output, reasoning = split_embedded_reasoning(text, "gpt-4o")
    assert output == text
    assert reasoning == ""


def test_multiline_reasoning_block():
    text = "<reasoning>\nstep one\nstep two\n</reasoning>\nfinal answer"
    output, reasoning = split_embedded_reasoning(text, "openai.gpt-oss-120b-1:0")
    assert output == "final answer"
    assert "step one" in reasoning and "step two" in reasoning
