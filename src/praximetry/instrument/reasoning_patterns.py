"""Registry of known text-embedded reasoning prefixes, keyed by model prefix.

Some providers (e.g. gpt-oss models served through Bedrock's OpenAI-compatible
endpoint) prepend every reply with a visible <reasoning>...</reasoning> block
with no structural separation from the answer. Providers that DO separate
reasoning structurally (Anthropic thinking blocks, OpenAI o1/o3's reasoning
field) never consult this table — their adapters populate reasoning_text
directly during parsing.
"""

from __future__ import annotations

import re

_REASONING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai.gpt-oss", re.compile(r"^\s*<reasoning>(.*?)</reasoning>\s*", re.S)),
]


def split_embedded_reasoning(text: str, model: str) -> tuple[str, str]:
    """Strip a known embedded reasoning block for `model`'s provider, if any.

    Returns:
        (output_text, reasoning_text) — text unchanged and reasoning_text ""
        if no pattern matches `model`.
    """
    for prefix, pattern in _REASONING_PATTERNS:
        if model.startswith(prefix):
            m = pattern.match(text)
            if m:
                return text[m.end() :].lstrip(), m.group(1).strip()
    return text, ""
