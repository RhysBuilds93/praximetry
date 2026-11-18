"""Exact, project-scoped SDK overrides persisted by the CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import get_config


def project_overrides(project: str) -> dict[str, Any]:
    """Read exact scoped entries without interpreting legacy global stage keys."""
    try:
        policy = json.loads(Path(".praximetry/overrides.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(policy, dict):
        return {}
    projects = policy.get("projects", {})
    if not isinstance(projects, dict):
        return {}
    targets = projects.get(project, {})
    return targets if isinstance(targets, dict) else {}


def load_override(target: str) -> dict[str, Any] | None:
    """Read projects[project][target]; legacy global stages are never applied."""
    override = project_overrides(get_config().project).get(target)
    return override if isinstance(override, dict) else None


def validate_override(override: dict[str, Any]) -> None:
    """Validate the executable subset of hosted winning policies."""
    transforms = override.get("transforms") or []
    if not isinstance(transforms, list) or any(t != "llm_rewrite" for t in transforms):
        raise ValueError(
            "Unsupported stored prompt transforms; only materialized llm_rewrite is supported."
        )
    if transforms or override.get("rewritten_system_prompt") is not None:
        if not isinstance(override.get("rewritten_system_prompt"), str):
            raise ValueError("llm_rewrite requires a materialized rewritten_system_prompt.")


def apply_prompt_override(
    kwargs: dict[str, Any], messages_key: str, override: dict[str, Any]
) -> None:
    """Replace system text only when the optional SHA-256 original-text guard matches."""
    replacement = override.get("rewritten_system_prompt")
    if replacement is None:
        return
    if messages_key == "contents":
        config = kwargs.get("config")
        if config is None:
            config = {}
        if isinstance(config, dict):
            original = config.get("system_instruction", "")
        elif callable(getattr(config, "model_copy", None)):
            original = getattr(config, "system_instruction", "") or ""
        else:
            raise ValueError("Unsupported Gemini config for stored prompt rewrite.")
        expected = override.get("original_system_prompt_hash")
        if expected and (
            not isinstance(original, str)
            or hashlib.sha256(original.encode()).hexdigest() != expected
        ):
            return
        kwargs["config"] = (
            {**config, "system_instruction": replacement}
            if isinstance(config, dict)
            else config.model_copy(update={"system_instruction": replacement})
        )
        return
    messages = kwargs.get(messages_key)
    if isinstance(kwargs.get("system"), str):
        original = kwargs["system"]
        index = None
    else:
        index = next(
            (
                i
                for i, m in enumerate(messages or [])
                if isinstance(m, dict) and m.get("role") == "system"
            ),
            None,
        )
        original = messages[index].get("content") if index is not None else ""
    expected = override.get("original_system_prompt_hash")
    if expected and (
        not isinstance(original, str)
        or hashlib.sha256(original.encode()).hexdigest() != expected
    ):
        return
    if isinstance(kwargs.get("system"), str):
        kwargs["system"] = replacement
    elif isinstance(messages, list):
        updated = list(messages)
        if index is None:
            updated.insert(0, {"role": "system", "content": replacement})
        else:
            updated[index] = {**messages[index], "content": replacement}
        kwargs[messages_key] = updated
