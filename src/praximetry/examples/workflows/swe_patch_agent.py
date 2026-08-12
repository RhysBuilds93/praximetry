"""Workflow 6: bug-fixing coding agent, modelled on SWE-bench.

SWE-bench (https://www.swebench.com) gives an agent a repo plus a real issue and
asks for a patch; the patch is scored by *actually running the repo's tests*, not
by comparing it to the reference diff. That's the honest kind of eval, and it
maps cleanly onto praximetry: `run_tests` here really does exec the patched source
and assert on it, so the optimizer's "quality" number is a genuine resolve rate.

Pipeline:
    localize     (LLM)  -> which file the issue is in
    propose_patch(LLM)  -> the replacement implementation
    run_tests    (tool) -> exec patched source, run the tests, PASS/FAIL
    resolve      (task) -> the whole chain, scored exact against "RESOLVED"

Try:
    python -m praximetry.examples.workflows.swe_patch_agent
    praximetry eval --stage localize -m praximetry.examples.workflows.swe_patch_agent --fail-under 0.9
    praximetry optimize --stage localize -m praximetry.examples.workflows.swe_patch_agent
    praximetry apply --stage localize
"""
from __future__ import annotations

import praximetry

from ._real import default_model, premium_model, real_chat

praximetry.init(project="swe-patch")

# A tiny repo with three real bugs.
REPO: dict[str, str] = {
    "stats.py": (
        "def mean(xs):\n"
        "    return sum(xs) / len(xs)\n"
    ),
    "text.py": (
        "def truncate(s, n):\n"
        "    return s[:n] + '...'\n"
    ),
    "dates.py": (
        "def is_leap(y):\n"
        "    return y % 4 == 0\n"
    ),
}

# The repo's own tests. Run for real against the patched source.
TESTS = {
    "stats.py": lambda ns: ns["mean"]([]) == 0.0 and ns["mean"]([2, 4]) == 3.0,
    "text.py": lambda ns: ns["truncate"]("hi", 10) == "hi"
                          and ns["truncate"]("abcdef", 3) == "abc...",
    "dates.py": lambda ns: ns["is_leap"](2000) and not ns["is_leap"](1900)
                           and ns["is_leap"](2024),
}

# Prompt written the way they end up in real repos: a repo map on every call,
# duplicated house rules, and a stack of examples nobody has pruned.
RULES = (
    "You are a senior engineer. Return code only, no prose.\n"
    "Return code only, no prose. Preserve the public signature.\n"
    "Do not add new dependencies. Do not add new dependencies.\n"
    "Match the surrounding style. Return code only, no prose.\n"
)

REPO_MAP = "\n\n".join(f"### {p}\n{src}" for p, src in REPO.items())

FEW_SHOT = "".join(
    f"Example {i}: {issue} -> {path}\n"
    for i, (issue, path) in enumerate([
        ("crash averaging an empty list", "stats.py"),
        ("truncate mangles short strings", "text.py"),
        ("1900 reported as a leap year", "dates.py"),
        ("mean() divides by zero", "stats.py"),
        ("ellipsis added when not needed", "text.py"),
        ("century leap rule ignored", "dates.py"),
    ], start=1)
)


@praximetry.stage("localize")
def localize(issue: str) -> str:
    messages = [
        {"role": "system", "content": RULES},
        {"role": "user", "content":
            f"Repository:\n{REPO_MAP}\n\n{FEW_SHOT}\nWhich file must change? "
            f"Reply with exactly the filename, nothing else.\nIssue: {issue}"},
    ]
    raw = real_chat(premium_model(), messages).strip()
    return next((p for p in REPO if p in raw), raw)


@praximetry.stage("propose_patch")
def propose_patch(path: str, issue: str) -> str:
    messages = [
        {"role": "system", "content": RULES},
        {"role": "user", "content":
            f"File {path}:\n{REPO.get(path, '')}\n\nRewrite it to fix the issue below. "
            "Return the complete file contents only, no markdown fences, no prose.\n"
            f"Issue: {issue}"},
    ]
    return real_chat(default_model(), messages)


@praximetry.stage("run_tests")
def run_tests(path: str, patch: str) -> str:
    """Non-LLM: actually execute the patched module and run the repo's tests."""
    ns: dict = {}
    try:
        exec(compile(patch, path, "exec"), ns)  # noqa: S102 - sandboxed example repo
        return "PASS" if TESTS[path](ns) else "FAIL"
    except Exception:  # noqa: BLE001 - a crashing patch is just a failing patch
        return "FAIL"


@praximetry.stage("resolve")
def resolve(issue: str) -> str:
    path = localize(issue)
    if path not in REPO:
        return "FAILED"
    patch = propose_patch(path, issue)
    return "RESOLVED" if run_tests(path, patch) == "PASS" else "FAILED"


ISSUES = [
    "mean() raises ZeroDivisionError on an empty list",
    "truncate() appends an ellipsis even when the string is shorter than n",
    "is_leap(1900) returns True but 1900 was not a leap year",
    "averaging no samples crashes the report job",
    "short strings come back with a trailing ... in the UI",
    "century years break the calendar view",
]

if __name__ == "__main__":
    for issue in ISSUES:
        print(f"  {resolve(issue):9s} {issue}")
    print("\nTraffic recorded. Try: praximetry-cloud detect")
