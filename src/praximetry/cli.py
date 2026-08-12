"""praximetry CLI: summary | eval

`eval` is the customer-facing half of the hosted eval pipeline: it captures
this stage's hosted golden examples as LLM request shapes (no real LLM call,
see `eval/capture.py`) and hands them to the cloud, which does the actual
model call and scoring with its own credentials. Capture has to happen here
because a golden example can only be *run* where `@praximetry.stage` was
registered — the customer's own process; the cloud never imports customer
agent code.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Drop-in observability for your LLM agents.")


@app.callback()
def _main() -> None:
    pass


def _import_module(module: str | None) -> None:
    """Import the user's agent module so @praximetry.stage registrations exist."""
    if module:
        sys.path.insert(0, str(Path.cwd()))
        importlib.import_module(module)


@app.command("eval")
def eval_cmd(
    stage: str = typer.Option(..., "--stage", help="Stage whose hosted corpus to evaluate"),
    module: str = typer.Option(None, "--module", "-m",
                                help="Python module defining your @praximetry.stage functions"),
    fail_under: float = typer.Option(0.9, "--fail-under",
                                      help="Exit non-zero if aggregate quality falls below this"),
) -> None:
    """CI gate: capture this stage's hosted golden corpus and have the cloud score it.

    Pulls the corpus from the hosted API, runs each example's real registered
    stage code in *this* process just far enough to capture the request shape
    it would send to an LLM (no real LLM call happens here), then pushes those
    shapes to the cloud, which makes the real call and scores the result.
    Exit 0 = passed, 1 = below --fail-under, 2 = the gate could not run (no
    key, empty corpus, nothing capturable, or the API call failed).

    Env: PRAXIMETRY_API_KEY (required), PRAXIMETRY_API_URL (default localhost).
    """
    from .eval import CaptureError, capture_request
    from .eval.hosted import EXIT_GATE_FAILED, EXIT_OK, EXIT_UNUSABLE, CloudError, client_from_env

    try:
        client = client_from_env()
        _import_module(module)
        dataset = client.fetch_corpus(stage)
        if not dataset.examples:
            typer.echo(f"No golden examples for stage '{stage}'.")
            raise typer.Exit(EXIT_UNUSABLE)

        captures = []
        for ex in dataset.examples:
            try:
                captures.append(capture_request(ex))
            except CaptureError as e:
                typer.echo(f"  [SKIP] {ex.stage}/{ex.id}: {e}")
        if not captures:
            typer.echo("No captures produced — nothing to evaluate.")
            raise typer.Exit(EXIT_UNUSABLE)

        result = client.push_captures(stage, captures)
    except CloudError as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(EXIT_UNUSABLE) from e

    quality, pass_rate = result["quality"], result["pass_rate"]
    typer.echo(
        f"quality={quality:.2f} pass_rate={pass_rate:.0%} cost=${result['cost_usd']:.4f} "
        f"examples={result['count']}"
    )
    if result.get("skipped"):
        typer.echo(f"skipped (no matching golden example): {', '.join(result['skipped'])}")

    if quality < fail_under:
        typer.echo(f"FAIL: quality {quality:.3f} < --fail-under {fail_under:.3f}")
        raise typer.Exit(EXIT_GATE_FAILED)
    typer.echo(f"PASS: quality {quality:.3f} >= --fail-under {fail_under:.3f}")
    raise typer.Exit(EXIT_OK)


@app.command()
def summary() -> None:
    """Print usage totals and per-stage breakdown."""
    from .store import get_store

    from .currency import fmt as money

    store = get_store()
    t = store.totals()
    typer.echo(f"calls={t['n']}  tokens_in={t['tin']}  tokens_out={t['tout']}  "
               f"cost={money(t['cost'])}\n")
    for s in store.stage_summary():
        typer.echo(
            f"  {str(s['stage']):<24} {s['model']:<24} n={s['n']:<6} "
            f"in={s['tin']:<9} out={s['tout']:<8} {money(s['cost'])}"
        )


if __name__ == "__main__":
    app()
