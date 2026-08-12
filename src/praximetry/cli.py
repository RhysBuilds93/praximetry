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
def optimize(
    stage: str = typer.Option(..., "--stage", help="Stage to trigger a hosted optimization run for"),
    module: str = typer.Option(None, "--module", "-m",
                                help="Python module defining your @praximetry.stage functions"),
    model: list[str] = typer.Option(None, "--model",
                                     help="Candidate model(s) to trial (repeatable)"),
    transform: list[str] = typer.Option(None, "--transform",
                                         help="Candidate prompt transform(s) to trial (repeatable)"),
    quality_tolerance: float = typer.Option(0.02, "--quality-tolerance",
                                             help="Max quality drop from baseline still considered a win"),
    max_trials: int = typer.Option(8, "--max-trials", help="Trial budget for the hosted run"),
) -> None:
    """Trigger a hosted optimization run for `stage`.

    Captures one fresh request shape for `stage` from *this* process's
    registered code (same capture step `eval` uses, no real LLM call happens
    here) and submits it to the cloud, which runs the actual trial loop
    (transforms, candidate models, scoring) with its own credentials. Exit 0
    on a successful submission, non-zero if the gate couldn't run (no key,
    API unreachable, empty corpus, nothing capturable).

    Env: PRAXIMETRY_API_KEY (required), PRAXIMETRY_API_URL (default localhost).
    """
    from .eval import CaptureError, capture_request
    from .eval.hosted import CloudError, client_from_env

    try:
        client = client_from_env()
        _import_module(module)
        dataset = client.fetch_corpus(stage)
        if not dataset.examples:
            typer.echo(f"No golden examples for stage '{stage}'.")
            raise typer.Exit(1)

        try:
            captured = capture_request(dataset.examples[0])
        except CaptureError as e:
            typer.echo(f"error: {e}")
            raise typer.Exit(1) from e

        result = client.push_optimize_capture(
            stage,
            captured,
            candidate_models=model or (),
            transforms=transform or (),
            quality_tolerance=quality_tolerance,
            max_trials=max_trials,
        )
    except CloudError as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(1) from e

    winner = result.get("winner")
    typer.echo(f"stage={result.get('stage', stage)} examples={result.get('examples', result.get('count', '?'))}")
    if winner:
        savings = result.get("savings_pct")
        savings_str = f" savings={savings:.0%}" if savings is not None else ""
        typer.echo(f"winner found: {winner}{savings_str}")
    else:
        typer.echo("no winner found (nothing beat baseline within tolerance)")
    if result.get("truncated"):
        typer.echo("warning: run was truncated (max_trials reached)")
    if result.get("errors"):
        typer.echo(f"errors: {result['errors']}")
    raise typer.Exit(0)


@app.command()
def apply(
    stage: str = typer.Option(..., "--stage", help="Stage to apply the hosted optimize winner for"),
) -> None:
    """Fetch the winning policy from a completed `optimize` run and write it
    locally to .praximetry/overrides.json.

    No capture, no optimization logic — just fetch-and-write. Exit codes:
    0 = applied, or ran fine and there was nothing to apply (no candidate beat
    baseline); 1 = no completed optimize run exists yet for this stage; 2 =
    the gate could not run at all (no key, API unreachable).

    Env: PRAXIMETRY_API_KEY (required), PRAXIMETRY_API_URL (default localhost).
    """
    import json
    import time
    from pathlib import Path

    from .eval.hosted import CloudError, client_from_env

    try:
        client = client_from_env()
        winner = client.fetch_winner(stage)
    except CloudError as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(2) from e

    if winner is None:
        typer.echo(
            f"no optimize run found for stage '{stage}' — run "
            f"`praximetry optimize --stage {stage} -m your_module` first"
        )
        raise typer.Exit(1)

    if not winner.get("model") and not winner.get("transforms"):
        typer.echo(f"optimize run for stage '{stage}' completed but found no winner — nothing to apply.")
        raise typer.Exit(0)

    path = Path(".praximetry") / "overrides.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stages: dict = {}
    if path.exists():
        try:
            stages = json.loads(path.read_text()).get("stages", {})
        except (json.JSONDecodeError, OSError):
            stages = {}
    stages[stage] = {
        "model": winner.get("model"),
        "transforms": winner.get("transforms") or [],
        "experiment_id": winner.get("experiment_id"),
        "savings_pct": winner.get("savings_pct"),
        "applied_at": time.time(),
    }
    path.write_text(json.dumps({"stages": stages}, indent=2))

    savings = winner.get("savings_pct")
    savings_str = f" savings={savings:.0%}" if savings is not None else ""
    typer.echo(f"applied stage='{stage}' model={winner.get('model')} "
               f"transforms={winner.get('transforms') or []}{savings_str} -> {path}")
    raise typer.Exit(0)


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
