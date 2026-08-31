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

from praximetry import __version__, _banner

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Drop-in observability for your LLM agents."
)


@app.callback()
def _main() -> None:
    if sys.stdout.isatty():
        typer.echo(_banner.render())
        typer.secho(f"  CLI  v{__version__}", fg=typer.colors.BRIGHT_BLACK)
        typer.echo("─" * 60)
        typer.echo("Type praximetry --help to see all commands.\n")


def _import_module(module: str | None) -> None:
    """Import the user's agent module so @praximetry.stage registrations exist."""
    if module:
        sys.path.insert(0, str(Path.cwd()))
        importlib.import_module(module)


@app.command("eval")
def eval_cmd(
    stage: str = typer.Option(None, "--stage", help="Stage whose hosted corpus to evaluate"),
    project: str = typer.Option(
        None, "--project", help="Project whose hosted corpus to evaluate (every stage under it)"
    ),
    module: str = typer.Option(
        None, "--module", "-m", help="Python module defining your @praximetry.stage functions"
    ),
    fail_under: float = typer.Option(
        None,
        "--fail-under",
        help="Exit non-zero if aggregate quality falls below this "
        "(default: the hosted project/stage config, or 0.9 "
        "for plain --stage runs)",
    ),
    config: str = typer.Option(
        None,
        "--config",
        help="Path to a JSON file with {project, stage, module, fail_under, scorers}; "
        "explicit CLI flags override values in the file",
    ),
    scorer: list[str] = typer.Option(
        None,
        "--scorer",
        help="Scorer(s) the cloud should use (repeatable); overrides --config "
        "and the hosted project/stage default. Unset everywhere = server default.",
    ),
) -> None:
    """CI gate: capture the hosted golden corpus and have the cloud score it.

    Pulls the corpus from the hosted API, runs each example's real registered
    stage code in *this* process just far enough to capture the request shape
    it would send to an LLM (no real LLM call happens here), then pushes those
    shapes to the cloud, which makes the real call and scores the result.
    Exit 0 = passed, 1 = below --fail-under, 2 = the gate could not run (no
    key, empty corpus, nothing capturable, or the API call failed).

    With no --stage/--project/--config, runs the default saved from the
    dashboard ("Save as default"); exits 2 if no default has been saved.
    --stage alone evaluates just that stage. --project alone evaluates every
    stage under that project and gates on the aggregate (unweighted mean of
    each stage's quality). --stage and --project together narrows to one stage
    within a project. --config loads a JSON file; any of --stage/--project/-m/
    --fail-under given alongside override the file's values.

    --scorer (repeatable) sets which scorers the cloud runs, resolved
    explicit flag > --config "scorers" > hosted project/stage default >
    unset (server default). The hosted tier needs a project; a bare --stage
    run has no addressable hosted config, same as --fail-under.

    Env: PRAXIMETRY_API_KEY (required), PRAXIMETRY_API_URL (default localhost).
    """
    import json
    import uuid

    from .eval import CaptureError, capture_request
    from .eval.hosted import EXIT_GATE_FAILED, EXIT_OK, EXIT_UNUSABLE, CloudError, client_from_env

    cfg: dict = {}
    if config:
        try:
            cfg = json.loads(Path(config).read_text())
        except OSError as e:
            typer.echo(f"error: could not read --config {config}: {e}")
            raise typer.Exit(EXIT_UNUSABLE) from e
        except json.JSONDecodeError as e:
            typer.echo(f"error: --config {config} is not valid JSON: {e}")
            raise typer.Exit(EXIT_UNUSABLE) from e
        stage = stage if stage is not None else cfg.get("stage")
        project = project if project is not None else cfg.get("project")
        module = module if module is not None else cfg.get("module")
        fail_under = fail_under if fail_under is not None else cfg.get("fail_under")

    hosted_default_scorers: list[str] | None = None
    if not stage and not project:
        try:
            client = client_from_env()
            default = client.fetch_eval_default()
        except CloudError as e:
            typer.echo(f"error: {e}")
            raise typer.Exit(EXIT_UNUSABLE) from e
        if default is None:
            typer.echo(
                "error: no --stage/--project given and no default saved — "
                "set one in the dashboard (Save as default) or pass --config"
            )
            raise typer.Exit(EXIT_UNUSABLE)
        stage = stage if stage is not None else default.get("stage")
        project = project if project is not None else default.get("project")
        fail_under = fail_under if fail_under is not None else default.get("fail_under")
        hosted_default_scorers = default.get("scorers")
        typer.echo(f"using default: project={project}" + (f" stage={stage}" if stage else ""))
    else:
        try:
            client = client_from_env()
        except CloudError as e:
            typer.echo(f"error: {e}")
            raise typer.Exit(EXIT_UNUSABLE) from e

    try:
        _import_module(module)
        dataset = client.fetch_corpus(stage=stage, project=project)
        scope = (
            f"project '{project}'"
            if not stage
            else (f"stage '{stage}' in project '{project}'" if project else f"stage '{stage}'")
        )
        if not dataset.examples:
            typer.echo(f"No golden examples for {scope}.")
            raise typer.Exit(EXIT_UNUSABLE)

        experiment_id = str(uuid.uuid4())
        captures_by_stage: dict[str, list] = {}
        for ex in dataset.examples:
            try:
                captures_by_stage.setdefault(ex.stage, []).append(capture_request(ex))
            except CaptureError as e:
                typer.echo(f"  [SKIP] {ex.stage}/{ex.id}: {e}")
        if not any(captures_by_stage.values()):
            typer.echo("No captures produced — nothing to evaluate.")
            raise typer.Exit(EXIT_UNUSABLE)

        if scorer:
            scorers = list(scorer)
        elif cfg.get("scorers") is not None:
            scorers = list(cfg["scorers"])
        elif hosted_default_scorers is not None:
            scorers = hosted_default_scorers
        elif project:
            scorers = client.fetch_eval_config(project=project, stage=stage).get("scorers")
        else:
            scorers = None

        push_results: dict[str, dict] = {}
        for stage_name, caps in captures_by_stage.items():
            push_results[stage_name] = client.push_captures(
                stage_name, caps, experiment_id=experiment_id, scorers=scorers
            )
            if project and not stage and push_results[stage_name].get("skipped"):
                typer.echo(
                    f"skipped (no matching golden example) in {stage_name}: "
                    f"{', '.join(push_results[stage_name]['skipped'])}"
                )

        if project and not stage:
            results = client.fetch_results(project=project, experiment_id=experiment_id)
            for stage_name, stage_result in results["stages"].items():
                typer.echo(
                    f"  {stage_name}: quality={stage_result['quality']:.2f} "
                    f"examples={stage_result['count']}"
                )
            quality = results["aggregate_quality"]
            if fail_under is None:
                fail_under = client.fetch_eval_config(project=project)["fail_under"]
            typer.echo(f"aggregate quality={quality:.2f}")
        else:
            if stage in push_results:
                result = push_results[stage]
            elif len(push_results) == 1:
                result = next(iter(push_results.values()))
            else:
                typer.echo(
                    f"error: no scored result for stage '{stage}' "
                    f"(got: {', '.join(sorted(push_results))})"
                )
                raise typer.Exit(EXIT_UNUSABLE)
            quality, pass_rate = result["quality"], result["pass_rate"]
            typer.echo(
                f"quality={quality:.2f} pass_rate={pass_rate:.0%} cost=${result['cost_usd']:.4f} "
                f"examples={result['count']}"
            )
            if result.get("skipped"):
                typer.echo(f"skipped (no matching golden example): {', '.join(result['skipped'])}")
            if fail_under is None:
                fail_under = (
                    client.fetch_eval_config(project=project, stage=stage)["fail_under"]
                    if project
                    else 0.9
                )
    except CloudError as e:
        typer.echo(f"error: {e}")
        raise typer.Exit(EXIT_UNUSABLE) from e

    if quality < fail_under:
        typer.echo(f"FAIL: quality {quality:.3f} < --fail-under {fail_under:.3f}")
        raise typer.Exit(EXIT_GATE_FAILED)
    typer.echo(f"PASS: quality {quality:.3f} >= --fail-under {fail_under:.3f}")
    raise typer.Exit(EXIT_OK)


@app.command()
def optimize(
    stage: str = typer.Option(
        ..., "--stage", help="Stage to trigger a hosted optimization run for"
    ),
    module: str = typer.Option(
        None, "--module", "-m", help="Python module defining your @praximetry.stage functions"
    ),
    model: list[str] = typer.Option(
        None, "--model", help="Candidate model(s) to trial (repeatable)"
    ),
    transform: list[str] = typer.Option(
        None, "--transform", help="Candidate prompt transform(s) to trial (repeatable)"
    ),
    quality_tolerance: float = typer.Option(
        0.02, "--quality-tolerance", help="Max quality drop from baseline still considered a win"
    ),
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
    typer.echo(
        f"stage={result.get('stage', stage)} examples={result.get('examples', result.get('count', '?'))}"
    )
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
        typer.echo(
            f"optimize run for stage '{stage}' completed but found no winner — nothing to apply."
        )
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
    typer.echo(
        f"applied stage='{stage}' model={winner.get('model')} "
        f"transforms={winner.get('transforms') or []}{savings_str} -> {path}"
    )
    raise typer.Exit(0)


@app.command()
def summary() -> None:
    """Print usage totals and per-stage breakdown."""
    from .store import get_store

    from .currency import fmt as money

    store = get_store()
    t = store.totals()
    typer.echo(
        f"calls={t['n']}  tokens_in={t['tin']}  tokens_out={t['tout']}  "
        f"cost={money(t['cost'])}\n"
    )
    for s in store.stage_summary():
        typer.echo(
            f"  {str(s['stage']):<24} {s['model']:<24} n={s['n']:<6} "
            f"in={s['tin']:<9} out={s['tout']:<8} {money(s['cost'])}"
        )


if __name__ == "__main__":
    app()
