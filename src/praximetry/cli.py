"""praximetry CLI: summary"""
from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Drop-in observability for your LLM agents.")


@app.callback()
def _main() -> None:
    pass


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
