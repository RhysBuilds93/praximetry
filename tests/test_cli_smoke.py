from typer.testing import CliRunner

from praximetry.cli import app
from praximetry.runtime import record_call

runner = CliRunner()


def _traffic(n=6):
    for _ in range(n):
        record_call(
            provider="fake",
            model="claude-opus-4-8",
            stage="classify",
            input_tokens=900,
            output_tokens=12,
            cost_usd=0.005,
            messages=[
                {"role": "system", "content": "Be brief. Be brief. Be brief."},
                {"role": "user", "content": "ticket text"},
            ],
        )


def test_summary_command():
    _traffic()
    res = runner.invoke(app, ["summary"])
    assert res.exit_code == 0
    assert "classify" in res.output and "claude-opus-4-8" in res.output
