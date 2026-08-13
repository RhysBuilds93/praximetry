"""SQLite-backed trace store. Zero-config, thread-safe, WAL mode."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .config import get_config
from .models import Call, EvalResult, Experiment, Run

# Single source of truth for table shape: both the initial CREATE TABLE and the
# startup migration (below) are generated from this, so a column added here is
# picked up on existing on-disk databases with no new migration code.
_TABLES: dict[str, list[tuple[str, str]]] = {
    "runs": [
        ("id", "TEXT PRIMARY KEY"), ("project", "TEXT"), ("name", "TEXT"),
        ("started_at", "REAL"), ("ended_at", "REAL"),
        ("experiment_id", "TEXT"), ("metadata", "TEXT"),
    ],
    "calls": [
        ("id", "TEXT PRIMARY KEY"), ("run_id", "TEXT"), ("parent_call_id", "TEXT"),
        ("stage", "TEXT"), ("provider", "TEXT"), ("model", "TEXT"),
        ("messages", "TEXT"), ("response_text", "TEXT"),
        ("input_tokens", "INTEGER"), ("output_tokens", "INTEGER"),
        ("cost_usd", "REAL"), ("latency_ms", "REAL"),
        ("ts", "REAL"), ("error", "TEXT"), ("metadata", "TEXT"),
    ],
    "eval_results": [
        ("id", "TEXT PRIMARY KEY"), ("experiment_id", "TEXT"), ("run_id", "TEXT"),
        ("stage", "TEXT"), ("example_id", "TEXT"), ("scorer", "TEXT"),
        ("score", "REAL"), ("passed", "INTEGER"), ("detail", "TEXT"), ("ts", "REAL"),
    ],
    "experiments": [
        ("id", "TEXT PRIMARY KEY"), ("name", "TEXT"), ("stage", "TEXT"),
        ("variant", "TEXT"), ("created_at", "REAL"),
        ("quality", "REAL"), ("pass_rate", "REAL"), ("cost_usd", "REAL"),
        ("input_tokens", "INTEGER"), ("output_tokens", "INTEGER"), ("baseline", "INTEGER"),
    ],
}

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_calls_run ON calls(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_calls_stage ON calls(stage);",
]

_SCHEMA = "\n".join(
    f"CREATE TABLE IF NOT EXISTS {table} ("
    + ", ".join(f"{col} {type_}" for col, type_ in columns)
    + ");"
    for table, columns in _TABLES.items()
) + "\n" + "\n".join(_INDEXES)


class Store:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else get_config().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(_SCHEMA)
            self._migrate(c)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        """`CREATE TABLE IF NOT EXISTS` is a no-op against a pre-existing table, so a
        `.praximetry/*.db` created before a column existed stays missing it forever
        — not just the field, but silently breaking the positional INSERTs in
        `save_*` the moment they're next called against that file. Diff each
        table's on-disk columns against `_TABLES` and ALTER in whatever's missing,
        so adding a column to `_TABLES` is the only step a future schema change
        needs — no new one-off migration per column."""
        for table, columns in _TABLES.items():
            existing = {row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, type_ in columns:
                if col not in existing:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {type_.split()[0]}")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    # -- writes ------------------------------------------------------------
    def save_run(self, run: Run) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
                (run.id, run.project, run.name, run.started_at, run.ended_at,
                 run.experiment_id, json.dumps(run.metadata)),
            )

    def save_call(self, call: Call) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO calls
                   (id, run_id, parent_call_id, stage, provider, model, messages, response_text,
                    input_tokens, output_tokens, cost_usd, latency_ms, ts, error, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (call.id, call.run_id, call.parent_call_id, call.stage, call.provider, call.model,
                 json.dumps(call.messages), call.response_text,
                 call.input_tokens, call.output_tokens, call.cost_usd, call.latency_ms,
                 call.ts, call.error, json.dumps(call.metadata)),
            )

    def save_eval_result(self, r: EvalResult) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO eval_results VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r.id, r.experiment_id, r.run_id, r.stage, r.example_id,
                 r.scorer, r.score, int(r.passed), r.detail, r.ts),
            )

    def save_experiment(self, e: Experiment) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (e.id, e.name, e.stage, json.dumps(e.variant), e.created_at,
                 e.quality, e.pass_rate, e.cost_usd,
                 e.input_tokens, e.output_tokens, int(e.baseline)),
            )

    # -- reads -------------------------------------------------------------
    def runs(self, limit: int = 100) -> list[Run]:
        rows = self._conn().execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    @staticmethod
    def _row_to_run(r: sqlite3.Row) -> Run:
        d = dict(r)
        d["metadata"] = json.loads(d["metadata"] or "{}")
        return Run(**d)

    def calls(self, run_id: str | None = None, stage: str | None = None,
              limit: int = 1000) -> list[Call]:
        q, args = "SELECT * FROM calls", []
        conds = []
        if run_id:
            conds.append("run_id=?")
            args.append(run_id)
        if stage:
            conds.append("stage=?")
            args.append(stage)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        rows = self._conn().execute(q, args).fetchall()
        return [self._row_to_call(r) for r in rows]

    @staticmethod
    def _row_to_call(r: sqlite3.Row) -> Call:
        d = dict(r)
        d["messages"] = json.loads(d["messages"] or "[]")
        d["metadata"] = json.loads(d["metadata"] or "{}")
        return Call(**d)

    def stage_summary(self) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            """SELECT stage, model, COUNT(*) n, SUM(input_tokens) tin,
                      SUM(output_tokens) tout, SUM(cost_usd) cost, AVG(latency_ms) lat
               FROM calls GROUP BY stage, model ORDER BY cost DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def totals(self) -> dict[str, Any]:
        r = self._conn().execute(
            """SELECT COUNT(*) n, COALESCE(SUM(input_tokens),0) tin,
                      COALESCE(SUM(output_tokens),0) tout,
                      COALESCE(SUM(cost_usd),0) cost FROM calls"""
        ).fetchone()
        return dict(r)

    def cost_over_time(self, bucket_seconds: int = 3600) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            """SELECT CAST(ts / ? AS INTEGER) * ? bucket,
                      SUM(cost_usd) cost, SUM(input_tokens + output_tokens) tokens
               FROM calls GROUP BY bucket ORDER BY bucket""",
            (bucket_seconds, bucket_seconds),
        ).fetchall()
        return [dict(r) for r in rows]

    def experiments(self, stage: str | None = None) -> list[Experiment]:
        q, args = "SELECT * FROM experiments", []
        if stage:
            q += " WHERE stage=?"
            args.append(stage)
        q += " ORDER BY created_at DESC"
        out = []
        for r in self._conn().execute(q, args).fetchall():
            d = dict(r)
            d["variant"] = json.loads(d["variant"] or "{}")
            d["baseline"] = bool(d["baseline"])
            out.append(Experiment(**d))
        return out

    def eval_results(self, experiment_id: str | None = None) -> list[EvalResult]:
        q, args = "SELECT * FROM eval_results", []
        if experiment_id:
            q += " WHERE experiment_id=?"
            args.append(experiment_id)
        q += " ORDER BY ts DESC"
        rows = self._conn().execute(q, args).fetchall()
        return [EvalResult(**{**dict(r), "passed": bool(r["passed"])}) for r in rows]

    # -- bulk ingest (used by the remote collector) ------------------------
    def ingest(self, payload: dict[str, Any]) -> dict[str, int]:
        """Write a batch of serialized runs/calls/experiments/eval_results."""
        counts = {}
        for run in payload.get("runs", []):
            self.save_run(Run(**run))
        counts["runs"] = len(payload.get("runs", []))
        for call in payload.get("calls", []):
            self.save_call(Call(**call))
        counts["calls"] = len(payload.get("calls", []))
        for e in payload.get("experiments", []):
            self.save_experiment(Experiment(**e))
        counts["experiments"] = len(payload.get("experiments", []))
        for r in payload.get("eval_results", []):
            self.save_eval_result(EvalResult(**r))
        counts["eval_results"] = len(payload.get("eval_results", []))
        return counts


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def reset_store() -> None:
    """Testing hook: force re-open against current config."""
    global _store
    _store = None
