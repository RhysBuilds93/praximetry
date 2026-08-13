"""SQLite-backed trace store. Zero-config, thread-safe, WAL mode."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel

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

# Model backing each table — column order in _TABLES must match each model's
# `model_fields` declaration order (verified by the round-trip tests in
# tests/test_core.py). _to_row/_row_to_model below use this to drive generic
# save/read instead of hand-written positional tuples.
_MODELS: dict[str, type[BaseModel]] = {
    "runs": Run,
    "calls": Call,
    "eval_results": EvalResult,
    "experiments": Experiment,
}


def _unwrap_optional(annotation: Any) -> Any:
    """Strip `Optional[X]` / `X | None` down to `X`; pass through everything else."""
    args = get_args(annotation)
    if get_origin(annotation) is not None and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _to_row(obj: BaseModel) -> tuple:
    """Serialize `obj` into a tuple matching its table's column order in `_TABLES`."""
    values = []
    for name in type(obj).model_fields:
        value = getattr(obj, name)
        if isinstance(value, (dict, list)):
            values.append(json.dumps(value))
        elif isinstance(value, bool):
            values.append(int(value))
        else:
            values.append(value)
    return tuple(values)


def _row_to_model(cls: type[BaseModel], row: sqlite3.Row) -> BaseModel:
    """Inverse of `_to_row`: rehydrate a model instance from a raw sqlite row."""
    d = dict(row)
    for name, field in cls.model_fields.items():
        annotation = _unwrap_optional(field.annotation)
        origin = get_origin(annotation) or annotation
        if origin is dict:
            d[name] = json.loads(d[name] or "{}")
        elif origin is list:
            d[name] = json.loads(d[name] or "[]")
        elif origin is bool:
            d[name] = bool(d[name])
    return cls(**d)


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
    def _save(self, table: str, obj: BaseModel) -> None:
        placeholders = ",".join("?" for _ in _MODELS[table].model_fields)
        with self._conn() as c:
            c.execute(
                f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})",
                _to_row(obj),
            )

    def save_run(self, run: Run) -> None:
        self._save("runs", run)

    def save_call(self, call: Call) -> None:
        self._save("calls", call)

    def save_eval_result(self, r: EvalResult) -> None:
        self._save("eval_results", r)

    def save_experiment(self, e: Experiment) -> None:
        self._save("experiments", e)

    # -- reads -------------------------------------------------------------
    def runs(self, limit: int = 100) -> list[Run]:
        rows = self._conn().execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_model(Run, r) for r in rows]

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
        return [_row_to_model(Call, r) for r in rows]

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
        rows = self._conn().execute(q, args).fetchall()
        return [_row_to_model(Experiment, r) for r in rows]

    def eval_results(self, experiment_id: str | None = None) -> list[EvalResult]:
        q, args = "SELECT * FROM eval_results", []
        if experiment_id:
            q += " WHERE experiment_id=?"
            args.append(experiment_id)
        q += " ORDER BY ts DESC"
        rows = self._conn().execute(q, args).fetchall()
        return [_row_to_model(EvalResult, r) for r in rows]

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
