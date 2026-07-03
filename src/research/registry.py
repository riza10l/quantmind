"""
Research Run Registry
=====================
Every experiment (backtest, training run, feature selection, portfolio test)
is recorded with enough context to reproduce it: dataset hash, git commit,
parameters, random seed, metrics, and artifact paths.

Usage:
    registry = RunRegistry(db)
    run_id = registry.start_run("ema_cross backtest", kind="backtest",
                                params={"fast": 9}, seed=42, dataset_hash=h)
    ... do work ...
    registry.finish_run(run_id, metrics={"sharpe": 1.2},
                        artifacts={"equity_curve": "runs/abc/equity.csv"})
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from typing import Any

import pandas as pd
from sqlalchemy import Column, DateTime, Integer, String, Table, Text, text

from src.core.database import DatabaseManager, _utc_now_naive, metadata
from src.core.logger import get_logger

logger = get_logger("research.registry")

research_runs_table = Table(
    "research_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(36), nullable=False, unique=True),
    Column("name", String(200), nullable=False),
    Column("kind", String(50), nullable=False),          # backtest | training | selection | portfolio
    Column("git_commit", String(40), nullable=True),
    Column("dataset_hash", String(64), nullable=True),
    Column("seed", Integer, nullable=True),
    Column("params", Text, nullable=True),               # JSON
    Column("metrics", Text, nullable=True),              # JSON
    Column("artifacts", Text, nullable=True),            # JSON: name -> path
    Column("status", String(20), nullable=False, default="running"),
    Column("created_at", DateTime, default=_utc_now_naive),
    Column("completed_at", DateTime, nullable=True),
)


def git_commit() -> str | None:
    """Current git HEAD commit hash, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic content hash of a DataFrame (index + values)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()


class RunRegistry:
    """Records research runs in the database for reproducibility."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        # Idempotent: creates research_runs if missing, leaves the rest alone.
        research_runs_table.create(db.engine, checkfirst=True)

    def start_run(
        self,
        name: str,
        kind: str,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
        dataset_hash: str | None = None,
    ) -> str:
        if not name.strip() or not kind.strip():
            raise ValueError("run name and kind are required")
        run_id = str(uuid.uuid4())
        with self.db.session() as session:
            session.execute(research_runs_table.insert().values(
                run_id=run_id,
                name=name,
                kind=kind,
                git_commit=git_commit(),
                dataset_hash=dataset_hash,
                seed=seed,
                params=json.dumps(params or {}, default=str),
                status="running",
            ))
        logger.info("run_started", run_id=run_id, name=name, kind=kind)
        return run_id

    def finish_run(
        self,
        run_id: str,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, str] | None = None,
        status: str = "completed",
    ) -> None:
        with self.db.session() as session:
            result = session.execute(
                research_runs_table.update()
                .where(research_runs_table.c.run_id == run_id)
                .values(
                    metrics=json.dumps(metrics or {}, default=str),
                    artifacts=json.dumps(artifacts or {}, default=str),
                    status=status,
                    completed_at=_utc_now_naive(),
                )
            )
            if result.rowcount == 0:
                raise ValueError(f"unknown run_id: {run_id}")
        logger.info("run_finished", run_id=run_id, status=status)

    def log_run(
        self,
        name: str,
        kind: str,
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        seed: int | None = None,
        dataset_hash: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> str:
        """One-shot: record an already-completed run."""
        run_id = self.start_run(name, kind, params=params, seed=seed,
                                dataset_hash=dataset_hash)
        self.finish_run(run_id, metrics=metrics, artifacts=artifacts)
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        df = pd.read_sql(
            text("SELECT * FROM research_runs WHERE run_id = :rid"),
            self.db.engine, params={"rid": run_id},
        )
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        for key in ("params", "metrics", "artifacts"):
            row[key] = json.loads(row[key]) if row.get(key) else {}
        return row

    def list_runs(self, kind: str | None = None, limit: int = 50) -> pd.DataFrame:
        query = "SELECT run_id, name, kind, git_commit, dataset_hash, seed, status, created_at, completed_at FROM research_runs "
        params: dict[str, Any] = {}
        if kind:
            query += "WHERE kind = :kind "
            params["kind"] = kind
        query += f"ORDER BY created_at DESC LIMIT {int(limit)}"
        return pd.read_sql(text(query), self.db.engine, params=params)
