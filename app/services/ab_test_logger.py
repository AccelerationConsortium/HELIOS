"""A/B test logger for comparing RL vs rule-based strategy selection.

Records campaign outcomes per treatment group and provides
statistical analysis for deciding when RL is safe to deploy.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ABTestLogger", "ABTestRecord"]


@dataclass
class ABTestRecord:
    """Single A/B test observation."""

    campaign_id: str
    treatment: str  # "rule_based" | "rl_dqn" | "rl_ppo" | "rl_q_learning"
    n_rounds: int
    final_kpi: float | None
    converged: bool
    target_reached: bool
    direction: str = "maximize"
    best_kpi: float | None = None
    total_runs: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ABTestLogger:
    """Logs A/B test results to SQLite for offline analysis."""

    TABLE_NAME = "ab_test_results"
    DDL = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        campaign_id    TEXT PRIMARY KEY,
        treatment      TEXT NOT NULL,
        n_rounds       INTEGER,
        final_kpi      REAL,
        best_kpi       REAL,
        converged      INTEGER,
        target_reached INTEGER,
        direction      TEXT,
        total_runs     INTEGER,
        timestamp      TEXT
    )
    """

    def __init__(self, db_path: str = "otbot.db"):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(self.DDL)
            conn.commit()
            conn.close()
        except Exception:
            logger.debug("Failed to create AB test table", exc_info=True)

    def log_result(self, record: ABTestRecord) -> None:
        """Log a campaign result."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {self.TABLE_NAME}
                (campaign_id, treatment, n_rounds, final_kpi, best_kpi,
                 converged, target_reached, direction, total_runs, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.campaign_id,
                    record.treatment,
                    record.n_rounds,
                    record.final_kpi,
                    record.best_kpi,
                    int(record.converged),
                    int(record.target_reached),
                    record.direction,
                    record.total_runs,
                    record.timestamp,
                ),
            )
            conn.commit()
            conn.close()
            logger.info(
                "AB test logged: campaign=%s treatment=%s kpi=%s",
                record.campaign_id,
                record.treatment,
                record.final_kpi,
            )
        except Exception:
            logger.debug("Failed to log AB test result", exc_info=True)

    def get_results(
        self,
        treatment: str | None = None,
        limit: int = 1000,
    ) -> list[ABTestRecord]:
        """Query results, optionally filtered by treatment."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            if treatment:
                rows = conn.execute(
                    f"SELECT * FROM {self.TABLE_NAME} WHERE treatment = ? "
                    f"ORDER BY timestamp DESC LIMIT ?",
                    (treatment, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM {self.TABLE_NAME} "
                    f"ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            conn.close()

            return [
                ABTestRecord(
                    campaign_id=r["campaign_id"],
                    treatment=r["treatment"],
                    n_rounds=r["n_rounds"],
                    final_kpi=r["final_kpi"],
                    best_kpi=r["best_kpi"],
                    converged=bool(r["converged"]),
                    target_reached=bool(r["target_reached"]),
                    direction=r["direction"] or "maximize",
                    total_runs=r["total_runs"] or 0,
                    timestamp=r["timestamp"] or "",
                )
                for r in rows
            ]
        except Exception:
            logger.debug("Failed to query AB test results", exc_info=True)
            return []

    def compute_summary(self) -> dict[str, Any]:
        """Compute per-treatment summary statistics.

        Returns:
            {
                "rule_based": {"n": 50, "avg_kpi": 0.85, "convergence_rate": 0.7, ...},
                "rl_dqn": {"n": 10, "avg_kpi": 0.88, "convergence_rate": 0.8, ...},
            }
        """
        results = self.get_results()
        if not results:
            return {}

        # Group by treatment
        groups: dict[str, list[ABTestRecord]] = {}
        for r in results:
            groups.setdefault(r.treatment, []).append(r)

        summary: dict[str, Any] = {}
        for treatment, records in groups.items():
            kpis = [r.final_kpi for r in records if r.final_kpi is not None]
            summary[treatment] = {
                "n": len(records),
                "avg_kpi": sum(kpis) / len(kpis) if kpis else None,
                "median_kpi": sorted(kpis)[len(kpis) // 2] if kpis else None,
                "best_kpi": max(kpis) if kpis else None,
                "worst_kpi": min(kpis) if kpis else None,
                "convergence_rate": (
                    sum(1 for r in records if r.converged) / len(records)
                ),
                "target_rate": (
                    sum(1 for r in records if r.target_reached) / len(records)
                ),
                "avg_rounds": (
                    sum(r.n_rounds for r in records) / len(records)
                ),
            }

        return summary
