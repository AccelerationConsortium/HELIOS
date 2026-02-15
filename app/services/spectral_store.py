"""Spectral data store for raw instrument curves linked to runs and campaigns.

Stores EIS Nyquist spectra, LSV curves, CV data, XRD patterns, UV-Vis spectra
etc. in a normalised SQLite table.  Provides export helpers for shipping numeric
matrices to the Nexus embedding / SSL pipeline.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.db import connection, json_dumps, parse_json, run_txn, utcnow_iso

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SpectralRecord:
    """A single spectral / curve measurement."""

    record_id: str
    run_id: str
    campaign_id: str
    technique: str  # "eis" | "lsv" | "cv" | "xrd" | "uv_vis"
    raw_data: dict[str, Any]
    metadata: dict[str, Any]
    timestamp: str  # ISO-8601


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SpectralStore:
    """SQLite-backed store for :class:`SpectralRecord` objects."""

    def __init__(self) -> None:
        self.ensure_table()

    # ---- schema ----

    def ensure_table(self) -> None:
        """Create the ``spectral_records`` table if it does not exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS spectral_records (
            record_id   TEXT PRIMARY KEY,
            run_id      TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            technique   TEXT NOT NULL,
            raw_data_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            timestamp   TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_spectral_campaign
            ON spectral_records(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_spectral_run
            ON spectral_records(run_id);
        CREATE INDEX IF NOT EXISTS idx_spectral_technique
            ON spectral_records(technique);
        """
        with connection() as conn:
            conn.executescript(ddl)
            conn.commit()

    # ---- write ----

    def store(self, record: SpectralRecord) -> str:
        """Persist a :class:`SpectralRecord`.  Returns ``record_id``."""

        def _txn(conn):  # type: ignore[override]
            conn.execute(
                "INSERT INTO spectral_records "
                "(record_id, run_id, campaign_id, technique, raw_data_json, metadata_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.record_id,
                    record.run_id,
                    record.campaign_id,
                    record.technique,
                    json_dumps(record.raw_data),
                    json_dumps(record.metadata),
                    record.timestamp,
                ),
            )
            return record.record_id

        return run_txn(_txn)

    # ---- read ----

    def query(
        self, campaign_id: str, technique: str | None = None
    ) -> list[SpectralRecord]:
        """Query records by *campaign_id* and optional *technique* filter."""

        def _txn(conn):  # type: ignore[override]
            if technique is not None:
                rows = conn.execute(
                    "SELECT * FROM spectral_records "
                    "WHERE campaign_id = ? AND technique = ? ORDER BY timestamp",
                    (campaign_id, technique),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM spectral_records "
                    "WHERE campaign_id = ? ORDER BY timestamp",
                    (campaign_id,),
                ).fetchall()
            return [_row_to_record(r) for r in rows]

        return run_txn(_txn)

    def get_for_run(self, run_id: str) -> list[SpectralRecord]:
        """Return all spectral records associated with *run_id*."""

        def _txn(conn):  # type: ignore[override]
            rows = conn.execute(
                "SELECT * FROM spectral_records WHERE run_id = ? ORDER BY timestamp",
                (run_id,),
            ).fetchall()
            return [_row_to_record(r) for r in rows]

        return run_txn(_txn)

    # ---- export ----

    def export_for_nexus(self, campaign_id: str) -> list[list[float]]:
        """Flatten spectral data into a numeric matrix for Nexus APIs.

        Each row corresponds to one record.  Numeric values are extracted from
        ``raw_data`` in a deterministic order (sorted keys, only float-coercible
        leaf values and list elements are included).
        """
        records = self.query(campaign_id)
        matrix: list[list[float]] = []
        for rec in records:
            row = _flatten_to_floats(rec.raw_data)
            if row:
                matrix.append(row)
        return matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_record(row) -> SpectralRecord:  # sqlite3.Row
    return SpectralRecord(
        record_id=row["record_id"],
        run_id=row["run_id"],
        campaign_id=row["campaign_id"],
        technique=row["technique"],
        raw_data=parse_json(row["raw_data_json"], {}),
        metadata=parse_json(row["metadata_json"], {}),
        timestamp=row["timestamp"],
    )


def _flatten_to_floats(data: dict[str, Any]) -> list[float]:
    """Extract all numeric values from *data* in sorted-key order."""
    values: list[float] = []
    for key in sorted(data.keys()):
        val = data[key]
        if isinstance(val, (int, float)):
            values.append(float(val))
        elif isinstance(val, list):
            for item in val:
                try:
                    values.append(float(item))
                except (TypeError, ValueError):
                    pass
    return values
