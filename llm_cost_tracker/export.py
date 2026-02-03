"""
Export usage records to CSV, JSON, and SQLite.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Sequence

from .models import UsageRecord

_CSV_FIELDS = [
    "timestamp",
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "total_tokens",
    "input_cost",
    "output_cost",
    "cached_input_cost",
    "total_cost",
]


def export_csv(records: Sequence[UsageRecord], path: str | Path) -> None:
    """
    Write *records* to a CSV file at *path*.

    Parameters
    ----------
    records : sequence of UsageRecord
    path : str or Path
    """
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = rec.model_dump()
            row["timestamp"] = rec.timestamp.isoformat()
            writer.writerow(row)


def export_json(records: Sequence[UsageRecord], path: str | Path) -> None:
    """
    Write *records* to a JSON file at *path*.

    The output is a JSON array of objects, one per record.

    Parameters
    ----------
    records : sequence of UsageRecord
    path : str or Path
    """
    path = Path(path)
    data = []
    for rec in records:
        d = rec.model_dump()
        d["timestamp"] = rec.timestamp.isoformat()
        # metadata may contain non-serializable objects; best-effort
        d["metadata"] = rec.metadata
        data.append(d)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


_SQLITE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS usage_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    input_cost      REAL    NOT NULL DEFAULT 0.0,
    output_cost     REAL    NOT NULL DEFAULT 0.0,
    cached_input_cost REAL  NOT NULL DEFAULT 0.0,
    total_cost      REAL    NOT NULL DEFAULT 0.0
);
"""


def export_sqlite(records: Sequence[UsageRecord], path: str | Path) -> None:
    """
    Write *records* to a SQLite database at *path*.

    Creates the ``usage_records`` table if it does not already exist and
    appends the given records.

    Parameters
    ----------
    records : sequence of UsageRecord
    path : str or Path
    """
    path = Path(path)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_SQLITE_SCHEMA)
        conn.executemany(
            """
            INSERT INTO usage_records (
                timestamp, provider, model,
                input_tokens, output_tokens, cached_input_tokens, total_tokens,
                input_cost, output_cost, cached_input_cost, total_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    rec.timestamp.isoformat(),
                    rec.provider,
                    rec.model,
                    rec.input_tokens,
                    rec.output_tokens,
                    rec.cached_input_tokens,
                    rec.total_tokens,
                    rec.input_cost,
                    rec.output_cost,
                    rec.cached_input_cost,
                    rec.total_cost,
                )
                for rec in records
            ],
        )
        conn.commit()
    finally:
        conn.close()
