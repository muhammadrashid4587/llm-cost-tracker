"""Tests for export functionality (CSV, JSON, SQLite)."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_cost_tracker.export import export_csv, export_json, export_sqlite
from llm_cost_tracker.models import UsageRecord
from llm_cost_tracker.tracker import CostTracker


def _sample_records() -> list[UsageRecord]:
    """Create a small set of records for testing."""
    tracker = CostTracker()
    tracker.record(model="gpt-4o", provider="openai", input_tokens=100, output_tokens=50)
    tracker.record(model="gpt-4o-mini", provider="openai", input_tokens=200, output_tokens=100)
    tracker.record(model="claude-sonnet-4-6", provider="anthropic", input_tokens=150, output_tokens=75)
    return tracker.records


class TestExportCSV:
    """CSV export tests."""

    def test_creates_file(self, tmp_path: Path):
        path = tmp_path / "test.csv"
        export_csv(_sample_records(), path)
        assert path.exists()

    def test_correct_header(self, tmp_path: Path):
        path = tmp_path / "test.csv"
        export_csv(_sample_records(), path)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            assert "model" in reader.fieldnames
            assert "provider" in reader.fieldnames
            assert "total_cost" in reader.fieldnames
            assert "timestamp" in reader.fieldnames

    def test_correct_row_count(self, tmp_path: Path):
        path = tmp_path / "test.csv"
        records = _sample_records()
        export_csv(records, path)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == len(records)

    def test_data_integrity(self, tmp_path: Path):
        path = tmp_path / "test.csv"
        records = _sample_records()
        export_csv(records, path)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows[0]["model"] == "gpt-4o"
        assert int(rows[0]["input_tokens"]) == 100

    def test_empty_records(self, tmp_path: Path):
        path = tmp_path / "test.csv"
        export_csv([], path)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 0

    def test_string_path(self, tmp_path: Path):
        path = str(tmp_path / "test.csv")
        export_csv(_sample_records(), path)
        assert Path(path).exists()


class TestExportJSON:
    """JSON export tests."""

    def test_creates_file(self, tmp_path: Path):
        path = tmp_path / "test.json"
        export_json(_sample_records(), path)
        assert path.exists()

    def test_valid_json(self, tmp_path: Path):
        path = tmp_path / "test.json"
        export_json(_sample_records(), path)
        data = json.loads(path.read_text())
        assert isinstance(data, list)

    def test_correct_count(self, tmp_path: Path):
        path = tmp_path / "test.json"
        records = _sample_records()
        export_json(records, path)
        data = json.loads(path.read_text())
        assert len(data) == len(records)

    def test_data_fields(self, tmp_path: Path):
        path = tmp_path / "test.json"
        export_json(_sample_records(), path)
        data = json.loads(path.read_text())
        entry = data[0]
        assert "model" in entry
        assert "provider" in entry
        assert "total_cost" in entry
        assert "timestamp" in entry
        assert "input_tokens" in entry
        assert "output_tokens" in entry

    def test_empty_records(self, tmp_path: Path):
        path = tmp_path / "test.json"
        export_json([], path)
        data = json.loads(path.read_text())
        assert data == []


class TestExportSQLite:
    """SQLite export tests."""

    def test_creates_file(self, tmp_path: Path):
        path = tmp_path / "test.db"
        export_sqlite(_sample_records(), path)
        assert path.exists()

    def test_correct_row_count(self, tmp_path: Path):
        path = tmp_path / "test.db"
        records = _sample_records()
        export_sqlite(records, path)
        conn = sqlite3.connect(str(path))
        cursor = conn.execute("SELECT COUNT(*) FROM usage_records")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == len(records)

    def test_data_integrity(self, tmp_path: Path):
        path = tmp_path / "test.db"
        export_sqlite(_sample_records(), path)
        conn = sqlite3.connect(str(path))
        cursor = conn.execute(
            "SELECT model, provider, input_tokens FROM usage_records ORDER BY id"
        )
        rows = cursor.fetchall()
        conn.close()
        assert rows[0] == ("gpt-4o", "openai", 100)
        assert rows[1] == ("gpt-4o-mini", "openai", 200)

    def test_appends_on_second_export(self, tmp_path: Path):
        path = tmp_path / "test.db"
        records = _sample_records()
        export_sqlite(records, path)
        export_sqlite(records, path)  # second call
        conn = sqlite3.connect(str(path))
        cursor = conn.execute("SELECT COUNT(*) FROM usage_records")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == len(records) * 2

    def test_empty_records(self, tmp_path: Path):
        path = tmp_path / "test.db"
        export_sqlite([], path)
        conn = sqlite3.connect(str(path))
        cursor = conn.execute("SELECT COUNT(*) FROM usage_records")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0

    def test_schema_columns(self, tmp_path: Path):
        path = tmp_path / "test.db"
        export_sqlite(_sample_records(), path)
        conn = sqlite3.connect(str(path))
        cursor = conn.execute("PRAGMA table_info(usage_records)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {
            "id", "timestamp", "provider", "model",
            "input_tokens", "output_tokens", "cached_input_tokens",
            "total_tokens", "input_cost", "output_cost",
            "cached_input_cost", "total_cost",
        }
        assert expected == columns


class TestTrackerExportMethods:
    """Test the convenience export methods on CostTracker."""

    def test_export_csv(self, tmp_path: Path):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=50, output_tokens=25)
        path = tmp_path / "out.csv"
        tracker.export_csv(str(path))
        assert path.exists()

    def test_export_json(self, tmp_path: Path):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=50, output_tokens=25)
        path = tmp_path / "out.json"
        tracker.export_json(str(path))
        data = json.loads(path.read_text())
        assert len(data) == 1

    def test_export_sqlite(self, tmp_path: Path):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=50, output_tokens=25)
        path = tmp_path / "out.db"
        tracker.export_sqlite(str(path))
        conn = sqlite3.connect(str(path))
        cursor = conn.execute("SELECT COUNT(*) FROM usage_records")
        assert cursor.fetchone()[0] == 1
        conn.close()
