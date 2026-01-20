"""SQLite-based benchmark history database for efficient storage and retrieval.

This module provides compact, efficient storage for benchmark history:
- Compacts individual JSON run files into a single database
- Stores time series data efficiently with compression
- Provides fast queries for aggregates and trends
- Manages automatic archival and retention

Why SQLite:
    - Single file, no server required
    - ACID compliant
    - Fast for our access patterns (mostly writes, occasional reads)
    - Built-in compression via ZLIB
    - Portable across machines

Classes:
    BenchmarkHistoryDB: Main database interface
    ArchiveManager: Automatic archival of old runs
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Database schema version for migrations
SCHEMA_VERSION = 1


@dataclass
class BenchmarkSummary:
    """Summary of a benchmark run for quick queries.

    Attributes:
        run_id: Unique identifier for the run.
        timestamp: When the benchmark was run.
        duration_sec: Total benchmark duration.
        model_name: Model that was benchmarked.
        gpu_name: GPU used.
        throughput: Samples per second.
        latency_ms: Average latency in milliseconds.
        memory_mb: Peak memory usage in MB.
        compressed: Whether full data is compressed.
    """

    run_id: str
    timestamp: datetime
    duration_sec: float
    model_name: str | None
    gpu_name: str | None
    throughput: float | None
    latency_ms: float | None
    memory_mb: float | None
    compressed: bool = False


class BenchmarkHistoryDB:
    """SQLite database for benchmark history.

    Provides efficient storage and retrieval of benchmark results.
    Supports compression, automatic archival, and fast queries.

    Example:
        >>> db = BenchmarkHistoryDB("benchmark_results/history.db")
        >>> db.initialize()
        >>> db.ingest_json_file("benchmark_results/history/run_001.json")
        >>> runs = db.query_runs(limit=10)
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connection.

        Yields:
            SQLite connection.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise

    def initialize(self) -> None:
        """Initialize database schema.

        Creates tables if they don't exist and runs migrations.
        """
        with self._connection() as conn:
            # Schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # Check current version
            cur = conn.execute("SELECT MAX(version) FROM schema_version")
            current_version = cur.fetchone()[0] or 0

            if current_version < SCHEMA_VERSION:
                self._run_migrations(conn, current_version)

    def _run_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run database migrations.

        Args:
            conn: Database connection.
            from_version: Current schema version.
        """
        if from_version < 1:
            # Initial schema
            conn.executescript("""
                -- Benchmark runs summary table
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    duration_sec REAL,
                    model_name TEXT,
                    gpu_name TEXT,
                    cuda_version TEXT,
                    pytorch_version TEXT,
                    throughput REAL,
                    latency_ms REAL,
                    memory_mb REAL,
                    power_watts REAL,
                    temperature_c REAL,
                    compressed INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Full benchmark data (possibly compressed)
                CREATE TABLE IF NOT EXISTS benchmark_data (
                    run_id TEXT PRIMARY KEY,
                    data_json TEXT,  -- Raw JSON if not compressed
                    data_compressed BLOB,  -- GZIP compressed JSON
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
                );

                -- Time series snapshots for resource tracking
                CREATE TABLE IF NOT EXISTS resource_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp_offset_ms INTEGER NOT NULL,
                    cpu_pct REAL,
                    memory_mb REAL,
                    gpu_util_pct REAL,
                    gpu_memory_mb REAL,
                    disk_read_mb REAL,
                    disk_write_mb REAL,
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
                );

                -- Aggregate metrics by category
                CREATE TABLE IF NOT EXISTS metric_aggregates (
                    run_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL,
                    unit TEXT,
                    PRIMARY KEY (run_id, category, metric_name),
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
                );

                -- Indices for fast queries
                CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON benchmark_runs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_runs_model ON benchmark_runs(model_name);
                CREATE INDEX IF NOT EXISTS idx_runs_gpu ON benchmark_runs(gpu_name);
                CREATE INDEX IF NOT EXISTS idx_runs_archived ON benchmark_runs(archived);
                CREATE INDEX IF NOT EXISTS idx_snapshots_run ON resource_snapshots(run_id);
                CREATE INDEX IF NOT EXISTS idx_aggregates_run ON metric_aggregates(run_id);
            """)

            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (1,))
            conn.commit()
            logger.info("Database schema initialized to version 1")

    def ingest_json_file(
        self,
        json_path: Path | str,
        compress: bool = True,
        delete_after: bool = False,
    ) -> str | None:
        """Ingest a JSON benchmark file into the database.

        Args:
            json_path: Path to JSON file.
            compress: Compress the full data.
            delete_after: Delete the JSON file after ingestion.

        Returns:
            Run ID if successful, None otherwise.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            logger.warning(f"File not found: {json_path}")
            return None

        try:
            with open(json_path) as f:
                data = json.load(f)

            run_id = self._ingest_data(data, compress)

            if delete_after and run_id:
                json_path.unlink()
                logger.info(f"Deleted ingested file: {json_path}")

            return run_id

        except Exception as e:
            logger.error(f"Failed to ingest {json_path}: {e}")
            return None

    def _ingest_data(self, data: dict[str, Any], compress: bool = True) -> str:
        """Ingest benchmark data dict into database.

        Args:
            data: Benchmark data dictionary.
            compress: Whether to compress the full data.

        Returns:
            Run ID.
        """
        # Extract run ID or generate one
        run_id = data.get("run_id") or data.get("timestamp", datetime.now().isoformat())

        # Extract summary fields
        timestamp = data.get("timestamp", datetime.now().isoformat())
        duration = data.get("duration_sec") or data.get("total_time_sec")
        model_name = data.get("model_name") or data.get("model", {}).get("name")
        gpu_name = data.get("gpu_name") or data.get("system", {}).get("gpu_name")
        cuda_version = data.get("cuda_version") or data.get("system", {}).get("cuda_version")
        pytorch_version = data.get("pytorch_version") or data.get("system", {}).get("pytorch_version")

        # Extract key metrics
        throughput = data.get("throughput") or data.get("samples_per_sec")
        latency = data.get("latency_ms") or data.get("avg_latency_ms")
        memory = data.get("memory_mb") or data.get("peak_memory_mb")
        power = data.get("power_watts") or data.get("avg_power_watts")
        temperature = data.get("temperature_c") or data.get("avg_temperature_c")

        with self._connection() as conn:
            # Insert summary
            conn.execute("""
                INSERT OR REPLACE INTO benchmark_runs
                (run_id, timestamp, duration_sec, model_name, gpu_name, cuda_version,
                 pytorch_version, throughput, latency_ms, memory_mb, power_watts,
                 temperature_c, compressed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, timestamp, duration, model_name, gpu_name, cuda_version,
                pytorch_version, throughput, latency, memory, power,
                temperature, 1 if compress else 0
            ))

            # Insert full data
            json_str = json.dumps(data, default=str)
            if compress:
                compressed = gzip.compress(json_str.encode("utf-8"))
                conn.execute("""
                    INSERT OR REPLACE INTO benchmark_data (run_id, data_compressed)
                    VALUES (?, ?)
                """, (run_id, compressed))
            else:
                conn.execute("""
                    INSERT OR REPLACE INTO benchmark_data (run_id, data_json)
                    VALUES (?, ?)
                """, (run_id, json_str))

            # Insert resource snapshots if present
            snapshots = data.get("resource_snapshots") or data.get("time_series", {}).get("snapshots", [])
            if snapshots:
                conn.executemany("""
                    INSERT INTO resource_snapshots
                    (run_id, timestamp_offset_ms, cpu_pct, memory_mb, gpu_util_pct, gpu_memory_mb,
                     disk_read_mb, disk_write_mb)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        run_id,
                        s.get("timestamp_offset_ms", 0),
                        s.get("cpu_pct") or s.get("process_cpu_percent"),
                        s.get("memory_mb") or s.get("process_memory_mb"),
                        s.get("gpu_util_pct") or s.get("gpu_utilization"),
                        s.get("gpu_memory_mb") or s.get("gpu_memory_used_mb"),
                        s.get("disk_read_mb"),
                        s.get("disk_write_mb"),
                    )
                    for s in snapshots
                ])

            # Insert aggregated metrics
            self._extract_aggregates(conn, run_id, data)

            conn.commit()
            logger.info(f"Ingested benchmark run: {run_id}")
            return run_id

    def _extract_aggregates(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        data: dict[str, Any],
    ) -> None:
        """Extract and store aggregated metrics.

        Args:
            conn: Database connection.
            run_id: Run identifier.
            data: Full benchmark data.
        """
        aggregates = []

        # Power metrics
        if "power_metrics" in data:
            pm = data["power_metrics"]
            for key, value in pm.items():
                if isinstance(value, (int, float)) and value is not None:
                    aggregates.append((run_id, "power", key, float(value), None))

        # Latent space metrics
        if "latent_space_metrics" in data:
            lm = data["latent_space_metrics"]
            for key, value in lm.items():
                if isinstance(value, (int, float)) and value is not None:
                    aggregates.append((run_id, "latent", key, float(value), None))

        # Compute metrics
        if "compute_metrics" in data:
            cm = data["compute_metrics"]
            for key, value in cm.items():
                if isinstance(value, (int, float)) and value is not None:
                    aggregates.append((run_id, "compute", key, float(value), None))

        # Diagnostic metrics
        if "diagnostic_metrics" in data:
            dm = data["diagnostic_metrics"]
            for key, value in dm.items():
                if isinstance(value, (int, float)) and value is not None:
                    aggregates.append((run_id, "diagnostic", key, float(value), None))

        # Resource metrics
        if "resource_usage" in data:
            ru = data["resource_usage"]
            for key, value in ru.items():
                if isinstance(value, (int, float)) and value is not None:
                    aggregates.append((run_id, "resource", key, float(value), None))

        if aggregates:
            conn.executemany("""
                INSERT OR REPLACE INTO metric_aggregates
                (run_id, category, metric_name, value, unit)
                VALUES (?, ?, ?, ?, ?)
            """, aggregates)

    def query_runs(
        self,
        model_name: str | None = None,
        gpu_name: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BenchmarkSummary]:
        """Query benchmark runs.

        Args:
            model_name: Filter by model name.
            gpu_name: Filter by GPU name.
            since: Only runs after this time.
            until: Only runs before this time.
            include_archived: Include archived runs.
            limit: Maximum results to return.
            offset: Offset for pagination.

        Returns:
            List of BenchmarkSummary objects.
        """
        conditions = []
        params: list[Any] = []

        if not include_archived:
            conditions.append("archived = 0")

        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)

        if gpu_name:
            conditions.append("gpu_name = ?")
            params.append(gpu_name)

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        with self._connection() as conn:
            # Note: where_clause only contains hardcoded column names from above,
            # all user values are parameterized - no SQL injection risk
            cur = conn.execute(f"""
                SELECT run_id, timestamp, duration_sec, model_name, gpu_name,
                       throughput, latency_ms, memory_mb, compressed
                FROM benchmark_runs
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, params)  # noqa: S608 - where_clause is safe (hardcoded column names only)

            return [
                BenchmarkSummary(
                    run_id=row["run_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    duration_sec=row["duration_sec"] or 0.0,
                    model_name=row["model_name"],
                    gpu_name=row["gpu_name"],
                    throughput=row["throughput"],
                    latency_ms=row["latency_ms"],
                    memory_mb=row["memory_mb"],
                    compressed=bool(row["compressed"]),
                )
                for row in cur.fetchall()
            ]

    def get_full_data(self, run_id: str) -> dict[str, Any] | None:
        """Get full benchmark data for a run.

        Args:
            run_id: Run identifier.

        Returns:
            Full benchmark data dict, or None if not found.
        """
        with self._connection() as conn:
            cur = conn.execute("""
                SELECT data_json, data_compressed
                FROM benchmark_data
                WHERE run_id = ?
            """, (run_id,))
            row = cur.fetchone()

            if not row:
                return None

            if row["data_compressed"]:
                json_str = gzip.decompress(row["data_compressed"]).decode("utf-8")
            else:
                json_str = row["data_json"]

            return json.loads(json_str) if json_str else None

    def get_resource_snapshots(
        self,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Get resource snapshots for a run.

        Args:
            run_id: Run identifier.

        Returns:
            List of snapshot dictionaries.
        """
        with self._connection() as conn:
            cur = conn.execute("""
                SELECT timestamp_offset_ms, cpu_pct, memory_mb, gpu_util_pct,
                       gpu_memory_mb, disk_read_mb, disk_write_mb
                FROM resource_snapshots
                WHERE run_id = ?
                ORDER BY timestamp_offset_ms
            """, (run_id,))

            return [dict(row) for row in cur.fetchall()]

    def get_metric_trend(
        self,
        metric_name: str,
        category: str,
        model_name: str | None = None,
        limit: int = 100,
    ) -> list[tuple[datetime, float]]:
        """Get trend for a specific metric over time.

        Args:
            metric_name: Name of the metric.
            category: Metric category (power, latent, compute, etc.).
            model_name: Optional model name filter.
            limit: Maximum data points.

        Returns:
            List of (timestamp, value) tuples.
        """
        params: list[Any] = [category, metric_name]
        model_filter = ""
        if model_name:
            model_filter = "AND r.model_name = ?"
            params.append(model_name)
        params.append(limit)

        with self._connection() as conn:
            # Note: model_filter is either empty or hardcoded "AND r.model_name = ?"
            cur = conn.execute(f"""
                SELECT r.timestamp, m.value
                FROM metric_aggregates m
                JOIN benchmark_runs r ON m.run_id = r.run_id
                WHERE m.category = ? AND m.metric_name = ?
                {model_filter}
                ORDER BY r.timestamp DESC
                LIMIT ?
            """, params)  # noqa: S608 - model_filter is safe (hardcoded or empty)

            return [
                (datetime.fromisoformat(row["timestamp"]), row["value"])
                for row in cur.fetchall()
            ]

    def archive_old_runs(self, days: int = 7) -> int:
        """Mark old runs as archived.

        Args:
            days: Archive runs older than this many days.

        Returns:
            Number of runs archived.
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self._connection() as conn:
            cur = conn.execute("""
                UPDATE benchmark_runs
                SET archived = 1
                WHERE timestamp < ? AND archived = 0
            """, (cutoff.isoformat(),))
            conn.commit()
            return cur.rowcount

    def delete_old_runs(self, days: int = 365) -> int:
        """Delete runs older than specified days.

        Args:
            days: Delete runs older than this many days.

        Returns:
            Number of runs deleted.
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self._connection() as conn:
            # Get run IDs to delete
            cur = conn.execute("""
                SELECT run_id FROM benchmark_runs
                WHERE timestamp < ?
            """, (cutoff.isoformat(),))
            run_ids = [row["run_id"] for row in cur.fetchall()]

            if not run_ids:
                return 0

            # Delete from all tables using parameterized placeholders
            # Note: placeholders is safe - just "?,?,?" repeated for number of run_ids
            placeholders = ",".join("?" * len(run_ids))
            conn.execute(
                f"DELETE FROM resource_snapshots WHERE run_id IN ({placeholders})",  # noqa: S608
                run_ids
            )
            conn.execute(
                f"DELETE FROM metric_aggregates WHERE run_id IN ({placeholders})",  # noqa: S608
                run_ids
            )
            conn.execute(
                f"DELETE FROM benchmark_data WHERE run_id IN ({placeholders})",  # noqa: S608
                run_ids
            )
            conn.execute(
                f"DELETE FROM benchmark_runs WHERE run_id IN ({placeholders})",  # noqa: S608
                run_ids
            )
            conn.commit()

            # Vacuum to reclaim space
            conn.execute("VACUUM")

            logger.info(f"Deleted {len(run_ids)} old benchmark runs")
            return len(run_ids)

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics.
        """
        with self._connection() as conn:
            stats: dict[str, Any] = {}

            # Total runs
            cur = conn.execute("SELECT COUNT(*) as count FROM benchmark_runs")
            stats["total_runs"] = cur.fetchone()["count"]

            # Archived runs
            cur = conn.execute("SELECT COUNT(*) as count FROM benchmark_runs WHERE archived = 1")
            stats["archived_runs"] = cur.fetchone()["count"]

            # Active runs
            stats["active_runs"] = stats["total_runs"] - stats["archived_runs"]

            # Compressed runs
            cur = conn.execute("SELECT COUNT(*) as count FROM benchmark_runs WHERE compressed = 1")
            stats["compressed_runs"] = cur.fetchone()["count"]

            # Total snapshots
            cur = conn.execute("SELECT COUNT(*) as count FROM resource_snapshots")
            stats["total_snapshots"] = cur.fetchone()["count"]

            # Database file size
            stats["database_size_mb"] = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

            # Date range
            cur = conn.execute("SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM benchmark_runs")
            row = cur.fetchone()
            stats["oldest_run"] = row["oldest"]
            stats["newest_run"] = row["newest"]

            return stats

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class ArchiveManager:
    """Manages automatic archival and cleanup of benchmark history.

    Provides automated management of benchmark runs:
    - Ingests new JSON files into database
    - Archives old runs
    - Deletes expired runs
    - Maintains recent file count

    Example:
        >>> manager = ArchiveManager("benchmark_results/history.db", "benchmark_results/history")
        >>> manager.run()  # Process all pending tasks
    """

    def __init__(
        self,
        db_path: Path | str,
        history_dir: Path | str,
        max_recent_files: int = 50,
        archive_after_days: int = 7,
        retention_days: int = 365,
    ) -> None:
        """Initialize archive manager.

        Args:
            db_path: Path to SQLite database.
            history_dir: Directory containing JSON history files.
            max_recent_files: Keep this many recent files.
            archive_after_days: Archive runs older than this.
            retention_days: Delete runs older than this.
        """
        self.db = BenchmarkHistoryDB(db_path)
        self.history_dir = Path(history_dir)
        self.max_recent_files = max_recent_files
        self.archive_after_days = archive_after_days
        self.retention_days = retention_days

    def run(self) -> dict[str, int]:
        """Run all archive management tasks.

        Returns:
            Summary of actions taken.
        """
        self.db.initialize()

        results = {
            "ingested": self.ingest_new_files(),
            "archived": self.db.archive_old_runs(self.archive_after_days),
            "deleted": 0,
            "trimmed": self.trim_recent_files(),
        }

        if self.retention_days > 0:
            results["deleted"] = self.db.delete_old_runs(self.retention_days)

        logger.info(f"Archive management complete: {results}")
        return results

    def ingest_new_files(self) -> int:
        """Ingest all JSON files not yet in database.

        Returns:
            Number of files ingested.
        """
        if not self.history_dir.exists():
            return 0

        ingested = 0
        for json_file in sorted(self.history_dir.glob("*.json")):
            # Check if already ingested by run_id
            run_id = json_file.stem  # Use filename as run_id
            existing = self.db.query_runs(limit=1)
            if any(r.run_id == run_id for r in self.db.query_runs(limit=1000)):
                continue

            if self.db.ingest_json_file(json_file, compress=True):
                ingested += 1

        return ingested

    def trim_recent_files(self) -> int:
        """Remove oldest JSON files if over limit.

        Returns:
            Number of files removed.
        """
        if not self.history_dir.exists():
            return 0

        json_files = sorted(
            self.history_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        removed = 0
        for old_file in json_files[self.max_recent_files:]:
            try:
                old_file.unlink()
                removed += 1
                logger.info(f"Removed old history file: {old_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_file}: {e}")

        return removed

    def get_status(self) -> dict[str, Any]:
        """Get status of archive system.

        Returns:
            Status dictionary.
        """
        db_stats = self.db.get_statistics()

        # Count JSON files
        json_count = 0
        json_size_mb = 0.0
        if self.history_dir.exists():
            json_files = list(self.history_dir.glob("*.json"))
            json_count = len(json_files)
            json_size_mb = sum(f.stat().st_size for f in json_files) / (1024 * 1024)

        return {
            **db_stats,
            "recent_json_files": json_count,
            "recent_json_size_mb": json_size_mb,
            "max_recent_files": self.max_recent_files,
            "archive_after_days": self.archive_after_days,
            "retention_days": self.retention_days,
        }

    def close(self) -> None:
        """Close database connection."""
        self.db.close()
