"""Test suite for CogSynDelta logging infrastructure.

Tests the silent error logging system defined in ADR-0004 and
specs/silent-error-logging/. Validates:
- LogConfig configuration and environment variable parsing
- SkipMetrics thread-safe counting
- JSON log entry format and file operations
- DEV vs DEBUG behavior differences
- Graceful handling of logging failures

Why comprehensive tests: The logging infrastructure is critical for
observability into graceful degradation points. Silent failures without
visibility could mask serious issues over time.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any

import pytest

from cogsyndelta.core.logging_config import (
    LogConfig,
    LogLevel,
    SkipLogEntry,
    SkipMetrics,
    configure_logging,
    get_config,
    get_logger,
    get_skip_metrics,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# LogLevel Tests
# =============================================================================


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_dev_level_value(self) -> None:
        """Verify DEV level has correct string value."""
        assert LogLevel.DEV.value == "DEV"

    def test_debug_level_value(self) -> None:
        """Verify DEBUG level has correct string value."""
        assert LogLevel.DEBUG.value == "DEBUG"

    def test_levels_are_distinct(self) -> None:
        """Verify DEV and DEBUG are distinct levels."""
        assert LogLevel.DEV != LogLevel.DEBUG


# =============================================================================
# SkipLogEntry Tests
# =============================================================================


class TestSkipLogEntry:
    """Tests for SkipLogEntry dataclass."""

    def test_entry_creation_with_required_fields(self) -> None:
        """Test creating entry with only required fields."""
        entry = SkipLogEntry(
            timestamp="2026-01-19T12:00:00+00:00",
            level="DEBUG",
            operation="test_operation",
            category="test_category",
            message="Test message",
        )
        assert entry.timestamp == "2026-01-19T12:00:00+00:00"
        assert entry.level == "DEBUG"
        assert entry.operation == "test_operation"
        assert entry.category == "test_category"
        assert entry.message == "Test message"
        assert entry.memory_id is None
        assert entry.exception_type is None
        assert entry.context is None

    def test_entry_creation_with_all_fields(self) -> None:
        """Test creating entry with all optional fields."""
        entry = SkipLogEntry(
            timestamp="2026-01-19T12:00:00+00:00",
            level="DEBUG",
            operation="test_operation",
            category="test_category",
            message="Test message",
            memory_id="mem_123",
            exception_type="KeyError",
            context={"key": "value"},
        )
        assert entry.memory_id == "mem_123"
        assert entry.exception_type == "KeyError"
        assert entry.context == {"key": "value"}

    def test_to_json_excludes_none_values(self) -> None:
        """Verify JSON output excludes None fields for cleaner output."""
        entry = SkipLogEntry(
            timestamp="2026-01-19T12:00:00+00:00",
            level="DEBUG",
            operation="test_op",
            category="test_cat",
            message="Test",
        )
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert "memory_id" not in parsed
        assert "exception_type" not in parsed
        assert "context" not in parsed

    def test_to_json_includes_non_none_values(self) -> None:
        """Verify JSON output includes all non-None fields."""
        entry = SkipLogEntry(
            timestamp="2026-01-19T12:00:00+00:00",
            level="DEBUG",
            operation="test_op",
            category="test_cat",
            message="Test",
            memory_id="abc123",
            context={"extra": "data"},
        )
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert parsed["memory_id"] == "abc123"
        assert parsed["context"] == {"extra": "data"}

    def test_to_json_is_valid_json(self) -> None:
        """Verify to_json produces parseable JSON."""
        entry = SkipLogEntry(
            timestamp="2026-01-19T12:00:00+00:00",
            level="DEBUG",
            operation="test_op",
            category="test_cat",
            message="Message with \"quotes\" and 'apostrophes'",
        )
        json_str = entry.to_json()
        parsed = json.loads(json_str)  # Should not raise
        assert parsed["message"] == "Message with \"quotes\" and 'apostrophes'"


# =============================================================================
# SkipMetrics Tests
# =============================================================================


class TestSkipMetrics:
    """Tests for SkipMetrics aggregation class."""

    def test_initial_counts_empty(self) -> None:
        """Verify new SkipMetrics has empty counts."""
        metrics = SkipMetrics()
        assert metrics.counts == {}
        assert metrics.total() == 0

    def test_increment_single_category(self) -> None:
        """Test incrementing a single category."""
        metrics = SkipMetrics()
        metrics.increment("test_category")
        assert metrics.counts["test_category"] == 1
        metrics.increment("test_category")
        assert metrics.counts["test_category"] == 2

    def test_increment_multiple_categories(self) -> None:
        """Test incrementing multiple categories."""
        metrics = SkipMetrics()
        metrics.increment("cat_a")
        metrics.increment("cat_b")
        metrics.increment("cat_a")
        assert metrics.counts == {"cat_a": 2, "cat_b": 1}

    def test_increment_with_count(self) -> None:
        """Test incrementing by more than 1."""
        metrics = SkipMetrics()
        metrics.increment("bulk_skip", count=10)
        assert metrics.counts["bulk_skip"] == 10

    def test_total_returns_sum(self) -> None:
        """Verify total() returns sum of all categories."""
        metrics = SkipMetrics()
        metrics.increment("a", count=5)
        metrics.increment("b", count=3)
        metrics.increment("c", count=2)
        assert metrics.total() == 10

    def test_reset_clears_and_returns_counts(self) -> None:
        """Verify reset() clears metrics and returns final counts."""
        metrics = SkipMetrics()
        metrics.increment("test", count=5)
        final = metrics.reset()
        assert final == {"test": 5}
        assert metrics.counts == {}
        assert metrics.total() == 0

    def test_summary_empty_metrics(self) -> None:
        """Test summary for empty metrics."""
        metrics = SkipMetrics()
        summary = metrics.summary()
        assert "No graceful degradation skips recorded" in summary

    def test_summary_with_counts(self) -> None:
        """Test summary includes all categories and total."""
        metrics = SkipMetrics()
        metrics.increment("temporal_continuity_missing", count=3)
        metrics.increment("archive_corrupted", count=1)
        summary = metrics.summary()
        assert "temporal_continuity_missing: 3" in summary
        assert "archive_corrupted: 1" in summary
        assert "TOTAL: 4" in summary

    def test_thread_safety(self) -> None:
        """Test concurrent increments don't lose counts.

        Why: Multiple threads may trigger skips simultaneously.
        The counter must be thread-safe to avoid losing skip counts.
        """
        metrics = SkipMetrics()
        num_threads = 10
        increments_per_thread = 100

        def increment_many() -> None:
            for _ in range(increments_per_thread):
                metrics.increment("concurrent_category")

        threads = [Thread(target=increment_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Counter uses thread-safe operations
        expected = num_threads * increments_per_thread
        assert metrics.counts["concurrent_category"] == expected


# =============================================================================
# LogConfig Tests
# =============================================================================


class TestLogConfig:
    """Tests for LogConfig configuration class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LogConfig()
        assert config.level == LogLevel.DEV
        assert config.log_dir == Path("logs")
        assert config.log_file == "memory_debug.jsonl"
        assert config.console_output is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LogConfig(
            level=LogLevel.DEBUG,
            log_dir=Path("/custom/logs"),
            log_file="custom.jsonl",
            console_output=True,
        )
        assert config.level == LogLevel.DEBUG
        assert config.log_dir == Path("/custom/logs")
        assert config.log_file == "custom.jsonl"
        assert config.console_output is True

    def test_log_path_property(self) -> None:
        """Test log_path combines dir and file."""
        config = LogConfig(log_dir=Path("/var/log"), log_file="test.jsonl")
        assert config.log_path == Path("/var/log/test.jsonl")


# =============================================================================
# configure_logging Tests
# =============================================================================


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_default_returns_dev_level(self) -> None:
        """Test default configuration uses DEV level."""
        # Clear any env var
        os.environ.pop("COGSYNDELTA_LOG_LEVEL", None)
        config = configure_logging()
        assert config.level == LogLevel.DEV

    def test_explicit_level_argument(self) -> None:
        """Test explicit level argument overrides default."""
        config = configure_logging(level=LogLevel.DEBUG)
        assert config.level == LogLevel.DEBUG

    def test_string_level_argument(self) -> None:
        """Test string level argument works."""
        config = configure_logging(level="DEBUG")
        assert config.level == LogLevel.DEBUG
        config = configure_logging(level="dev")
        assert config.level == LogLevel.DEV

    def test_env_var_override(self) -> None:
        """Test COGSYNDELTA_LOG_LEVEL env var is respected."""
        os.environ["COGSYNDELTA_LOG_LEVEL"] = "DEBUG"
        try:
            config = configure_logging()
            assert config.level == LogLevel.DEBUG
        finally:
            os.environ.pop("COGSYNDELTA_LOG_LEVEL", None)

    def test_explicit_level_beats_env_var(self) -> None:
        """Test explicit level argument beats env var."""
        os.environ["COGSYNDELTA_LOG_LEVEL"] = "DEBUG"
        try:
            config = configure_logging(level=LogLevel.DEV)
            assert config.level == LogLevel.DEV
        finally:
            os.environ.pop("COGSYNDELTA_LOG_LEVEL", None)

    def test_custom_log_dir(self) -> None:
        """Test custom log directory configuration."""
        config = configure_logging(log_dir=Path("/custom/path"))
        assert config.log_dir == Path("/custom/path")

    def test_string_log_dir_converted_to_path(self) -> None:
        """Test string log_dir is converted to Path."""
        config = configure_logging(log_dir="/string/path")
        assert config.log_dir == Path("/string/path")


# =============================================================================
# CogSynDeltaLogger Tests
# =============================================================================


class TestCogSynDeltaLogger:
    """Tests for CogSynDeltaLogger class."""

    @pytest.fixture(autouse=True)
    def reset_global_state(self) -> Any:
        """Reset global metrics and config before each test."""
        get_skip_metrics().reset()
        configure_logging(level=LogLevel.DEV)
        yield
        get_skip_metrics().reset()
        os.environ.pop("COGSYNDELTA_LOG_LEVEL", None)

    def test_skip_increments_metrics_at_dev_level(self) -> None:
        """Test skip() always increments metrics even at DEV level."""
        configure_logging(level=LogLevel.DEV)
        logger = get_logger("test.module")

        logger.skip(
            category="test_skip",
            operation="test_op",
            message="Test skip message",
        )

        metrics = get_skip_metrics()
        assert metrics.counts["test_skip"] == 1

    def test_skip_increments_metrics_at_debug_level(self) -> None:
        """Test skip() increments metrics at DEBUG level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(level=LogLevel.DEBUG, log_dir=Path(tmpdir))
            logger = get_logger("test.module")

            logger.skip(
                category="debug_skip",
                operation="test_op",
                message="Test skip message",
            )

            metrics = get_skip_metrics()
            assert metrics.counts["debug_skip"] == 1

    def test_skip_writes_jsonl_at_debug_level(self) -> None:
        """Test skip() writes to JSONL file at DEBUG level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = configure_logging(level=LogLevel.DEBUG, log_dir=Path(tmpdir))
            logger = get_logger("test.module")

            logger.skip(
                category="jsonl_test",
                operation="test_op",
                message="Test message for JSONL",
                memory_id="mem_456",
            )

            # Verify file was written
            log_path = config.log_path
            assert log_path.exists()

            with open(log_path) as f:
                line = f.readline()
                entry = json.loads(line)

            assert entry["category"] == "jsonl_test"
            assert entry["operation"] == "test_op"
            assert entry["message"] == "Test message for JSONL"
            assert entry["memory_id"] == "mem_456"
            assert "timestamp" in entry

    def test_skip_does_not_write_at_dev_level(self) -> None:
        """Test skip() does NOT write files at DEV level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = configure_logging(level=LogLevel.DEV, log_dir=Path(tmpdir))
            logger = get_logger("test.module")

            logger.skip(
                category="dev_skip",
                operation="test_op",
                message="Should not be written",
            )

            # Verify no file was written
            assert not config.log_path.exists()

    def test_skip_with_exception(self) -> None:
        """Test skip() handles exception parameter correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = configure_logging(level=LogLevel.DEBUG, log_dir=Path(tmpdir))
            logger = get_logger("test.module")

            try:
                raise KeyError("test key")
            except KeyError as e:
                logger.skip(
                    category="exception_test",
                    operation="test_op",
                    message="Test with exception",
                    exception=e,
                )

            with open(config.log_path) as f:
                entry = json.loads(f.readline())

            assert entry["exception_type"] == "KeyError"

    def test_skip_with_context(self) -> None:
        """Test skip() includes context in log entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = configure_logging(level=LogLevel.DEBUG, log_dir=Path(tmpdir))
            logger = get_logger("test.module")

            logger.skip(
                category="context_test",
                operation="test_op",
                message="Test with context",
                context={"tiers_checked": ["active", "short"], "retry_count": 3},
            )

            with open(config.log_path) as f:
                entry = json.loads(f.readline())

            assert entry["context"]["tiers_checked"] == ["active", "short"]
            assert entry["context"]["retry_count"] == 3

    def test_multiple_skips_append_to_file(self) -> None:
        """Test multiple skip() calls append to same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = configure_logging(level=LogLevel.DEBUG, log_dir=Path(tmpdir))
            logger = get_logger("test.module")

            logger.skip(category="first", operation="op1", message="First skip")
            logger.skip(category="second", operation="op2", message="Second skip")
            logger.skip(category="third", operation="op3", message="Third skip")

            with open(config.log_path) as f:
                lines = f.readlines()

            assert len(lines) == 3
            assert json.loads(lines[0])["category"] == "first"
            assert json.loads(lines[1])["category"] == "second"
            assert json.loads(lines[2])["category"] == "third"

    def test_logger_standard_methods(self) -> None:
        """Test standard logging methods (debug, info, warning, error) exist."""
        logger = get_logger("test.module")

        # These should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoggingIntegration:
    """Integration tests for logging system."""

    @pytest.fixture(autouse=True)
    def reset_global_state(self) -> Any:
        """Reset global metrics before each test."""
        get_skip_metrics().reset()
        yield
        get_skip_metrics().reset()

    def test_global_metrics_shared(self) -> None:
        """Test all loggers share global metrics singleton."""
        logger1 = get_logger("module.a")
        logger2 = get_logger("module.b")

        logger1.skip(category="shared_test", operation="op1", message="From logger1")
        logger2.skip(category="shared_test", operation="op2", message="From logger2")

        metrics = get_skip_metrics()
        assert metrics.counts["shared_test"] == 2

    def test_get_config_initializes_if_needed(self) -> None:
        """Test get_config() creates default config if not configured."""
        # This should work without explicit configure_logging call
        config = get_config()
        assert config is not None
        assert isinstance(config.level, LogLevel)

    def test_logging_failures_dont_crash(self) -> None:
        """Test logging failures are handled gracefully.

        Why: Per constitution, logging itself must use graceful degradation.
        A logging failure should never crash the application.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = configure_logging(level=LogLevel.DEBUG, log_dir=Path(tmpdir))
            logger = get_logger("test.module")

            # Make log path unwritable
            log_path = config.log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch()
            log_path.chmod(0o000)

            try:
                # This should NOT raise despite unwritable file
                logger.skip(
                    category="unwritable_test",
                    operation="test_op",
                    message="Should not crash",
                )
                # Metrics should still be incremented
                assert get_skip_metrics().counts["unwritable_test"] == 1
            finally:
                log_path.chmod(0o644)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_category_allowed(self) -> None:
        """Test empty category string is technically allowed."""
        metrics = SkipMetrics()
        metrics.increment("")
        assert metrics.counts[""] == 1

    def test_unicode_in_message(self) -> None:
        """Test Unicode characters in log messages."""
        entry = SkipLogEntry(
            timestamp="2026-01-19T12:00:00+00:00",
            level="DEBUG",
            operation="test_op",
            category="unicode_test",
            message="Test with émojis 🎉 and symbols ™",
        )
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert "émojis 🎉" in parsed["message"]

    def test_large_context_dict(self) -> None:
        """Test handling of large context dictionaries."""
        large_context = {f"key_{i}": f"value_{i}" for i in range(100)}
        entry = SkipLogEntry(
            timestamp="2026-01-19T12:00:00+00:00",
            level="DEBUG",
            operation="test_op",
            category="large_context",
            message="Test with large context",
            context=large_context,
        )
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert len(parsed["context"]) == 100

    def test_nested_context_dict(self) -> None:
        """Test handling of nested context dictionaries."""
        nested_context = {
            "level1": {
                "level2": {
                    "level3": "deep_value",
                },
            },
        }
        entry = SkipLogEntry(
            timestamp="2026-01-19T12:00:00+00:00",
            level="DEBUG",
            operation="test_op",
            category="nested_context",
            message="Test with nested context",
            context=nested_context,
        )
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert parsed["context"]["level1"]["level2"]["level3"] == "deep_value"
