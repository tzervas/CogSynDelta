"""Configurable logging infrastructure for CogSynDelta.

This module provides a configurable logging system with two modes:
- DEV (default): Counts silent skips without console spam, aggregates metrics
- DEBUG: Full verbose logging with structured JSON output for analysis

The logging system is designed to instrument graceful degradation points
per ADR-0004, enabling visibility into silent failures without impacting
system stability or performance.

Why this approach:
    Silent failures are intentional (graceful degradation), but we need
    visibility for debugging and analysis. DEV mode gives aggregated counts,
    DEBUG gives full context. Structured JSON enables automated analysis.

Example:
    >>> from cogsyndelta.core.logging_config import get_logger, get_skip_metrics
    >>> logger = get_logger(__name__)
    >>> logger.skip("temporal_continuity_missing", memory_id="abc123")
    >>> metrics = get_skip_metrics()
    >>> print(metrics.counts)
    {'temporal_continuity_missing': 1}
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import MutableMapping

__all__ = [
    "LogLevel",
    "LogConfig",
    "SkipMetrics",
    "SkipLogEntry",
    "CogSynDeltaLogger",
    "get_logger",
    "get_skip_metrics",
    "configure_logging",
]


class LogLevel(Enum):
    """Log levels for CogSynDelta logging.

    DEV: Default level. Counts skips silently, no console output for individual skips.
    DEBUG: Verbose level. Logs every skip with full context to JSONL file.
    """

    DEV = "DEV"
    DEBUG = "DEBUG"


@dataclass
class SkipLogEntry:
    """Structured log entry for a graceful degradation skip.

    Attributes:
        timestamp: ISO 8601 timestamp of the skip event.
        level: Log level at time of skip.
        operation: Name of the operation that encountered the skip.
        category: Category of skip for metric aggregation.
        message: Human-readable description of what was skipped.
        memory_id: Optional memory ID involved in the skip.
        exception_type: Optional exception type that triggered the skip.
        context: Optional additional context as key-value pairs.
    """

    timestamp: str
    level: str
    operation: str
    category: str
    message: str
    memory_id: str | None = None
    exception_type: str | None = None
    context: dict[str, Any] | None = None

    def to_json(self) -> str:
        """Convert entry to JSON string for JSONL output.

        Returns:
            JSON string representation of the log entry.
        """
        data = asdict(self)
        # Remove None values for cleaner output
        data = {k: v for k, v in data.items() if v is not None}
        return json.dumps(data, default=str)


class SkipMetrics:
    """Aggregated metrics for graceful degradation skips.

    Thread-safe counter for skip events, queryable at any time.
    Supports reset for testing and session boundaries.

    Attributes:
        counts: Counter dict mapping category to skip count.

    Example:
        >>> metrics = SkipMetrics()
        >>> metrics.increment("temporal_continuity_missing")
        >>> metrics.increment("temporal_continuity_missing")
        >>> metrics.counts["temporal_continuity_missing"]
        2
    """

    def __init__(self) -> None:
        """Initialize empty metrics counter."""
        self._counts: Counter[str] = Counter()

    @property
    def counts(self) -> dict[str, int]:
        """Get current skip counts by category.

        Returns:
            Dict mapping category names to skip counts.
        """
        return dict(self._counts)

    def increment(self, category: str, count: int = 1) -> None:
        """Increment skip count for a category.

        Args:
            category: Skip category name (e.g., "temporal_continuity_missing").
            count: Number to increment by, defaults to 1.
        """
        self._counts[category] += count

    def reset(self) -> dict[str, int]:
        """Reset all counts and return final values.

        Useful for session boundaries or test isolation.

        Returns:
            Dict of counts before reset.
        """
        final = self.counts
        self._counts.clear()
        return final

    def total(self) -> int:
        """Get total skip count across all categories.

        Returns:
            Sum of all category counts.
        """
        return sum(self._counts.values())

    def summary(self) -> str:
        """Get human-readable summary of skip metrics.

        Returns:
            Formatted string with counts per category.
        """
        if not self._counts:
            return "No graceful degradation skips recorded."
        lines = ["Graceful degradation skip summary:"]
        for category, count in sorted(self._counts.items()):
            lines.append(f"  {category}: {count}")
        lines.append(f"  TOTAL: {self.total()}")
        return "\n".join(lines)


# Global singleton instances
_skip_metrics = SkipMetrics()
_log_config: LogConfig | None = None


@dataclass
class LogConfig:
    """Configuration for CogSynDelta logging.

    Attributes:
        level: Log level (DEV or DEBUG).
        log_dir: Directory for log files.
        log_file: Name of the JSONL log file.
        console_output: Whether to also output to console at DEBUG level.
    """

    level: LogLevel = field(default_factory=lambda: LogLevel.DEV)
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    log_file: str = "memory_debug.jsonl"
    console_output: bool = False

    def __post_init__(self) -> None:
        """Ensure log directory exists."""
        if self.level == LogLevel.DEBUG:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                # Fall back gracefully if we can't create log dir
                pass

    @property
    def log_path(self) -> Path:
        """Get full path to log file.

        Returns:
            Path to JSONL log file.
        """
        return self.log_dir / self.log_file


def configure_logging(
    level: LogLevel | str | None = None,
    log_dir: Path | str | None = None,
    console_output: bool = False,
) -> LogConfig:
    """Configure CogSynDelta logging.

    Can be called multiple times to reconfigure. Reads from environment
    variable COGSYNDELTA_LOG_LEVEL if level not specified.

    Args:
        level: Log level (DEV or DEBUG). Defaults to env var or DEV.
        log_dir: Directory for log files. Defaults to ./logs.
        console_output: Whether to output to console at DEBUG level.

    Returns:
        The configured LogConfig instance.

    Example:
        >>> configure_logging(level=LogLevel.DEBUG)
        >>> configure_logging(level="DEBUG")  # String also works
    """
    global _log_config

    # Determine level from argument, env var, or default
    if level is None:
        env_level = os.environ.get("COGSYNDELTA_LOG_LEVEL", "DEV").upper()
        level = LogLevel.DEBUG if env_level == "DEBUG" else LogLevel.DEV
    elif isinstance(level, str):
        level = LogLevel.DEBUG if level.upper() == "DEBUG" else LogLevel.DEV

    # Determine log directory
    if log_dir is None:
        log_dir = Path("logs")
    elif isinstance(log_dir, str):
        log_dir = Path(log_dir)

    _log_config = LogConfig(
        level=level,
        log_dir=log_dir,
        console_output=console_output,
    )

    return _log_config


def get_config() -> LogConfig:
    """Get current logging configuration, initializing if needed.

    Returns:
        Current LogConfig instance.
    """
    global _log_config
    if _log_config is None:
        _log_config = configure_logging()
    return _log_config


def get_skip_metrics() -> SkipMetrics:
    """Get the global skip metrics instance.

    Returns:
        Global SkipMetrics singleton for querying skip counts.

    Example:
        >>> metrics = get_skip_metrics()
        >>> print(metrics.summary())
    """
    return _skip_metrics


class CogSynDeltaLogger:
    """Logger wrapper with skip tracking for graceful degradation.

    Wraps standard logging with additional skip() method that:
    - Always increments skip metrics (regardless of level)
    - At DEBUG level, writes structured JSON to log file
    - At DEV level, skips silently (metrics only)

    Attributes:
        name: Logger name (usually module __name__).

    Example:
        >>> logger = CogSynDeltaLogger("cogsyndelta.memory")
        >>> logger.skip(
        ...     category="temporal_continuity_missing",
        ...     operation="_ensure_temporal_continuity",
        ...     message="Memory not found in any tier",
        ...     memory_id="abc123",
        ... )
    """

    def __init__(self, name: str) -> None:
        """Initialize logger with given name.

        Args:
            name: Logger name, typically __name__ of the module.
        """
        self.name = name
        self._logger = logging.getLogger(name)

    def skip(
        self,
        category: str,
        operation: str,
        message: str,
        memory_id: str | None = None,
        exception: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log a graceful degradation skip event.

        Always increments skip metrics. At DEBUG level, writes full
        structured log entry to JSONL file.

        Args:
            category: Skip category for metric aggregation.
            operation: Name of the operation that skipped.
            message: Human-readable description.
            memory_id: Optional memory ID involved.
            exception: Optional exception that triggered the skip.
            context: Optional additional context.

        Example:
            >>> logger.skip(
            ...     category="archive_corrupted",
            ...     operation="_load_from_archive",
            ...     message="Skipping corrupted archive entry",
            ...     exception=pickle_error,
            ...     context={"file": "archive_001.pkl"},
            ... )
        """
        # Always increment metrics
        _skip_metrics.increment(category)

        config = get_config()
        if config.level != LogLevel.DEBUG:
            return

        # Create structured log entry
        entry = SkipLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level="DEBUG",
            operation=operation,
            category=category,
            message=message,
            memory_id=memory_id,
            exception_type=type(exception).__name__ if exception else None,
            context=context,
        )

        # Write to JSONL file
        try:
            with open(config.log_path, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")
        except OSError:
            # Graceful degradation for logging itself
            pass

        # Optional console output
        if config.console_output:
            self._logger.debug(
                "[SKIP] %s: %s (memory_id=%s)",
                category,
                message,
                memory_id,
            )

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            msg: Message format string.
            *args: Format arguments.
            **kwargs: Additional logging kwargs.
        """
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message.

        Args:
            msg: Message format string.
            *args: Format arguments.
            **kwargs: Additional logging kwargs.
        """
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            msg: Message format string.
            *args: Format arguments.
            **kwargs: Additional logging kwargs.
        """
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message.

        Args:
            msg: Message format string.
            *args: Format arguments.
            **kwargs: Additional logging kwargs.
        """
        self._logger.error(msg, *args, **kwargs)


def get_logger(name: str) -> CogSynDeltaLogger:
    """Get a CogSynDeltaLogger for the given module.

    Args:
        name: Logger name, typically __name__.

    Returns:
        CogSynDeltaLogger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.skip("my_category", "my_op", "Something was skipped")
    """
    return CogSynDeltaLogger(name)
