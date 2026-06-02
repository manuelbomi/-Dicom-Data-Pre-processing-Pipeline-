"""Structured logging with progress tracking and metrics collection.

Provides:
  - Structured JSON logging for production deployments
  - Rich console progress bars for interactive use
  - Metrics collection and aggregation (throughput, latency, error rates)
  - Stage-level timing and resource tracking
  - Integration-ready metric export (CloudWatch, Prometheus format)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, TextIO


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production environments.

    Produces one JSON object per log line, compatible with CloudWatch
    Logs, ELK Stack, and other log aggregation systems.
    """

    def __init__(
        self,
        service_name: str = "dicom-pipeline",
        include_timestamp: bool = True,
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        if self.include_timestamp:
            log_entry["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        # Include extra fields
        for key in ("stage", "sample_key", "duration_sec", "throughput", "worker_id"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for interactive use."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = record.getMessage()

        # Add extra context if available
        extras = []
        if hasattr(record, "stage"):
            extras.append(f"stage={record.stage}")
        if hasattr(record, "throughput"):
            extras.append(f"{record.throughput:.1f} img/s")
        if hasattr(record, "duration_sec"):
            extras.append(f"{record.duration_sec:.2f}s")

        extra_str = f" [{', '.join(extras)}]" if extras else ""

        return (
            f"{color}{timestamp} {record.levelname:8s}{self.RESET} "
            f"{record.name:30s} {msg}{extra_str}"
        )


def setup_logging(
    level: str = "INFO",
    structured: bool = False,
    log_file: Optional[str] = None,
    service_name: str = "dicom-pipeline",
) -> None:
    """Configure logging for the pipeline.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        structured: Use JSON structured logging (for production).
        log_file: Optional file path for log output.
        service_name: Service name for structured logs.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    if structured:
        console.setFormatter(StructuredFormatter(service_name=service_name))
    else:
        console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter(service_name=service_name))
        root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------------

@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Aggregated summary of a metric over time."""

    name: str
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    last: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)

    def record(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.last = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 4),
            "min": round(self.min, 4) if self.min != float("inf") else None,
            "max": round(self.max, 4) if self.max != float("-inf") else None,
            "last": round(self.last, 4),
        }


class MetricsCollector:
    """Collect and aggregate pipeline metrics.

    Tracks throughput, latency, error rates, and custom metrics across
    pipeline stages. Provides both real-time access and periodic export.

    Example::

        metrics = MetricsCollector()
        metrics.record("images_processed", 1, tags={"stage": "preprocess"})
        metrics.record("processing_time_ms", 45.2, tags={"stage": "preprocess"})

        with metrics.timer("pixel_processing"):
            process_pixels(image)

        summary = metrics.get_summary("processing_time_ms")
        print(f"Avg processing time: {summary.mean:.1f}ms")
    """

    def __init__(self) -> None:
        self._summaries: Dict[str, MetricSummary] = {}
        self._history: List[MetricPoint] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._max_history = 100_000

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            tags: Optional key-value tags for filtering.
        """
        if name not in self._summaries:
            self._summaries[name] = MetricSummary(name=name)
        self._summaries[name].record(value)

        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
        )
        if len(self._history) < self._max_history:
            self._history.append(point)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[name] += amount

    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        return self._counters.get(name, 0)

    def get_summary(self, name: str) -> Optional[MetricSummary]:
        """Get aggregated summary for a metric."""
        return self._summaries.get(name)

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> Generator[None, None, None]:
        """Context manager to time a block of code.

        Args:
            name: Metric name for the timing.
            tags: Optional tags.

        Example::

            with metrics.timer("stage.preprocess"):
                preprocess(image)
        """
        t0 = time.monotonic()
        try:
            yield
        finally:
            duration_ms = (time.monotonic() - t0) * 1000
            self.record(name, duration_ms, tags)

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get all metric summaries as a dictionary."""
        result = {}
        for name, summary in self._summaries.items():
            result[name] = summary.to_dict()
        for name, count in self._counters.items():
            result[f"counter.{name}"] = {"name": name, "count": count}
        return result

    def export_json(self, path: str) -> None:
        """Export all metrics to a JSON file."""
        data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summaries": self.get_all_summaries(),
            "counters": dict(self._counters),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self) -> None:
        """Reset all metrics."""
        self._summaries.clear()
        self._history.clear()
        self._counters.clear()


class ProgressTracker:
    """Track and display pipeline progress.

    Provides both programmatic progress tracking and optional console
    progress bar display.

    Args:
        total: Total number of items to process.
        description: Display description.
        update_interval_sec: Minimum time between display updates.

    Example::

        tracker = ProgressTracker(total=10000, description="Processing")
        for item in items:
            process(item)
            tracker.update(1)
        tracker.close()
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        update_interval_sec: float = 1.0,
    ) -> None:
        self.total = total
        self.description = description
        self.update_interval = update_interval_sec

        self._processed = 0
        self._failed = 0
        self._start_time = time.monotonic()
        self._last_update = 0.0
        self._logger = logging.getLogger("progress")

    def update(self, n: int = 1, failed: int = 0) -> None:
        """Update progress by n items."""
        self._processed += n
        self._failed += failed

        now = time.monotonic()
        if now - self._last_update >= self.update_interval:
            self._display()
            self._last_update = now

    def _display(self) -> None:
        """Display current progress."""
        elapsed = time.monotonic() - self._start_time
        rate = self._processed / max(elapsed, 0.001)
        pct = self._processed / max(self.total, 1) * 100

        remaining = self.total - self._processed
        eta = remaining / max(rate, 0.001)

        bar_width = 30
        filled = int(bar_width * self._processed / max(self.total, 1))
        bar = "=" * filled + ">" + " " * (bar_width - filled - 1)

        msg = (
            f"\r{self.description}: [{bar}] {pct:5.1f}% "
            f"({self._processed}/{self.total}) "
            f"{rate:.1f} img/s "
            f"ETA: {_format_time(eta)} "
            f"Errors: {self._failed}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self) -> None:
        """Finalize progress tracking."""
        self._display()
        sys.stdout.write("\n")
        elapsed = time.monotonic() - self._start_time
        rate = self._processed / max(elapsed, 0.001)
        self._logger.info(
            "%s complete: %d items in %s (%.1f img/s, %d errors)",
            self.description,
            self._processed,
            _format_time(elapsed),
            rate,
            self._failed,
        )

    @property
    def elapsed_sec(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def throughput(self) -> float:
        return self._processed / max(self.elapsed_sec, 0.001)


def _format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"
