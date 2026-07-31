"""ProgressReporter: emit ProgressEvent to TTY, JSONL file, or discard.

This module provides three output strategies:

- ``DisabledReporter`` — the default, does nothing (backward compatible).
- ``TTYReporter`` — prints one-line events to stderr using in-place
  carriage-return for TTY, or plain lines for non-TTY.
- ``JSONLReporter`` — appends JSON Lines records to a file, suitable for
  downstream parsing (dashboard, post-run analysis).

All reporters implement the same ``ProgressReporter`` protocol so callers
inject one strategy and never need to know which is active.
"""

from __future__ import annotations

import os
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO

from .models import EventKind, ProgressEvent, Stage


class ProgressReporter(ABC):
    """Protocol for consuming structured lifecycle progress events.

    Subclasses must implement ``report``.  Optional ``open``/``close`` hooks
    let them allocate or release external resources.
    """

    def open(self) -> None:
        """Prepare the reporter (no-op by default)."""

    def close(self) -> None:
        """Flush and release the reporter (no-op by default)."""

    @abstractmethod
    def report(self, event: ProgressEvent) -> None:
        """Handle a single progress event.

        This method is called synchronously from the thread that owns the job.
        Implementations must not block for long periods.
        """


# ---------------------------------------------------------------------------
# No-op: the default reporter that does nothing.
# ---------------------------------------------------------------------------


class DisabledReporter(ProgressReporter):
    """Silent reporter that discards all events (default behavior)."""

    def report(self, event: ProgressEvent) -> None:
        pass


# ---------------------------------------------------------------------------
# TTY / line-based reporter
# ---------------------------------------------------------------------------


class TTYReporter(ProgressReporter):
    """Display progress events on stderr.

    When stderr is a TTY, events are rendered inline using carriage-return so
    only one line is visible at a time.  When stderr is a pipe (non-TTY),
    events are emitted as separate lines (useful for log tailing).

    Args:
        stream: File-like object for output. Defaults to ``sys.stderr``.
    """

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._last_stage: Stage | None = None
        self._last_len = 0

    def report(self, event: ProgressEvent) -> None:
        text = str(event)
        is_tty = hasattr(self._stream, "isatty") and self._stream.isatty()

        with self._lock:
            if is_tty:
                # In-place: clear previous line, write new one, keep cursor.
                # Clear to end of line (CSI K) and move cursor back (CSI g).
                padding = " " * max(0, self._last_len - len(text))
                self._stream.write(f"\r{text}{padding}\r")
                self._stream.flush()
                self._last_len = max(self._last_len, len(text))
            else:
                # Line-based: print a full line with newline.
                self._stream.write(text + "\n")
                self._stream.flush()
                self._last_len = len(text)


# ---------------------------------------------------------------------------
# JSONL file reporter
# ---------------------------------------------------------------------------


class JSONLReporter(ProgressReporter):
    """Append progress events as JSON Lines to a file.

    Each call to ``report`` writes exactly one JSON object followed by a
    newline.  A lock ensures thread safety when multiple threads emit events.

    Args:
        path: Path to the JSONL file.  The file is created if it does not exist.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Open in append mode; the file will be truncated only if we explicitly
        # create it via open().  Append is safer for long-running jobs that may
        # be restarted (different PID writes a new line).
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def open(self) -> None:
        # Truncate on first open to avoid stale data from a previous run.
        self._fh = open(self._path, "w", encoding="utf-8")  # noqa: SIM115

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()

    def report(self, event: ProgressEvent) -> None:
        with self._lock:
            self._fh.write(event.as_json() + "\n")
            self._fh.flush()


# ---------------------------------------------------------------------------
# Event dispatcher: fan-out to one or more reporters
# ---------------------------------------------------------------------------


class EventDispatcher(ProgressReporter):
    """Dispatch a single event to multiple reporters.

    Useful when you want both a TTY display AND a JSONL file simultaneously.
    """

    def __init__(self, reporters: list[ProgressReporter]) -> None:
        self._reporters = reporters

    def open(self) -> None:
        for r in self._reporters:
            r.open()

    def close(self) -> None:
        for r in self._reporters:
            r.close()

    def report(self, event: ProgressEvent) -> None:
        for r in self._reporters:
            r.report(event)


# ---------------------------------------------------------------------------
# Builder / factory
# ---------------------------------------------------------------------------


def create_progress_reporter(
    mode: str = "disabled",
    output_path: str | None = None,
) -> ProgressReporter:
    """Build a ``ProgressReporter`` from a mode string.

    Args:
        mode: One of ``"disabled"``, ``"text"``, or ``"jsonl"``.
            - ``"disabled"`` — no-op (default, backward compatible).
            - ``"text"`` — TTY reporter on stderr.
            - ``"jsonl"`` — JSONL reporter writing to ``output_path``.
        output_path: Required when ``mode="jsonl"``.  The file path for JSONL output.

    Returns:
        A ``ProgressReporter`` instance.

    Raises:
        ValueError: If ``mode`` is unknown or ``output_path`` is missing for ``"jsonl"``.
    """

    if mode == "disabled":
        return DisabledReporter()
    if mode == "text":
        return TTYReporter()
    if mode == "jsonl":
        if output_path is None:
            raise ValueError("--progress jsonl requires --progress-output to specify the output path")
        return JSONLReporter(output_path)
    raise ValueError(f"Unknown --progress mode: {mode!r}. Choose from: disabled, text, jsonl")