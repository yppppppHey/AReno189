"""Structured lifecycle / progress events for long-running AReno jobs.

Public API:

- ``Stage`` — canonical job phases (CREATED, INITIALIZING, RUNNING, …).
- ``EventKind`` — TIMELINE vs DIAGNOSTIC event origin.
- ``ProgressEvent`` — atomic event with stage, step, message, extra.
- ``ProgressReporter`` — protocol / base class for event consumers.
- ``DisabledReporter``, ``TTYReporter``, ``JSONLReporter``, ``EventDispatcher``
  — concrete reporter implementations.
- ``create_progress_reporter(mode, output_path)`` — factory function.

Default mode is ``disabled`` so existing code is unchanged.  Enable with
``--progress text`` or ``--progress jsonl --progress-output /path/to/events.jsonl``
on the ``areno train`` / ``areno serve`` CLI.
"""

from .models import EventKind, ProgressEvent, Stage
from .reporter import (
    DisabledReporter,
    EventDispatcher,
    JSONLReporter,
    ProgressReporter,
    TTYReporter,
    create_progress_reporter,
)

__all__ = [
    "EventKind",
    "ProgressEvent",
    "Stage",
    "ProgressReporter",
    "DisabledReporter",
    "TTYReporter",
    "JSONLReporter",
    "EventDispatcher",
    "create_progress_reporter",
]