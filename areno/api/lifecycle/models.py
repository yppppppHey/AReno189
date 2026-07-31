"""Stage and progress event definitions for the AReno lifecycle SDK.

This module defines the data types that represent the structured lifecycle of
a long-running AReno job — training, serving, or performance profiling.  The
type system is deliberately small and focused:

- ``Stage`` enumerates the canonical phases of every job.
- ``ProgressEvent`` is the atomic event emitted at each phase transition or
  milestone.
- ``EventKind`` distinguishes discovery events from the stage pipeline so that
  downstream consumers can separate setup telemetry from the core timeline.

All types use ``dataclass(frozen=True)`` for immutable, JSON-safe
interchange with the dashboard state consumer.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class Stage(str, enum.Enum):
    """Canonical stages of an AReno lifecycle job.

    Every long-running operation (train, serve, profile) is decomposed into
    these stages.  A stage is *active* from the moment it is emitted until the
    next stage supersedes it.
    """

    # --- Common stages ---
    CREATED = "created"
    """Job object has been constructed; no heavy work started yet."""
    VALIDATING = "validating"
    """CLI/SDK options are being validated; no model or dataset loaded."""
    INITIALIZING = "initializing"
    """Loading tokenizer, model weights, allocating backend workers."""
    RUNNING = "running"
    """Core work is in progress (rollout/train loop, serve loop, profile loop)."""
    SUCCEEDED = "succeeded"
    """Job completed successfully; all stages were emitted in order."""
    FAILED = "failed"
    """Job failed.  The ``failed_at_stage`` field on the event identifies
    where the pipeline stopped."""

    # --- Train-specific stages ---
    SMOKING = "smoking"
    """Smoke inference or smoke train probing is running."""
    TUNING = "tuning"
    """Auto-tune (memory / parameter probing) is running."""
    SAVING = "saving"
    """Checkpoint save is in progress."""

    # --- Serve-specific stages ---
    LOADING = "loading"
    """Model loading and CUDA-graph capture during serve startup."""
    PROBING = "probing"
    """Health-check probe is running after serve startup."""

    # --- Profile-specific stages ---
    PROFILING = "profiling"
    """Profiling capture (py-spy, nsight, metric aggregation) is running."""

    def __repr__(self) -> str:
        return f"Stage.{self.name}"


# ---------------------------------------------------------------------------
# Event kind: a very thin filter so consumers can separate "discovery" from
# the real stage timeline.
# ---------------------------------------------------------------------------


class EventKind(str, enum.Enum):
    """Distinguishes the origin of a progress event.

    ``TIMELINE`` events belong to the canonical stage pipeline and should drive
    the in-place TTY display.  ``DIAGNOSTIC`` events provide extra telemetry
    (warnings, deprecations) without advancing the stage.
    """

    TIMELINE = "timeline"
    DIAGNOSTIC = "diagnostic"


# ---------------------------------------------------------------------------
# The atomic event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressEvent:
    """A single point in the structured progress timeline.

    Every event carries:

    - ``stage`` — which phase the job is now in (or was in when the event was
      captured).
    - ``kind`` — whether this is a core-timeline event or a diagnostic side
      note.
    - ``step`` — an integer monotonic counter from the caller (``None`` for
      events that do not track a step, e.g. failure events before the first
      step).
    - ``message`` — a human-readable summary.  The first 120 characters are
      suitable for TTY one-liners; the full string is suitable for JSONL logs.
    - ``extra`` — free-form key/value pairs for structured consumers (dashboard,
      TensorBoard sidecar, post-run analysis).  Values must be JSON-serializable.
    - ``timestamp`` — wall-clock time in seconds since the Unix epoch, recorded
      by the caller at emission time.

    Failure contracts: when a job fails, the last emitted event has
    ``stage=Stage.FAILED`` and ``extra.failed_at_stage`` contains the name of
    the stage that was active when the failure occurred.  This allows
    post-mortem to identify the last completed stage without exposing
    training samples or hidden internals.
    """

    stage: Stage
    kind: EventKind = EventKind.TIMELINE
    step: int | None = None
    message: str = ""
    extra: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def failed_at_stage(self) -> Stage | None:
        """If this event represents a failure, return the last successfully
        completed stage from ``extra``; otherwise ``None``."""
        name = self.extra.get("failed_at_stage")
        if name is None:
            return None
        try:
            return Stage(name)
        except ValueError:
            return None

    def as_json(self) -> str:
        """Serialize to a JSON Lines record for non-TTY consumption."""

        import json

        return json.dumps(
            {
                "stage": self.stage.value,
                "kind": self.kind.value,
                "step": self.step,
                "message": self.message,
                "extra": self.extra,
                "timestamp": self.timestamp,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def __str__(self) -> str:
        """One-line display suitable for in-place TTY rendering."""

        step_str = f" step={self.step}" if self.step is not None else ""
        msg = self.message[:120] if self.message else ""
        return f"[{self.stage.value}]{step_str} {msg}"