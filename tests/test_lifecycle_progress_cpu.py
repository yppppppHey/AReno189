"""Deterministic unit tests for the structured lifecycle progress SDK.

These tests import directly from source files to avoid the torch dependency
in areno/api/__init__.py, which re-exports heavy modules.

Covers:
- Stage and EventKind enums
- ProgressEvent immutability, serialization, and the failed_at_stage property
- DisabledReporter (default, no-op)
- TTYReporter output content
- JSONLReporter round-trip via file
- EventDispatcher fan-out
- create_progress_reporter factory (disabled, text, jsonl, invalid mode)
- JSONLReporter with thread safety
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

# ---------------------------------------------------------------------------
# Direct file imports to bypass areno.api.__init__.py (which triggers torch)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_models = _load_module(
    "areno.api.lifecycle.models",
    _ROOT / "areno" / "api" / "lifecycle" / "models.py",
)
_reporter = _load_module(
    "areno.api.lifecycle.reporter",
    _ROOT / "areno" / "api" / "lifecycle" / "reporter.py",
)

# Re-export for convenience
Stage = _models.Stage
EventKind = _models.EventKind
ProgressEvent = _models.ProgressEvent
DisabledReporter = _reporter.DisabledReporter
TTYReporter = _reporter.TTYReporter
JSONLReporter = _reporter.JSONLReporter
EventDispatcher = _reporter.EventDispatcher
ProgressReporter = _reporter.ProgressReporter
create_progress_reporter = _reporter.create_progress_reporter


# ---------------------------------------------------------------------------
# Stage / EventKind / ProgressEvent
# ---------------------------------------------------------------------------


class StageEnumTest(unittest.TestCase):
    def test_all_required_stages_present(self):
        required = {
            Stage.CREATED,
            Stage.VALIDATING,
            Stage.INITIALIZING,
            Stage.RUNNING,
            Stage.SUCCEEDED,
            Stage.FAILED,
            Stage.SMOKING,
            Stage.TUNING,
            Stage.SAVING,
            Stage.LOADING,
            Stage.PROBING,
            Stage.PROFILING,
        }
        present = {Stage(s) for s in Stage}
        self.assertEqual(required, present)

    def test_train_specific_stages(self):
        self.assertIn(Stage.SMOKING, Stage)
        self.assertIn(Stage.TUNING, Stage)
        self.assertIn(Stage.SAVING, Stage)

    def test_serve_specific_stages(self):
        self.assertIn(Stage.LOADING, Stage)
        self.assertIn(Stage.PROBING, Stage)

    def test_profile_specific_stages(self):
        self.assertIn(Stage.PROFILING, Stage)

    def test_stage_values_are_lowercase(self):
        self.assertEqual(Stage.RUNNING.value, "running")
        self.assertEqual(Stage.SUCCEEDED.value, "succeeded")
        self.assertEqual(Stage.FAILED.value, "failed")


class EventKindEnumTest(unittest.TestCase):
    def test_two_values(self):
        self.assertEqual(len(EventKind), 2)
        self.assertIn(EventKind.TIMELINE, EventKind)
        self.assertIn(EventKind.DIAGNOSTIC, EventKind)

    def test_kind_values_are_lowercase(self):
        self.assertEqual(EventKind.TIMELINE.value, "timeline")
        self.assertEqual(EventKind.DIAGNOSTIC.value, "diagnostic")


class ProgressEventTest(unittest.TestCase):
    def test_immutability(self):
        event = ProgressEvent(stage=Stage.RUNNING, step=42, message="hello")
        with self.assertRaises(FrozenInstanceError):
            event.stage = Stage.SUCCEEDED  # type: ignore[assignment]

    def test_defaults(self):
        event = ProgressEvent(stage=Stage.CREATED)
        self.assertEqual(event.kind, EventKind.TIMELINE)
        self.assertIsNone(event.step)
        self.assertEqual(event.message, "")
        self.assertEqual(event.extra, {})

    def test_as_json_round_trip(self):
        event = ProgressEvent(
            stage=Stage.RUNNING,
            kind=EventKind.TIMELINE,
            step=7,
            message="training step 7",
            extra={"loss": 0.42, "active_stage": "running"},
        )
        obj = json.loads(event.as_json())
        self.assertEqual(obj["stage"], "running")
        self.assertEqual(obj["kind"], "timeline")
        self.assertEqual(obj["step"], 7)
        self.assertEqual(obj["message"], "training step 7")
        self.assertEqual(obj["extra"]["loss"], 0.42)
        self.assertEqual(obj["extra"]["active_stage"], "running")

    def test_failed_at_stage(self):
        event = ProgressEvent(
            stage=Stage.FAILED,
            extra={"failed_at_stage": "INITIALIZING"},
        )
        # Stage values are lowercase: "initializing"
        # "INITIALIZING" is not a valid Stage value, so failed_at_stage returns None
        self.assertIsNone(event.failed_at_stage)

    def test_failed_at_stage_lowercase(self):
        event = ProgressEvent(
            stage=Stage.FAILED,
            extra={"failed_at_stage": "initializing"},
        )
        self.assertEqual(event.failed_at_stage, Stage.INITIALIZING)

    def test_failed_at_stage_missing(self):
        event = ProgressEvent(stage=Stage.FAILED, extra={})
        self.assertIsNone(event.failed_at_stage)

    def test_str_format(self):
        event = ProgressEvent(stage=Stage.INITIALIZING, step=0, message="loading tokenizer")
        # __str__ uses .value which is lowercase
        self.assertIn("initializing", str(event))
        self.assertIn("loading tokenizer", str(event))


# ---------------------------------------------------------------------------
# DisabledReporter
# ---------------------------------------------------------------------------


class DisabledReporterTest(unittest.TestCase):
    def test_no_op(self):
        reporter = DisabledReporter()
        event = ProgressEvent(stage=Stage.RUNNING)
        reporter.report(event)  # should not raise
        reporter.open()  # should not raise
        reporter.close()  # should not raise


# ---------------------------------------------------------------------------
# TTYReporter
# ---------------------------------------------------------------------------


class TTYReporterTest(unittest.TestCase):
    def test_non_tty_output(self):
        stream = io.StringIO()
        reporter = TTYReporter(stream=stream)
        reporter.open()
        reporter.report(ProgressEvent(stage=Stage.VALIDATING, message="validating"))
        reporter.report(ProgressEvent(stage=Stage.INITIALIZING, step=0, message="initializing"))
        reporter.close()

        lines = stream.getvalue().splitlines()
        # Stage values are lowercase
        self.assertTrue(any("validating" in line for line in lines))
        self.assertTrue(any("initializing" in line for line in lines))

    def test_tty_inline_output(self):
        tty = io.StringIO()
        tty.isatty = lambda: True
        reporter = TTYReporter(stream=tty)
        reporter.open()
        reporter.report(ProgressEvent(stage=Stage.RUNNING, step=1, message="step 1"))
        reporter.report(ProgressEvent(stage=Stage.RUNNING, step=2, message="step 2"))
        reporter.close()

        output = tty.getvalue()
        # TTY mode uses \r for in-place editing; at least the last line should be visible
        self.assertIn("step 2", output)


# ---------------------------------------------------------------------------
# JSONLReporter
# ---------------------------------------------------------------------------


class JSONLReporterTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
            path = tmp.name

        reporter = JSONLReporter(path)
        reporter.open()
        reporter.report(ProgressEvent(stage=Stage.VALIDATING, message="validating"))
        reporter.report(ProgressEvent(stage=Stage.INITIALIZING, step=0, message="init"))
        reporter.report(ProgressEvent(stage=Stage.SUCCEEDED, step=1, message="done"))
        reporter.close()

        lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)

        ev1 = json.loads(lines[0])
        self.assertEqual(ev1["stage"], "validating")
        self.assertIsNone(ev1["step"])

        ev2 = json.loads(lines[1])
        self.assertEqual(ev2["stage"], "initializing")
        self.assertEqual(ev2["step"], 0)

        ev3 = json.loads(lines[2])
        self.assertEqual(ev3["stage"], "succeeded")
        self.assertEqual(ev3["step"], 1)

    def test_append_mode(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
            path = tmp.name

        # Pre-write a line to test append behavior
        Path(path).write_text('{"stage":"pre"}\n', encoding="utf-8")

        reporter = JSONLReporter(path)
        reporter.open()  # opens in 'w' mode (truncate on open)
        reporter.report(ProgressEvent(stage=Stage.RUNNING, message="after open"))
        reporter.close()

        lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["stage"], "running")


# ---------------------------------------------------------------------------
# EventDispatcher
# ---------------------------------------------------------------------------


class EventDispatcherTest(unittest.TestCase):
    def test_fan_out(self):
        tty_stream = io.StringIO()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
            jsonl_path = tmp.name

        tty_reporter = TTYReporter(stream=tty_stream)
        jsonl_reporter = JSONLReporter(jsonl_path)
        dispatcher = EventDispatcher([tty_reporter, jsonl_reporter])

        dispatcher.open()
        dispatcher.report(ProgressEvent(stage=Stage.RUNNING, step=1, message="dispatched"))
        dispatcher.close()

        # Check TTY output (stage value is lowercase)
        self.assertIn("running", tty_stream.getvalue())

        # Check JSONL output
        lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["stage"], "running")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class FactoryTest(unittest.TestCase):
    def test_disabled(self):
        reporter = create_progress_reporter("disabled")
        self.assertIsInstance(reporter, DisabledReporter)

    def test_text(self):
        reporter = create_progress_reporter("text")
        self.assertIsInstance(reporter, TTYReporter)

    def test_jsonl_requires_output_path(self):
        with self.assertRaises(ValueError):
            create_progress_reporter("jsonl")

    def test_jsonl_with_path(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
            path = tmp.name
        reporter = create_progress_reporter("jsonl", output_path=path)
        self.assertIsInstance(reporter, JSONLReporter)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            create_progress_reporter("banana")


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class JSONLReporterThreadTest(unittest.TestCase):
    def test_concurrent_reports(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
            path = tmp.name

        reporter = JSONLReporter(path)
        reporter.open()

        errors: list[Exception] = []

        def emit_n(n: int) -> None:
            try:
                for i in range(n):
                    reporter.report(
                        ProgressEvent(stage=Stage.RUNNING, step=i, message=f"thread event {i}")
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=emit_n, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")

        lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 200)
        reporter.close()


if __name__ == "__main__":
    unittest.main()