from __future__ import annotations

import unittest
from types import SimpleNamespace

from scanner.models import ItemObservation, ScanStatus, ShopScan
from scanner.pipeline import GameCapturePipeline


class FakeRepository:
    def __init__(self) -> None:
        self.events = []
        self.saved = []

    def append_event(self, scan_id, event, **data) -> None:
        self.events.append((scan_id, event, data))

    def save_manifest(self, scan) -> None:
        self.saved.append(scan.status)


class FakeTracker:
    def __init__(self) -> None:
        self.visited = []

    def mark_visited(self, track) -> None:
        self.visited.append(track)


class PartialCaptureFinalizationTests(unittest.TestCase):
    def test_partial_capture_becomes_captured_and_queued(self) -> None:
        pipeline = object.__new__(GameCapturePipeline)
        pipeline.repository = FakeRepository()
        pipeline.tracker = FakeTracker()
        submitted = []
        pipeline.analysis_queue = SimpleNamespace(submit=lambda scan_id: submitted.append(scan_id))
        scan = ShopScan(
            "partial-1",
            status=ScanStatus.CAPTURING,
            occupied_slots=3,
            captured_slots=1,
            slots={
                1: ItemObservation(
                    slot=1,
                    row=0,
                    column=1,
                    status=ScanStatus.CAPTURED,
                )
            },
        )
        track = object()

        outcome = pipeline._finalize_partial_capture(
            scan,
            track,
            reason="RuntimeError:game_focus_lost",
        )

        self.assertIsNotNone(outcome)
        self.assertEqual(scan.status, ScanStatus.QUEUED)
        self.assertEqual(submitted, ["partial-1"])
        self.assertEqual(pipeline.tracker.visited, [track])
        self.assertTrue(
            any(event == "partial_capture_finalized" for _, event, _ in pipeline.repository.events)
        )


if __name__ == "__main__":
    unittest.main()
