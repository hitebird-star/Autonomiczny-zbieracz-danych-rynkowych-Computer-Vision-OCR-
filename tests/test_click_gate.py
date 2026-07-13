from __future__ import annotations

import unittest

from PIL import Image

from scanner.detection import TrackedShop
from scanner.models import ScanStatus, ShopScan
from scanner.pipeline import AutonomousMarketLoop, CaptureOutcome


class _Detector:
    def detect(self, image, *, screen_offset):
        return []


class _Coverage:
    def __init__(self) -> None:
        self.dup_floors: list[int | None] = []

    def should_skip_click(self, pos, *, dup_floor=None):
        self.dup_floors.append(dup_floor)
        return True

    def cell_of(self, pos):
        return (0, 0)

    def dups_in_cell(self, cell):
        return 2


class _Pipeline:
    close_blocked = False
    stall_blocked = False
    stall_count = 0
    current_position = (400, 700)

    def __init__(self) -> None:
        self.captured: list[str] = []

    def capture(self, track):
        self.captured.append(track.track_id)
        return CaptureOutcome(ShopScan("scan", status=ScanStatus.CAPTURED))


class ClickGateTests(unittest.TestCase):
    def _loop(self, tracker, pipeline):
        loop = AutonomousMarketLoop(
            _Detector(),
            tracker,
            pipeline,
            object(),
            lambda: (Image.new("RGB", (10, 10)), (0, 0)),
        )
        coverage = _Coverage()
        loop.set_coverage_map(coverage)
        loop._test_coverage = coverage
        return loop

    def test_fresh_target_beats_exhausted_cell(self) -> None:
        """Komórkowy dedup nie może wycinać pierwszego kliknięcia w nowy cel."""

        fresh = TrackedShop("fresh-shop", (300, 400), attempts=0)

        class Tracker:
            calls = 0

            def update(self, candidates):
                self.calls += 1
                return [fresh] if self.calls == 1 else []

            def next_unvisited(self, visible):
                return visible[0] if visible else None

        pipeline = _Pipeline()
        outcomes = self._loop(Tracker(), pipeline).scan_current_view()

        self.assertEqual(pipeline.captured, ["fresh-shop"])
        self.assertEqual(len(outcomes), 1)

    def test_retry_is_still_skipped_in_exhausted_cell(self) -> None:
        """Anti-reskan zostaje aktywny wyłącznie dla realnej ponownej próby."""

        retry = TrackedShop("retry-shop", (300, 400), attempts=1)

        class Tracker:
            def update(self, candidates):
                return [retry]

            def next_unvisited(self, visible):
                return visible[0]

        pipeline = _Pipeline()
        loop = self._loop(Tracker(), pipeline)
        outcomes = loop.scan_current_view()

        self.assertEqual(pipeline.captured, [])
        self.assertEqual(outcomes, [])
        self.assertEqual(loop._test_coverage.dup_floors, [4])


if __name__ == "__main__":
    unittest.main()
