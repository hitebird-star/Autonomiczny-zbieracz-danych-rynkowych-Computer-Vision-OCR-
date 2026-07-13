from __future__ import annotations

import unittest

from scanner.models import ScanStatus, ShopScan
from scanner.pipeline import AutonomousMarketLoop, CaptureOutcome


class _Coverage:
    def __init__(self) -> None:
        self.path_calls = 0

    def path_to_next_target(self, pos, **kwargs):
        self.path_calls += 1
        return ((0, 0), (1, 0))

    def cell_center(self, cell):
        return (float(cell[0] * 20 + 10), float(cell[1] * 20 + 10))

    def all_done(self, *, dup_floor=None):
        return False

    def blocked_ahead(self, pos, target):
        return False

    def cell_of(self, point):
        return (0, 0)

    def scans_in_cell(self, cell):
        return 1

    def dups_in_cell(self, cell):
        return 0

    def record_block(self, point, reason):
        raise AssertionError(f"unexpected block: {reason}")


class _Pipeline:
    current_position = (10, 10)


class CoverageDriveTests(unittest.TestCase):
    def test_scans_current_view_before_first_coverage_movement(self) -> None:
        calls: list[str] = []

        class Loop(AutonomousMarketLoop):
            def scan_current_view(self, *, max_shops=0):
                calls.append("scan")
                return []

            def _drive_toward_target(self, target):
                calls.append("move")
                return True

        loop = Loop(None, None, _Pipeline(), None, lambda: None)
        coverage = _Coverage()
        loop.set_coverage_map(coverage)

        # Zatrzymaj pętlę po pierwszym pełnym przebiegu bez uzależniania testu
        # od wewnętrznej polityki saturacji mapy.
        def done_after_first_path(*, dup_floor=None):
            return coverage.path_calls >= 1

        coverage.all_done = done_after_first_path
        loop.run(())

        self.assertEqual(calls, ["scan"])

    def test_max_shops_stops_coverage_drive_before_movement(self) -> None:
        class Loop(AutonomousMarketLoop):
            def __init__(self):
                super().__init__(None, None, _Pipeline(), None, lambda: None)
                self.scan_limits: list[int] = []
                self.moves = 0

            def scan_current_view(self, *, max_shops=0):
                self.scan_limits.append(max_shops)
                return [CaptureOutcome(ShopScan("captured", status=ScanStatus.CAPTURED))]

            def _drive_toward_target(self, target):
                self.moves += 1
                return True

        loop = Loop()
        loop.set_coverage_map(_Coverage())

        outcomes = loop.run((), max_shops=1)

        self.assertEqual(loop.scan_limits, [1])
        self.assertEqual(loop.moves, 0)
        self.assertEqual(len(outcomes), 1)


if __name__ == "__main__":
    unittest.main()
