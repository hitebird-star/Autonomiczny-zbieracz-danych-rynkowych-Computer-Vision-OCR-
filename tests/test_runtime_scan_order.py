from __future__ import annotations

import unittest

from PIL import Image

from scanner.detection import ShopCandidate, ShopTracker
from scanner.models import ScanStatus, ShopScan
from scanner.pipeline import AutonomousMarketLoop, CaptureOutcome


class RuntimeScanOrderTests(unittest.TestCase):
    def test_auto_preserves_detector_hybrid_order_for_current_view(self) -> None:
        """Live loop must not re-sort candidates by a second forward/distance pass.

        The detector already returns runtime order: likely_false targets are pushed
        down, then candidates are ordered by distance. A later pure-distance reorder
        would pull the close false target back to the front and recreate the
        "bot tunnels forward / ignores side shops" behaviour seen in live scans.
        """

        safe_target = (400, 460)
        close_false_target = (500, 300)

        class Detector:
            def detect(self, image, *, screen_offset):
                return [
                    ShopCandidate(safe_target, safe_target, 20, 160.0, 1.0, False),
                    ShopCandidate(close_false_target, close_false_target, 20, 100.0, -1.0, True),
                    ShopCandidate((560, 300), (560, 300), 20, 160.0, 0.5, False),
                    ShopCandidate((400, 480), (400, 480), 20, 180.0, 0.5, False),
                ]

            def mask_image(self, image):
                return Image.new("L", image.size)

        class Pipeline:
            close_blocked = False
            stall_count = 0
            stall_blocked = False

            def capture(self, track):
                self.captured_position = track.position
                return CaptureOutcome(
                    ShopScan("scan-1", status=ScanStatus.CAPTURED)
                )

        pipeline = Pipeline()
        loop = AutonomousMarketLoop(
            Detector(),
            ShopTracker(),
            pipeline,
            object(),
            lambda: (Image.new("RGB", (800, 600)), (0, 0)),
        )

        loop.scan_current_view(max_shops=1)

        self.assertEqual(pipeline.captured_position, safe_target)


if __name__ == "__main__":
    unittest.main()
