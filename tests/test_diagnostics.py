from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scanner.diagnostics import AutoDiagnostics
from scanner.detection import ShopCandidate, TrackedShop
from scanner.navigation import MovementStep


class DiagnosticsTests(unittest.TestCase):
    def test_writes_scene_mask_overlay_and_json_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            diagnostics = AutoDiagnostics(temp)
            candidate = ShopCandidate((110, 220), (10, 20), 42, 15.5)
            track = TrackedShop("shop-00001", (110, 220))
            legacy = TrackedShop("shop-legacy", (130, 230))

            diagnostics.record_detection(
                Image.new("RGB", (80, 60), "gray"),
                Image.new("L", (80, 60), 255),
                [candidate],
                [track],
                track,
                screen_offset=(100, 200),
                legacy_pick=legacy,
            )
            diagnostics.record_interaction(
                {"name": "click", "attempt": 1, "target": [110, 220]}
            )

            class Outcome:
                duplicate = False

                class Scan:
                    scan_id = "scan-1"
                    status = type("Status", (), {"value": "captured"})()
                    seller = "Kocur"
                    error = None

                scan = Scan()

            diagnostics.record_capture(track, Outcome())
            diagnostics.record_movement(
                index=1,
                total=4,
                step=MovementStep("d", 0.6, 0.9, 0, 0, "horizontal"),
            )

            self.assertTrue(
                (diagnostics.directory / "round_001_scene.png").exists()
            )
            self.assertTrue(
                (diagnostics.directory / "round_001_mask.png").exists()
            )
            self.assertTrue(
                (diagnostics.directory / "round_001_overlay.png").exists()
            )
            records = [
                json.loads(line)
                for line in diagnostics.events_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(records[0]["event"], "detection")
            self.assertEqual(records[0]["selected"], "shop-00001")
            self.assertEqual(records[0]["legacy_pick"], "shop-legacy")
            self.assertTrue(records[0]["ranking_changed"])
            self.assertIsNone(
                records[0]["candidates"][0]["hybrid_score"]
            )
            self.assertFalse(
                records[0]["candidates"][0]["likely_false"]
            )
            self.assertFalse(
                records[0]["candidates"][0]["legacy_selected"]
            )
            self.assertFalse(records[0]["candidates"][0]["track_visited"])
            self.assertFalse(records[0]["candidates"][0]["track_failed"])
            self.assertEqual(records[0]["candidates"][0]["track_attempts"], 0)
            self.assertFalse(records[0]["candidates"][0]["track_fingerprinted"])
            self.assertEqual(records[1]["event"], "interaction")
            self.assertEqual(records[2]["event"], "capture_outcome")
            self.assertEqual(records[3]["event"], "movement")
            self.assertEqual(records[3]["step_kind"], "horizontal")

    def test_can_cap_saved_detection_images_while_keeping_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            diagnostics = AutoDiagnostics(temp, max_image_rounds=1)
            candidate = ShopCandidate((110, 220), (10, 20), 42, 15.5)
            track = TrackedShop("shop-00001", (110, 220))

            for _ in range(3):
                diagnostics.record_detection(
                    Image.new("RGB", (80, 60), "gray"),
                    Image.new("L", (80, 60), 255),
                    [candidate],
                    [track],
                    track,
                    screen_offset=(100, 200),
                )

            self.assertTrue((diagnostics.directory / "round_001_scene.png").exists())
            self.assertFalse((diagnostics.directory / "round_002_scene.png").exists())
            records = [
                json.loads(line)
                for line in diagnostics.events_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(records), 3)
            self.assertTrue(records[0]["images_saved"])
            self.assertFalse(records[1]["images_saved"])
            self.assertEqual(records[1]["files"], {})


if __name__ == "__main__":
    unittest.main()
