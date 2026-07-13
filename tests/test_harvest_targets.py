from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scanner.detection.harvest_targets import _label, harvest_session


class HarvestTargetsTests(unittest.TestCase):
    def test_captured_is_confirmed_real_shop(self) -> None:
        self.assertEqual(_label("captured", None), "real")
        self.assertEqual(_label("queued", None), "real")

    def test_harvests_captured_outcome_as_real_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "20260626_151214"
            session.mkdir()
            Image.new("RGB", (160, 160), (20, 30, 40)).save(
                session / "round_001_scene.png"
            )
            events = [
                {
                    "event": "detection",
                    "files": {"scene": "round_001_scene.png"},
                    "candidates": [
                        {
                            "track_id": "shop-00001",
                            "local": [80, 80],
                            "area": 42,
                            "hybrid_score": 0.2,
                        }
                    ],
                },
                {
                    "event": "capture_outcome",
                    "track_id": "shop-00001",
                    "status": "captured",
                    "reason": None,
                },
            ]
            (session / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )

            rows = harvest_session(session, root / "dataset", 96)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "real")
            self.assertEqual(len(list((root / "dataset" / "real").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
