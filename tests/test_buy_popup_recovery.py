from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from scanner.capture.shop_capture import OccupiedSlot
from scanner.models import ShopScan
from scanner.pipeline import GameCapturePipeline
from tests.fakes import FakeInput, FakeScreen


class BuyPopupRecoveryTests(unittest.TestCase):
    def _make_pipeline(self) -> tuple[GameCapturePipeline, FakeInput]:
        pipeline = object.__new__(GameCapturePipeline)
        input_backend = FakeInput()
        input_backend._api = SimpleNamespace(
            rightClick=lambda x, y: input_backend.actions.append(
                ("rightClick", x, y)
            ),
            click=lambda x, y, button=None: input_backend.actions.append(
                ("api_click", x, y, button)
            ),
        )
        pipeline.input = input_backend
        pipeline.window_box = (100, 100, 400, 300)
        pipeline.shop_capturer = SimpleNamespace(
            geometry=SimpleNamespace(slot_center=lambda column, row: (123, 456)),
            screen=FakeScreen(Image.new("RGB", (400, 220), "black")),
        )
        return pipeline, input_backend

    def test_buy_popup_parse_failure_does_not_escape_shop(self) -> None:
        pipeline, input_backend = self._make_pipeline()
        fake_win_ocr = SimpleNamespace(
            recognize=lambda image: [{"text": "losowy tekst bez dialogu"}]
        )

        with patch.dict(sys.modules, {"win_ocr": fake_win_ocr}):
            result = pipeline._read_slot_from_buy_popup(
                ShopScan("scan-01"),
                OccupiedSlot(slot=0, row=0, column=0, residual=10.0),
            )

        self.assertIsNone(result)
        self.assertIn(("rightClick", 123, 456), input_backend.actions)
        self.assertNotIn(("press", "esc"), input_backend.actions)


if __name__ == "__main__":
    unittest.main()
