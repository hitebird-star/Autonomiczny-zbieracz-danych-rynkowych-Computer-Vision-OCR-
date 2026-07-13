from __future__ import annotations

import unittest

from PIL import Image

from scanner.capture import SellerReader, parse_seller
from scanner.config import GridGeometry
from tests.fakes import FakeScreen


class SellerReaderTests(unittest.TestCase):
    def test_parse_seller(self) -> None:
        self.assertEqual(parse_seller("Sklep Offline (DeanW)"), "DeanW")
        self.assertEqual(parse_seller("ep Offline (Freox)"), "Freox")
        self.assertEqual(parse_seller("Sklep UTI�ne (Freox)"), "Freox")
        self.assertEqual(parse_seller("bez tytułu"), "")

    def test_reads_seller_from_title_bar(self) -> None:
        reader = SellerReader(
            FakeScreen(Image.new("RGB", (320, 28), "black")),
            GridGeometry(origin=(100, 200)),
            recognizer=lambda image: [
                {"text": "Sklep Offline (DeanW)", "box": (10, 2, 200, 20)}
            ],
        )

        self.assertEqual(reader.read(), "DeanW")
