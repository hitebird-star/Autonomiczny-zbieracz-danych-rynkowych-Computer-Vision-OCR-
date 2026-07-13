"""Testy rdzenia detector_replay — predykaty akceptacji + detect + parowanie."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanner.analysis import detector_replay as dr


class AcceptPredicatesTest(unittest.TestCase):
    def test_dark_share_przyjmuje_ciemny_panel_z_tekstem(self):
        panel = np.full((120, 120), 30, dtype=np.uint8)
        panel[10:13, :] = 200  # jasne linie tekstu
        base = np.full((120, 120), 120, dtype=np.uint8)
        self.assertTrue(dr.accept_dark_share(panel, base))

    def test_dark_share_odrzuca_jasny_panel(self):
        panel = np.full((120, 120), 200, dtype=np.uint8)
        base = np.full((120, 120), 120, dtype=np.uint8)
        self.assertFalse(dr.accept_dark_share(panel, base))

    def test_darkened_share_przyjmuje_pociemnienie(self):
        base = np.full((120, 120), 160, dtype=np.uint8)
        panel = np.full((120, 120), 30, dtype=np.uint8)
        panel[10:13, :] = 200
        self.assertTrue(dr.accept_darkened_share(panel, base))

    def test_darkened_share_odrzuca_bez_zmiany(self):
        base = np.full((120, 120), 120, dtype=np.uint8)
        panel = base.copy()
        self.assertFalse(dr.accept_darkened_share(panel, base))

    def test_darkened_uniform_odroznia_jednolite_od_teksturowanego(self):
        base = np.full((120, 120), 120, dtype=np.uint8)
        # jednolity ciemny panel + cienkie jasne linie -> uniform
        uniform = np.full((120, 120), 30, dtype=np.uint8)
        uniform[10:13, :] = 200
        self.assertTrue(dr.accept_darkened_uniform(uniform, base))
        # teksturowany (na przemian 30/200) -> darkened przejdzie, uniform nie
        textured = np.full((120, 120), 30, dtype=np.uint8)
        textured[:, ::2] = 200
        self.assertTrue(dr.accept_darkened_share(textured, base))
        self.assertFalse(dr.accept_darkened_uniform(textured, base))


class DetectTest(unittest.TestCase):
    def _frame_with_panel(self):
        base = np.full((200, 200, 3), 120, dtype=np.uint8)
        hover = base.copy()
        hover[40:160, 40:160] = 30          # ciemny panel dymka
        hover[60:63, 40:160] = 200          # jasna linia tekstu (>150)
        return base, hover

    def test_dark_share_wykrywa_panel(self):
        base, hover = self._frame_with_panel()
        bbox = dr.detect(base, hover, dr.accept_dark_share)
        self.assertIsNotNone(bbox)
        left, top, right, bottom = bbox
        # bbox obejmuje narysowany panel (z paddingiem 10)
        self.assertLessEqual(left, 40)
        self.assertGreaterEqual(right, 160)

    def test_brak_ciemnego_panelu_zwraca_none(self):
        base = np.full((200, 200, 3), 120, dtype=np.uint8)
        hover = np.full((200, 200, 3), 200, dtype=np.uint8)  # caly jasny
        self.assertIsNone(dr.detect(base, hover, dr.accept_dark_share))


class FramePairTest(unittest.TestCase):
    def test_parowanie_baseline_hover(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "20260621_160000_test" / "frames"
            scan.mkdir(parents=True)
            img = Image.new("RGB", (10, 10))
            img.save(scan / "slot_007_baseline.png")
            img.save(scan / "slot_007_hover.png")
            img.save(scan / "slot_009_baseline.png")  # bez pary hover -> pomin
            pairs = dr.find_frame_pairs(tmp, "*")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].slot, "slot_007")


if __name__ == "__main__":
    unittest.main()
