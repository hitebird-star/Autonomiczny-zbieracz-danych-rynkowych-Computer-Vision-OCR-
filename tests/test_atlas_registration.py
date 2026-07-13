"""Testy korelacji fazowej Atlasu — odzysk translacji ekranu z obrazu (bez gry).

Kluczowy test: silna regularna krata + słaba aperiodyczna tekstura, przesunięcie MNIEJSZE
niż oczko kraty — czyli scenariusz, który degeneruje dopasowanie punktów-sklepów. Korelacja
fazowa na całej klatce ma odzyskać prawdziwe przesunięcie, bo tekstura łamie periodyczność.
"""

from __future__ import annotations

import unittest

import numpy as np

from scanner.atlas.registration import (
    crop_world_viewport,
    estimate_screen_shift,
    screen_shift_between_frames,
)


def _lattice(h=256, w=256, pitch=40, blob=6):
    img = np.zeros((h, w), dtype=np.float64)
    for y in range(pitch // 2, h, pitch):
        for x in range(pitch // 2, w, pitch):
            img[y : y + blob, x : x + blob] = 1.0
    return img


class RegistrationTests(unittest.TestCase):
    def test_recovers_shift_on_textured_image(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, (200, 220))
        dy, dx = 7, 12
        after = np.roll(base, shift=(dy, dx), axis=(0, 1))
        rdx, rdy, conf = estimate_screen_shift(base, after, max_shift_px=30)
        self.assertAlmostEqual(rdx, dx, delta=1.0)
        self.assertAlmostEqual(rdy, dy, delta=1.0)
        self.assertGreater(conf, 0.5)

    def test_recovers_small_shift_on_lattice_plus_texture(self):
        # to jest scenariusz zabijający match_shop_deltas: krata identycznych straganów,
        # ruch < oczko. Tekstura ziemi (aperiodyczna) daje jednoznaczny pik.
        rng = np.random.default_rng(1)
        base = _lattice(pitch=40) * 3.0 + rng.normal(0, 0.4, (256, 256))
        dy, dx = 5, 9  # mniejsze niż pitch=40
        after = np.roll(base, shift=(dy, dx), axis=(0, 1))
        rdx, rdy, conf = estimate_screen_shift(base, after, max_shift_px=18)
        self.assertAlmostEqual(rdx, dx, delta=1.0)
        self.assertAlmostEqual(rdy, dy, delta=1.0)

    def test_negative_shift_sign(self):
        rng = np.random.default_rng(2)
        base = rng.normal(0, 1, (180, 180))
        after = np.roll(base, shift=(-8, -5), axis=(0, 1))
        rdx, rdy, _ = estimate_screen_shift(base, after, max_shift_px=30)
        self.assertAlmostEqual(rdx, -5, delta=1.0)
        self.assertAlmostEqual(rdy, -8, delta=1.0)

    def test_rgb_image_accepted(self):
        rng = np.random.default_rng(3)
        base = rng.integers(0, 255, (160, 160, 3)).astype(np.float64)
        after = np.roll(base, shift=(4, 6), axis=(0, 1))
        rdx, rdy, _ = estimate_screen_shift(base, after, max_shift_px=20)
        self.assertAlmostEqual(rdx, 6, delta=1.0)
        self.assertAlmostEqual(rdy, 4, delta=1.0)

    def test_mismatched_size_raises(self):
        with self.assertRaises(ValueError):
            estimate_screen_shift(np.zeros((10, 10)), np.zeros((10, 12)))


class ViewportCropTests(unittest.TestCase):
    def _world(self, h=256, w=256):
        rng = np.random.default_rng(7)
        return _lattice(h, w, pitch=40) * 3.0 + rng.normal(0, 0.4, (h, w))

    def test_crop_recovers_shift_despite_strong_static_ui(self):
        # świat przesuwa się o (dy,dx); w prawym-górnym rogu SZCZEGÓŁOWE, silne, STATYCZNE
        # UI (jak minimapa) identyczne w obu klatkach → nie przesuwa się z kamerą i wnosi
        # mocny komponent zero-shift. Klocek z kadrem musi mimo to odzyskać prawdę.
        rng = np.random.default_rng(11)
        world = self._world()
        dy, dx = 6, 10
        before = world.copy()
        after = np.roll(world, shift=(dy, dx), axis=(0, 1))
        ui_block = rng.normal(0, 8.0, (70, 66))  # szczegółowy, silny, ten sam w obu
        before[0:70, 190:256] = ui_block
        after[0:70, 190:256] = ui_block

        cdx, cdy, conf = screen_shift_between_frames(before, after, max_shift_px=20)
        self.assertAlmostEqual(cdx, dx, delta=1.5)
        self.assertAlmostEqual(cdy, dy, delta=1.5)
        self.assertGreater(conf, 0.3)

    def test_crop_dimensions(self):
        cropped = crop_world_viewport(np.zeros((1080, 1920)))
        self.assertLess(cropped.shape[0], 1080)
        self.assertLess(cropped.shape[1], 1920)
        self.assertGreater(cropped.size, 0)

    def test_crop_too_aggressive_raises(self):
        with self.assertRaises(ValueError):
            crop_world_viewport(np.zeros((100, 100)), margins=(0.5, 0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
