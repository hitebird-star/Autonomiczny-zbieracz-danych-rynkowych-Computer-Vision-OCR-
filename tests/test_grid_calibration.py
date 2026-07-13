from __future__ import annotations

import unittest

from PIL import Image

from scanner.analysis.grid_calibration import (
    apply_calibration,
    calibration_from_points,
    render_grid_overlay,
)


class RenderGridOverlayTests(unittest.TestCase):
    def test_overlay_keeps_region_size(self):
        img = Image.new("RGB", (10 * 28, 10 * 28), (0, 0, 0))
        out = render_grid_overlay(img, cols=10, rows=10, cell=28)
        self.assertEqual(out.size, (280, 280))

    def test_overlay_draws_lines_and_dots(self):
        img = Image.new("RGB", (2 * 32, 2 * 32), (0, 0, 0))
        out = render_grid_overlay(img, cols=2, rows=2, cell=32)
        px = out.load()
        self.assertEqual(px[0, 0], (255, 0, 0))          # górno-lewa linia siatki
        self.assertEqual(px[16, 16], (0, 255, 0))        # środek slotu (0,0) = kropka klika

    def test_overlay_does_not_mutate_input(self):
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        before = img.tobytes()
        render_grid_overlay(img, cols=1, rows=1, cell=32)
        self.assertEqual(img.tobytes(), before)

    def test_invalid_dims_raise(self):
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        for kwargs in ({"cols": 0}, {"rows": 0}, {"cell": 0}):
            args = {"cols": 1, "rows": 1, "cell": 32, **kwargs}
            with self.assertRaises(ValueError):
                render_grid_overlay(img, **args)


class ApplyCalibrationTests(unittest.TestCase):
    def _cfg(self):
        return {"shop_origin": [3089, 266], "grid": {"cell": 28, "grid_dx": 8, "grid_dy": 27, "cols": 10, "rows": 10, "occ_residual": 12.0}}

    def test_restores_cell_keeps_origin(self):
        out = apply_calibration(self._cfg(), cell=32, grid_dx=7, grid_dy=32)
        self.assertEqual(out["grid"]["cell"], 32)
        self.assertEqual(out["grid"]["grid_dx"], 7)
        self.assertEqual(out["grid"]["grid_dy"], 32)
        self.assertEqual(out["shop_origin"], [3089, 266])     # origin nietknięty (okno przesunięte)
        self.assertEqual(out["grid"]["occ_residual"], 12.0)   # reszta nietknięta

    def test_set_origin(self):
        out = apply_calibration(self._cfg(), origin=(3091, 45))
        self.assertEqual(out["shop_origin"], [3091, 45])

    def test_does_not_mutate_input(self):
        cfg = self._cfg()
        apply_calibration(cfg, cell=32, origin=(1, 2))
        self.assertEqual(cfg["grid"]["cell"], 28)
        self.assertEqual(cfg["shop_origin"], [3089, 266])


class CalibrationFromPointsTests(unittest.TestCase):
    def test_computes_origin_cell_and_grid_offset(self):
        result = calibration_from_points(
            origin=(3089, 266),
            slot00_center=(3112, 314),
            slot10_center=(3144, 314),
        )

        self.assertEqual(result.origin, (3089, 266))
        self.assertEqual(result.cell, 32)
        self.assertEqual(result.grid_dx, 7)
        self.assertEqual(result.grid_dy, 32)

    def test_rejects_too_small_cell_distance(self):
        with self.assertRaises(ValueError):
            calibration_from_points(
                origin=(100, 100),
                slot00_center=(120, 120),
                slot10_center=(123, 120),
            )


if __name__ == "__main__":
    unittest.main()
