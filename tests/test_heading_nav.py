from __future__ import annotations

import math
import unittest

from scanner.analysis.heading_nav import (
    better_turn_key,
    direction_to,
    direction_to_envelope,
    facing_error_deg,
    is_outside,
    nearest_in_envelope,
    recovery_target,
    should_drive,
    signed_angle,
    turn_to_face,
)


class DirectionTests(unittest.TestCase):
    def test_unit_vector(self):
        self.assertEqual(direction_to((0, 0), (10, 0)), (1.0, 0.0))
        dx, dy = direction_to((0, 0), (3, 4))
        self.assertAlmostEqual(dx, 0.6)
        self.assertAlmostEqual(dy, 0.8)

    def test_same_point_zero(self):
        self.assertEqual(direction_to((5, 5), (5, 5)), (0.0, 0.0))


class SignedAngleTests(unittest.TestCase):
    def test_aligned_zero(self):
        self.assertAlmostEqual(signed_angle((1, 0), (1, 0)), 0.0)

    def test_ccw_positive(self):
        # od +X do +Y to +90° (CCW w standardowej ramce)
        self.assertAlmostEqual(math.degrees(signed_angle((1, 0), (0, 1))), 90.0)

    def test_cw_negative(self):
        self.assertAlmostEqual(math.degrees(signed_angle((1, 0), (0, -1))), -90.0)

    def test_normalized_to_pi(self):
        a = math.degrees(signed_angle((1, 0), (-1, 0.001)))
        self.assertLessEqual(abs(a), 180.0)

    def test_zero_vector_safe(self):
        self.assertEqual(signed_angle((0, 0), (1, 0)), 0.0)


class FacingTests(unittest.TestCase):
    def test_facing_error_zero_when_aligned(self):
        # heading +X, cel na +X → błąd 0
        self.assertAlmostEqual(facing_error_deg((1, 0), (0, 0), (10, 0)), 0.0)

    def test_facing_error_90(self):
        self.assertAlmostEqual(facing_error_deg((1, 0), (0, 0), (0, 10)), 90.0)

    def test_should_drive_within_tol(self):
        self.assertTrue(should_drive((1, 0), (0, 0), (10, 1), tol_deg=25))   # ~6° < 25
        self.assertFalse(should_drive((1, 0), (0, 0), (0, 10), tol_deg=25))  # 90° > 25

    def test_should_drive_no_heading_true(self):
        self.assertTrue(should_drive((0, 0), (0, 0), (10, 10)))   # brak headingu → jedź i zmierz

    def test_should_drive_on_target(self):
        self.assertTrue(should_drive((1, 0), (5, 5), (5, 5)))     # już na celu


class TurnToFaceTests(unittest.TestCase):
    def test_drive_when_facing(self):
        self.assertEqual(turn_to_face((1, 0), (0, 0), (10, 0)), "w")

    def test_turn_for_ccw_target_default_convention(self):
        # cel na +Y (CCW od +X). d_is_ccw=False (domyślnie 'd'=CW) → skręt CCW = 'a'
        self.assertEqual(turn_to_face((1, 0), (0, 0), (0, 10), d_is_ccw=False), "a")

    def test_turn_respects_convention_flag(self):
        # ta sama geometria, ale 'd'=CCW → CCW = 'd'
        self.assertEqual(turn_to_face((1, 0), (0, 0), (0, 10), d_is_ccw=True), "d")

    def test_turn_for_cw_target(self):
        # cel na -Y (CW). domyślnie CW = 'd'
        self.assertEqual(turn_to_face((1, 0), (0, 0), (0, -10), d_is_ccw=False), "d")

    def test_no_heading_drives(self):
        self.assertEqual(turn_to_face((0, 0), (0, 0), (0, 10)), "w")


class EnvelopeRecoveryTests(unittest.TestCase):
    """Powrót do farmy gdy bot wyszedł poza obszar (bug biegu 165822)."""

    ENV = (348.0, 501.0, 672.0, 794.0)

    def test_nearest_in_envelope_clamps(self):
        self.assertEqual(nearest_in_envelope((379, 639), self.ENV), (379, 672))  # Y poniżej → na dolną
        self.assertEqual(nearest_in_envelope((400, 700), self.ENV), (400, 700))  # w środku → bez zmian

    def test_is_outside(self):
        self.assertTrue(is_outside((379, 639), self.ENV))   # Y<672
        self.assertFalse(is_outside((400, 700), self.ENV))

    def test_direction_to_envelope_points_back_in(self):
        # bot poniżej dolnej krawędzi (Y=639) → kierunek powrotu ma +Y (w górę do farmy)
        dx, dy = direction_to_envelope((400, 639), self.ENV)
        self.assertGreater(dy, 0)                           # +Y = z powrotem w obszar
        self.assertAlmostEqual(dx, 0.0)

    def test_direction_zero_when_inside(self):
        self.assertEqual(direction_to_envelope((400, 700), self.ENV), (0.0, 0.0))

    def test_recovery_target_is_inset_inside(self):
        t = recovery_target((400, 639), self.ENV, inset=15.0)
        self.assertTrue(672 <= t[1] <= 794)                 # w kopercie
        self.assertGreater(t[1], 672)                       # GŁĘBIEJ niż sama krawędź


class RotateMeasureTests(unittest.TestCase):
    def test_keep_key_when_error_drops(self):
        self.assertEqual(better_turn_key(90.0, 60.0, "d"), "d")   # zmalał → trzymaj

    def test_switch_key_when_error_grows(self):
        self.assertEqual(better_turn_key(60.0, 80.0, "d"), "a")   # wzrósł → druga strona


if __name__ == "__main__":
    unittest.main()
