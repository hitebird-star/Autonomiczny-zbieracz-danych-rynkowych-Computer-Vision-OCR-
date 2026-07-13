"""Testy warstwy zabezpieczen nawigacji (nav_guards.py)."""
import unittest

from scanner.analysis.nav_guards import (
    FailReason,
    NavigateResult,
    REACHED,
    budget_exceeded,
    distance_regressing,
    is_ocr_frozen,
    is_oscillating,
    is_stuck,
    next_step_within_envelope,
    reached_target,
    step_budget,
    target_valid,
    within_envelope,
)


FARM_ENVELOPE = (348, 672, 501, 794)
# Koperta Glevia Farm: (x0, y0, x1, y1) = (348, 672, 501, 794)


class WithinEnvelopeTest(unittest.TestCase):
    def test_inside_center(self):
        self.assertTrue(within_envelope(420, 730, FARM_ENVELOPE))

    def test_on_left_edge(self):
        self.assertTrue(within_envelope(348, 730, FARM_ENVELOPE))

    def test_on_right_edge(self):
        self.assertTrue(within_envelope(501, 730, FARM_ENVELOPE))

    def test_on_top_edge(self):
        self.assertTrue(within_envelope(420, 672, FARM_ENVELOPE))

    def test_on_bottom_edge(self):
        self.assertTrue(within_envelope(420, 794, FARM_ENVELOPE))

    def test_outside_left(self):
        self.assertFalse(within_envelope(300, 730, FARM_ENVELOPE))

    def test_outside_right(self):
        self.assertFalse(within_envelope(550, 730, FARM_ENVELOPE))

    def test_outside_up(self):
        self.assertFalse(within_envelope(420, 600, FARM_ENVELOPE))

    def test_outside_down(self):
        self.assertFalse(within_envelope(420, 850, FARM_ENVELOPE))


class NextStepWithinEnvelopeTest(unittest.TestCase):
    def test_step_keeps_inside(self):
        self.assertTrue(next_step_within_envelope(400, 700, 7, 0, FARM_ENVELOPE))

    def test_step_would_exit_left(self):
        self.assertFalse(next_step_within_envelope(348, 700, -7, 0, FARM_ENVELOPE))

    def test_step_would_exit_right(self):
        self.assertFalse(next_step_within_envelope(501, 700, 7, 0, FARM_ENVELOPE))

    def test_step_would_exit_up(self):
        self.assertFalse(next_step_within_envelope(420, 672, 0, -7, FARM_ENVELOPE))

    def test_step_would_exit_down(self):
        self.assertFalse(next_step_within_envelope(420, 794, 0, 7, FARM_ENVELOPE))


class TargetValidTest(unittest.TestCase):
    def test_valid_target(self):
        self.assertTrue(target_valid(450, 720, FARM_ENVELOPE))

    def test_oob_target(self):
        self.assertFalse(target_valid(600, 300, FARM_ENVELOPE))

    def test_corner_target(self):
        self.assertTrue(target_valid(348, 672, FARM_ENVELOPE))


class ReachedTargetTest(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(reached_target((450, 720), (450, 720)))

    def test_within_eps(self):
        self.assertTrue(reached_target((450, 720), (451, 721), eps=3.0))

    def test_beyond_eps(self):
        self.assertFalse(reached_target((450, 720), (460, 720), eps=3.0))

    def test_distance_equals_eps(self):
        # dist = sqrt(3^2 + 0) = 3, eps=3 -> True (<=eps)
        self.assertTrue(reached_target((450, 720), (453, 720), eps=3.0))


class OcrFrozenTest(unittest.TestCase):
    def test_frozen(self):
        readings = [(400, 700), (401, 699), (400, 701), (400, 700), (401, 700)]
        self.assertTrue(is_ocr_frozen(readings, max_same=5, eps=2.0))

    def test_not_frozen_moving(self):
        readings = [(400, 700), (410, 700), (420, 700), (430, 700), (440, 700)]
        self.assertFalse(is_ocr_frozen(readings, max_same=5, eps=2.0))

    def test_not_enough_readings(self):
        readings = [(400, 700)]
        self.assertFalse(is_ocr_frozen(readings, max_same=5))


class StuckTest(unittest.TestCase):
    def test_stuck_same_positions(self):
        positions = [(400, 700)] * 5
        self.assertTrue(is_stuck(positions, max_same=5))

    def test_not_stuck_moving(self):
        positions = [(400, 700), (407, 700), (414, 700), (421, 700), (428, 700)]
        self.assertFalse(is_stuck(positions, max_same=5))


class DistanceRegressingTest(unittest.TestCase):
    def test_regressing(self):
        # dystans rosnie: 10, 15, 20, 25
        self.assertTrue(distance_regressing([10.0, 15.0, 20.0, 25.0], regress_threshold=3))

    def test_not_regressing_decreasing(self):
        self.assertFalse(distance_regressing([25.0, 20.0, 15.0, 10.0], regress_threshold=3))

    def test_not_enough_data(self):
        self.assertFalse(distance_regressing([10.0, 15.0], regress_threshold=3))

    def test_flat_then_increase(self):
        # 10, 10, 12, 14 -> ostatnie 4: 10,10,12,14 -> 14>12>10 ale 10==10 -> False
        self.assertFalse(distance_regressing([10.0, 10.0, 12.0, 14.0], regress_threshold=3))


class OscillatingTest(unittest.TestCase):
    def test_oscillating_small_zigzag(self):
        positions = [(400, 700), (405, 698), (402, 703), (398, 699), (403, 701)]
        self.assertTrue(is_oscillating(positions, window=5, max_amplitude=10.0))

    def test_not_oscillating_large_movement(self):
        positions = [(400, 700), (420, 700), (440, 700), (460, 700), (480, 700)]
        self.assertFalse(is_oscillating(positions, window=5, max_amplitude=10.0))

    def test_not_enough_positions(self):
        self.assertFalse(is_oscillating([(400, 700)], window=5))


class StepBudgetTest(unittest.TestCase):
    def test_budget_short_distance(self):
        # dist = sqrt(21^2+0) = 21, units=7, 21/7*2 = 6
        budget = step_budget((421, 700), (400, 700), 7.0)
        self.assertEqual(budget, 6)

    def test_budget_long_distance(self):
        # dist = sqrt(100^2+0) = 100, units=7, 100/7*2 ≈ 28.6 -> 29
        budget = step_budget((500, 700), (400, 700), 7.0)
        self.assertGreater(budget, 20)

    def test_budget_minimum_one(self):
        budget = step_budget((401, 700), (400, 700), 7.0)
        self.assertEqual(budget, 1)

    def test_budget_exceeded(self):
        self.assertTrue(budget_exceeded(10, 5))
        self.assertFalse(budget_exceeded(3, 5))


class FailReasonTest(unittest.TestCase):
    def test_all_reasons_are_strings(self):
        for reason in FailReason:
            self.assertIsInstance(reason.value, str)

    def test_navigate_result_failure(self):
        result = NavigateResult(
            success=False,
            reason=FailReason.OUT_OF_BOUNDS,
            steps_taken=5,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "out_of_bounds")

    def test_reached_is_success(self):
        self.assertTrue(REACHED.success)
        self.assertEqual(REACHED.reason, "reached")


if __name__ == "__main__":
    unittest.main()