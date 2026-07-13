"""Testy orkiestracji kalibracji (fit → walidacja → warunkowy zapis). Bez gry."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from scanner.atlas.calibrate import (
    CalibrationOutcome,
    fit_and_save,
    fit_calibration,
    main,
    make_version,
)
from scanner.atlas.calibration import GroundProjection
from scanner.atlas.config import AtlasConfig
from scanner.atlas.contracts import FrameSnapshot, MoveObservation

A_TRUE = [[0.030, -0.030], [0.016, 0.016]]


def _moves(directions, noise_px=0.2, seed=3, shops=6):
    rng = np.random.default_rng(seed)
    A_inv = np.linalg.inv(np.asarray(A_TRUE, float))
    out = []
    for i, dg in enumerate(directions):
        dp = A_inv @ np.array([-dg[0], -dg[1]])
        deltas = [tuple(dp + rng.normal(0, noise_px, 2)) for _ in range(shops)]
        out.append(
            MoveObservation(
                key="wad"[i % 3],
                started_at="2026-07-01T00:00:00",
                duration_s=0.5,
                player_before=(400.0, 700.0),
                player_after=(400.0 + dg[0], 700.0 + dg[1]),
                delta_game=(float(dg[0]), float(dg[1])),
                delta_screen=tuple((float(x), float(y)) for x, y in deltas),
                confidence=1.0,
            )
        )
    return out


class CalibrateOrchestrationTests(unittest.TestCase):
    def test_good_calibration_saves(self):
        moves = _moves([(3.2, 0.0), (0.0, 3.2), (2.2, 2.2)])
        with tempfile.TemporaryDirectory() as d:
            cfg = AtlasConfig(calibration_file=os.path.join(d, "cal.json"))
            outcome = fit_and_save(moves, cfg)
            self.assertTrue(outcome.ok, outcome.message)
            self.assertTrue(outcome.saved)
            self.assertTrue(os.path.exists(cfg.calibration_file))
            # zapisana projekcja wczytuje się i rzutuje
            proj = GroundProjection.load(cfg.calibration_file)
            self.assertIsNotNone(proj)
            self.assertIn("OK", outcome.message)

    def test_collinear_rejected_not_saved(self):
        moves = _moves([(3.2, 0.0), (6.4, 0.0)])  # równoległe
        with tempfile.TemporaryDirectory() as d:
            cfg = AtlasConfig(calibration_file=os.path.join(d, "cal.json"))
            outcome = fit_and_save(moves, cfg)
            self.assertFalse(outcome.ok)
            self.assertFalse(outcome.saved)
            self.assertFalse(os.path.exists(cfg.calibration_file))
            self.assertIn("odrzucona", outcome.message)

    def test_high_residual_not_saved(self):
        moves = _moves([(3.2, 0.0), (0.0, 3.2), (2.2, 2.2)], noise_px=12.0, shops=10)
        with tempfile.TemporaryDirectory() as d:
            cfg = AtlasConfig(calibration_file=os.path.join(d, "cal.json"), max_calib_residual_px=3.0)
            outcome = fit_and_save(moves, cfg)
            self.assertFalse(outcome.ok)
            self.assertFalse(outcome.saved)
            self.assertFalse(os.path.exists(cfg.calibration_file))

    def test_max_residual_override_allows_operator_to_save_weaker_live_fit(self):
        moves = _moves([(3.2, 0.0), (0.0, 3.2), (2.2, 2.2)], noise_px=5.0, shops=10)
        with tempfile.TemporaryDirectory() as d:
            cfg = AtlasConfig(calibration_file=os.path.join(d, "cal.json"), max_calib_residual_px=1.0)
            strict = fit_and_save(moves, cfg)
            self.assertFalse(strict.ok)
            relaxed = fit_and_save(moves, cfg, max_residual_px=20.0)
            self.assertTrue(relaxed.ok, relaxed.message)
            self.assertTrue(relaxed.saved)

    def test_low_residual_one_track_fit_is_rejected_as_silent_alias_risk(self):
        moves = _moves([(3.2, 0.0), (0.0, 3.2), (-3.2, 0.0), (0.0, -3.2)], shops=1)
        moves = [replace(move, confidence=0.25) for move in moves]
        outcome = fit_calibration(moves, max_residual_px=20.0)

        self.assertFalse(outcome.ok)
        self.assertIn("1-track", outcome.message)

    def test_too_few_moves(self):
        outcome = fit_calibration(_moves([(3.2, 0.0)]))
        self.assertFalse(outcome.ok)
        self.assertIn("za mało", outcome.message)

    def test_all_moves_too_small_rejected_before_fit(self):
        # oba ruchy zablokowane (<2.5u) → brak >=2 usable → odrzucenie przed fitem
        outcome = fit_calibration(_moves([(1.0, 0.0), (0.0, 2.0)]))
        self.assertFalse(outcome.ok)
        self.assertIn("dystansie", outcome.message)

    def test_blocked_moves_dropped_not_whole_run(self):
        # jeden krok zablokowany o stragan (0u) NIE zabija runu — filtrujemy i fitujemy resztę
        outcome = fit_calibration(_moves([(3.2, 0.0), (0.0, 3.2), (0.0, 0.0)]))
        self.assertTrue(outcome.ok, outcome.message)
        self.assertIn("odrzucono 1", outcome.message)

    def test_version_is_unique_stamp(self):
        self.assertTrue(make_version("v1").startswith("v1-"))


class FakeFeed:
    def __init__(self):
        self.calls = 0

    def capture_once(self):
        self.calls += 1
        shift = float(self.calls)
        return FrameSnapshot(
            timestamp=f"t{self.calls}",
            window_rect=(10, 20, 800, 600),
            player_game=(450.0 + shift, 720.0),
            shops_screen=((100.0 - shift, 100.0), (200.0 - shift, 100.0)),
        )


class MissingAfterMoveFeed:
    def __init__(self):
        self.calls = 0

    def capture_once(self):
        self.calls += 1
        player = None if self.calls >= 4 else (450.0 + self.calls, 720.0)
        return FrameSnapshot(
            timestamp=f"m{self.calls}",
            window_rect=(10, 20, 800, 600),
            player_game=player,
            shops_screen=((100.0, 100.0), (200.0, 100.0)),
        )


class FakeInput:
    def __init__(self):
        self.events = []

    def key_down(self, key):
        self.events.append(("down", key))

    def key_up(self, key):
        self.events.append(("up", key))


class FakeWindow:
    def is_foreground(self):
        return True


class AtlasCalibrationCliSafetyTests(unittest.TestCase):
    def test_cli_default_is_dry_run_and_does_not_move(self):
        from unittest.mock import patch

        feed = FakeFeed()
        input_backend = FakeInput()
        with patch(
            "scanner.atlas.calibrate._build_live_dependencies",
            return_value=(feed, input_backend, FakeWindow()),
        ):
            code = main(["--skip-boundary", "--min-shop-tracks", "2"])

        self.assertEqual(code, 0)
        self.assertEqual(input_backend.events, [])
        self.assertEqual(feed.calls, 1)

    def test_cli_arm_moves_and_returns_before_fit(self):
        from unittest.mock import patch

        feed = FakeFeed()
        input_backend = FakeInput()
        outcome = CalibrationOutcome(True, True, "OK")
        with patch(
            "scanner.atlas.calibrate._build_live_dependencies",
            return_value=(feed, input_backend, FakeWindow()),
        ), patch(
            "scanner.atlas.calibrate.fit_and_save",
            return_value=outcome,
        ):
            code = main(
                [
                    "--arm",
                    "--skip-boundary",
                    "--keys",
                    "w,d",
                    "--steps-per-key",
                    "1",
                    "--hold",
                    "0.01",
                    "--settle",
                    "0",
                    "--countdown",
                    "0",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(
            input_backend.events,
            [
                ("down", "w"),
                ("up", "w"),
                ("down", "s"),
                ("up", "s"),
                ("down", "d"),
                ("up", "d"),
                ("down", "a"),
                ("up", "a"),
            ],
        )

    def test_cli_arm_returns_even_when_coord_missing_after_forward_step(self):
        from unittest.mock import patch

        feed = MissingAfterMoveFeed()
        input_backend = FakeInput()
        with patch(
            "scanner.atlas.calibrate._build_live_dependencies",
            return_value=(feed, input_backend, FakeWindow()),
        ):
            code = main(
                [
                    "--arm",
                    "--skip-boundary",
                    "--keys",
                    "w",
                    "--steps-per-key",
                    "1",
                    "--hold",
                    "0.01",
                    "--settle",
                    "0",
                    "--countdown",
                    "0",
                    "--preflight-timeout",
                    "0",
                ]
            )

        self.assertEqual(code, 2)
        self.assertEqual(
            input_backend.events,
            [("down", "w"), ("up", "w"), ("down", "s"), ("up", "s")],
        )


if __name__ == "__main__":
    unittest.main()
