"""Testy fotogrametrii Atlasu — odzysk macierzy A z syntetycznych obserwacji ruchu.

Bez gry: generujemy znaną macierz A, syntetyczne kroki kalibracyjne (statyczne sklepy
przesuwają się o Δp = A⁻¹·(−Δg)), dorzucamy szum + outliery i sprawdzamy, czy fit
odzyskuje A oraz czy rzut jest odwracalny (round-trip).
"""

from __future__ import annotations

import unittest

import numpy as np

from scanner.atlas.calibration import (
    DEFAULT_ANCHOR,
    GroundProjection,
    check_direction_coherence,
    check_opposite_consistency,
)
from scanner.atlas.contracts import MoveObservation


def _move(key, delta_game, delta_screen):
    return MoveObservation(
        key=key,
        delta_game=(float(delta_game[0]), float(delta_game[1])),
        delta_screen=[(float(x), float(y)) for x, y in delta_screen],
        started_at="2026-07-02T00:00:00",
        duration_s=1.0,
        player_before=(445.0, 716.0),
        player_after=(445.0 + delta_game[0], 716.0 + delta_game[1]),
        confidence=1.0,
    )


def _synth_moves(A_true, moves_game, shops_per_move=6, noise_px=0.3, seed=1, outliers=1):
    """Zbuduj MoveObservation dla znanej A i zadanych Δg (kierunków ruchu)."""
    rng = np.random.default_rng(seed)
    A_inv = np.linalg.inv(np.asarray(A_true, float))
    out = []
    for i, dg in enumerate(moves_game):
        dp_true = A_inv @ np.array([-dg[0], -dg[1]])  # A·Δp = −Δg → Δp = A⁻¹·(−Δg)
        deltas = []
        for _ in range(shops_per_move):
            deltas.append(tuple(dp_true + rng.normal(0, noise_px, 2)))
        for _ in range(outliers):  # zły track = losowy outlier
            deltas.append(tuple(dp_true + rng.normal(0, 40, 2)))
        out.append(
            MoveObservation(
                key="wasd"[i % 4],
                delta_game=(float(dg[0]), float(dg[1])),
                delta_screen=[(float(x), float(y)) for x, y in deltas],
                started_at="2026-07-01T00:00:00",
                duration_s=0.5,
                player_before=(400.0, 700.0),
                player_after=(400.0 + dg[0], 700.0 + dg[1]),
                confidence=1.0,
            )
        )
    return out


class GroundProjectionCalibrationTests(unittest.TestCase):
    A_TRUE = [[0.030, -0.030], [0.016, 0.016]]  # izometryczna macierz px→u

    def test_recovers_matrix_from_clean_moves(self):
        moves = _synth_moves(self.A_TRUE, [(3.2, 0.0), (0.0, 3.2), (2.2, 2.2)], noise_px=0.2)
        proj, res = GroundProjection.fit_from_moves(moves)
        np.testing.assert_allclose(proj.A, np.asarray(self.A_TRUE), atol=2e-3)
        self.assertTrue(res.ok, f"fit powinien być zdrowy: {res}")
        self.assertLess(res.residual_px, 3.0)
        self.assertGreaterEqual(res.inliers, 12)

    def test_rejects_outliers(self):
        # dużo outlierów — odrzut po MAD ma je odsiać, A dalej blisko prawdy
        moves = _synth_moves(
            self.A_TRUE, [(3.2, 0.0), (0.0, 3.2), (2.2, 2.2)], noise_px=0.2, outliers=3
        )
        proj, res = GroundProjection.fit_from_moves(moves)
        np.testing.assert_allclose(proj.A, np.asarray(self.A_TRUE), atol=6e-3)
        self.assertLess(res.inliers, res.n_points)  # coś odrzucono

    def test_round_trip_projection(self):
        proj = GroundProjection(self.A_TRUE)
        player = (450.0, 720.0)
        for game in [(455.0, 725.0), (440.0, 700.0), (470.0, 740.0)]:
            px = proj.game_to_screen(player, game)
            back = proj.screen_to_game(player, px)
            np.testing.assert_allclose(back, game, atol=1e-9)

    def test_screen_to_game_uses_anchor(self):
        proj = GroundProjection(self.A_TRUE, anchor=DEFAULT_ANCHOR)
        player = (400.0, 700.0)
        # piksel w kotwicy = pozycja postaci (zerowy offset)
        self.assertEqual(proj.screen_to_game(player, DEFAULT_ANCHOR), player)

    def test_collinear_moves_raise(self):
        # dwa równoległe kierunki → układ zdegenerowany → czytelny ValueError, nie crash w inv
        moves = _synth_moves(self.A_TRUE, [(3.2, 0.0), (6.4, 0.0)], noise_px=0.2, outliers=0)
        with self.assertRaises(ValueError):
            GroundProjection.fit_from_moves(moves)

    def test_serialization_round_trip(self):
        proj = GroundProjection(self.A_TRUE, version="test-v1")
        restored = GroundProjection.from_dict(proj.to_dict())
        np.testing.assert_allclose(restored.A, proj.A)
        self.assertEqual(restored.version, "test-v1")

    def test_load_missing_or_corrupt_returns_none(self, ):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(GroundProjection.load(os.path.join(d, "brak.json")))
            bad = os.path.join(d, "zly.json")
            with open(bad, "w", encoding="utf-8") as f:
                f.write("{ to nie json")
            self.assertIsNone(GroundProjection.load(bad))


class OppositeConsistencyTests(unittest.TestCase):
    # Realny run z gęstej kraty: w i s to ruch przeciwny w GRZE, ale tracki dają
    # OBA zgodny (dodatni) y na ekranie → aliasing, kalibracja do odrzucenia.
    ALIASED_W = [
        _move("w", (5, 4), [(-15, 20), (-26, 20), (-26, 24), (-13, 37), (0, 11)]),
        _move("w", (4, 4), [(-7, 26), (-8, 32)]),
    ]
    ALIASED_S = [
        _move("s", (-4, -4), [(48, 41), (56, 39), (38, 36), (56, 33), (42, 29), (36, 48), (62, 32)]),
        _move("s", (-5, -4), [(49, 35)]),
    ]

    def test_flags_aliased_opposite_screen(self):
        problems = check_opposite_consistency(self.ALIASED_W + self.ALIASED_S)
        self.assertTrue(problems, "aliasowany run powinien zostać oznaczony")
        self.assertIn("w/s", problems[0])

    def test_passes_true_opposite_screen(self):
        # ekranowe delty faktycznie przeciwne (statyczne sklepy) → brak problemów
        w = _move("w", (4, 0), [(-30, -18), (-31, -17), (-29, -19)])
        s = _move("s", (-4, 0), [(30, 18), (31, 17), (29, 19)])
        self.assertEqual(check_opposite_consistency([w, s]), [])

    def test_skips_when_game_not_opposite(self):
        # postać zablokowana: Δgame nie są przeciwne → pary nie oceniamy (brak fałszywego alarmu)
        w = _move("w", (4, 4), [(-10, -10)])
        s = _move("s", (3, 3), [(-9, -9)])
        self.assertEqual(check_opposite_consistency([w, s]), [])

    def test_noop_without_opposite_pair(self):
        # tylko w i d — żadna para (w/s, a/d) nie jest kompletna
        w = _move("w", (4, 0), [(-30, -18)])
        d = _move("d", (0, 4), [(20, -25)])
        self.assertEqual(check_opposite_consistency([w, d]), [])

    def test_fit_rejects_aliased_run(self):
        # czyste a/d (druga oś) → układ dobrze uwarunkowany, ale aliasing w/s
        # sam w sobie musi wywalić `ok`.
        clean_ad = [
            _move("a", (0, 4), [(25, -15), (26, -14), (24, -16)]),
            _move("d", (0, -4), [(-25, 15), (-26, 14), (-24, 16)]),
        ]
        proj, res = GroundProjection.fit_from_moves(
            self.ALIASED_W + self.ALIASED_S + clean_ad
        )
        self.assertTrue(res.opposite_problems)
        self.assertFalse(res.ok, "fit z aliasowanymi trackami nie może być 'ok'")


class DirectionCoherenceTests(unittest.TestCase):
    def test_flags_incoherent_key(self):
        # realny run: `a` naciśnięte 4× leci w 4 różne ćwiartki → wektory się znoszą
        a = [
            _move("a", (-4, -3), [(1, 1)]),
            _move("a", (-3, 4), [(1, 1)]),
            _move("a", (4, 4), [(1, 1)]),
            _move("a", (3, -4), [(1, 1)]),
        ]
        problems = check_direction_coherence(a)
        self.assertTrue(problems)
        self.assertTrue(problems[0].startswith("a:"))

    def test_passes_coherent_key(self):
        w = [
            _move("w", (6, -1), [(1, 1)]),
            _move("w", (5, -1), [(1, 1)]),
            _move("w", (6, 0), [(1, 1)]),
        ]
        self.assertEqual(check_direction_coherence(w), [])

    def test_single_move_key_not_judged(self):
        # jeden ruch = brak podstaw do oceny kierunku
        self.assertEqual(check_direction_coherence([_move("w", (6, -1), [(1, 1)])]), [])


if __name__ == "__main__":
    unittest.main()
