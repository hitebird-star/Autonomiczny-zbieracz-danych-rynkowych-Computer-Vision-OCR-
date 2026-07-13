from __future__ import annotations

import unittest

from scanner.analysis.scan_order import forward_indices, forward_points


class ForwardConeTests(unittest.TestCase):
    def test_nearest_to_farthest_order(self) -> None:
        # heading +x; trzy punkty na wprost w roznych odleglosciach -> najblizszy pierwszy
        pts = [(30, 0), (10, 0), (20, 0)]
        self.assertEqual(forward_indices((0, 0), (1, 0), pts), [1, 2, 0])

    def test_rear_points_dropped(self) -> None:
        # heading +x; punkt za plecami (-x) odpada z polplaszczyzny (half_angle=90)
        pts = [(10, 0), (-10, 0)]
        self.assertEqual(forward_indices((0, 0), (1, 0), pts), [0])

    def test_side_kept_on_half_plane(self) -> None:
        # half_angle=90 = przednia polplaszczyzna: bok (90deg, cos=0) zostaje (cos_a>=cos_lim=0)
        pts = [(0, 10)]
        self.assertEqual(forward_indices((0, 0), (1, 0), pts), [0])

    def test_narrow_cone_drops_side(self) -> None:
        # wezszy stozek (60deg) tnie bok 90deg
        pts = [(10, 0), (0, 10)]
        self.assertEqual(forward_indices((0, 0), (1, 0), pts, half_angle_deg=60.0), [0])

    def test_radius_band(self) -> None:
        # pas osiagalnosci min_r..max_r (pierscien klikalny)
        pts = [(5, 0), (15, 0), (300, 0)]
        self.assertEqual(
            forward_indices((0, 0), (1, 0), pts, min_r=10.0, max_r=240.0), [1]
        )

    def test_heading_none_no_angle_filter(self) -> None:
        # nieznany kierunek (przed 1. 'w') -> brak filtra kata, sam dystans (wszystkie, najblizej pierwsze)
        pts = [(10, 0), (-5, 0), (0, 20)]
        self.assertEqual(forward_indices((0, 0), None, pts), [1, 0, 2])

    def test_zero_heading_degrades_like_none(self) -> None:
        pts = [(10, 0), (-5, 0)]
        self.assertEqual(
            forward_indices((0, 0), (0, 0), pts),
            forward_indices((0, 0), None, pts),
        )

    def test_full_circle_at_180(self) -> None:
        pts = [(10, 0), (-10, 0)]
        self.assertEqual(
            sorted(forward_indices((0, 0), (1, 0), pts, half_angle_deg=180.0)), [0, 1]
        )

    def test_center_offset(self) -> None:
        # center przesuniety: dystanse liczone wzgledem center, nie origin
        pts = [(110, 100), (130, 100)]
        self.assertEqual(forward_indices((100, 100), (1, 0), pts), [0, 1])

    def test_forward_points_wrapper(self) -> None:
        pts = [(30, 0), (10, 0)]
        self.assertEqual(forward_points((0, 0), (1, 0), pts), [(10, 0), (30, 0)])


if __name__ == "__main__":
    unittest.main()
