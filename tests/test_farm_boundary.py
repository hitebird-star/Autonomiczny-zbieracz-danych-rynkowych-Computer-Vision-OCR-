from __future__ import annotations

import unittest

from scanner.analysis.farm_boundary import (
    FarmBoundary,
    convex_hull,
    dist_to_boundary,
    expand_to_include,
    near_boundary,
    point_in_polygon,
    robust_point,
    should_turn_at_boundary,
    signed_distance_to_polygon,
    would_exit,
)


class RobustPointTests(unittest.TestCase):
    """Uśrednianie 20 odczytów OCR → jeden dokładny punkt (mediana, odporne)."""

    def test_median_of_readings(self) -> None:
        reads = [(100, 200), (101, 201), (99, 199), (100, 200), (101, 200)]
        est = robust_point(reads)
        self.assertEqual(est.point, (100, 200))   # mediana per-oś
        self.assertEqual(est.samples, 5)

    def test_skips_none_misreads(self) -> None:
        est = robust_point([(100, 200), None, (100, 200), None])
        self.assertEqual(est.point, (100, 200))
        self.assertEqual(est.samples, 2)

    def test_outlier_rejected_with_threshold(self) -> None:
        # jeden gruby misread ×... odrzucony przez reject_beyond
        reads = [(100, 200)] * 5 + [(100, 9999)]
        est = robust_point(reads, reject_beyond=50.0)
        self.assertEqual(est.point, (100, 200))
        self.assertEqual(est.samples, 5)          # outlier wyrzucony
        self.assertLess(est.spread, 1.0)

    def test_spread_flags_unstable(self) -> None:
        est = robust_point([(100, 200), (120, 200), (80, 200)])
        self.assertGreater(est.spread, 15.0)      # duzy rozrzut = niepewny punkt

    def test_empty_none(self) -> None:
        self.assertIsNone(robust_point([None, None]))


class FromPerimeterTests(unittest.TestCase):
    def test_ordered_points_become_polygon(self) -> None:
        pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        b = FarmBoundary.from_perimeter(pts)
        self.assertEqual(b.source, "perimeter_walk")
        self.assertEqual(b.polygon, [(0, 0), (10, 0), (10, 10), (0, 10)])

    def test_dense_points_thinned_by_min_gap(self) -> None:
        # 20 fps daje gęste prawie-identyczne punkty -> zlewane
        dense = [(0, 0), (0.3, 0), (0.6, 0), (10, 0), (10, 9.5), (10, 10)]
        b = FarmBoundary.from_perimeter(dense, min_gap=2.0)
        self.assertLess(len(b.polygon), len(dense))
        self.assertIn((0, 0), b.polygon)
        self.assertIn((10, 0), b.polygon)

    def test_closing_duplicate_removed(self) -> None:
        b = FarmBoundary.from_perimeter([(0, 0), (10, 0), (5, 8), (0.1, 0.1)], min_gap=2.0)
        # ostatni punkt blisko pierwszego = domkniecie -> usuniety
        self.assertEqual(len(b.polygon), 3)

    def test_bounds_rejects_systematic_ocr_misread(self) -> None:
        # cyfra-wstawka OCR: 41→4137 (cala seria klatek zgodna, spread≈0 -> spread-gate
        # NIE lapie); sanity-koperta odrzuca punkt fizycznie niemozliwy.
        pts = [(40, 67), (4137, 70), (50, 71), (43, 6), (39, 70)]
        b = FarmBoundary.from_perimeter(pts, min_gap=1.0, bounds=(30, 60, 60, 80))
        self.assertNotIn((4137, 70), b.polygon)   # X poza koperta
        self.assertNotIn((43, 6), b.polygon)       # Y poza koperta
        self.assertEqual(len(b.polygon), 3)        # zostaja 3 zdrowe punkty

    def test_bounds_none_keeps_all(self) -> None:
        # bez koperty zachowanie jak dawniej (kompatybilnosc wsteczna)
        b = FarmBoundary.from_perimeter([(0, 0), (9999, 0), (5, 8)], min_gap=1.0)
        self.assertEqual(len(b.polygon), 3)

# Kwadrat 0..10 (CCW) do testów geometrii.
SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


class HullTests(unittest.TestCase):
    def test_hull_of_square_with_interior_points(self) -> None:
        pts = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5), (3, 7), (8, 2)]
        hull = convex_hull(pts)
        self.assertEqual(set(hull), {(0, 0), (10, 0), (10, 10), (0, 10)})  # wnętrze odrzucone

    def test_degenerate_few_points(self) -> None:
        self.assertEqual(convex_hull([(1, 1), (2, 2)]), [(1, 1), (2, 2)])

    def test_hull_is_tighter_than_bbox(self) -> None:
        # trójkąt: hull pole 50 < bbox 100 (otoczka ciaśniejsza niż prostokąt)
        b = FarmBoundary(convex_hull([(0, 0), (10, 0), (0, 10), (2, 2)]))
        self.assertAlmostEqual(b.area(), 50.0)


class PointInPolygonTests(unittest.TestCase):
    def test_inside(self) -> None:
        self.assertTrue(point_in_polygon((5, 5), SQUARE))

    def test_outside(self) -> None:
        self.assertFalse(point_in_polygon((15, 5), SQUARE))
        self.assertFalse(point_in_polygon((5, -1), SQUARE))

    def test_too_few_points(self) -> None:
        self.assertFalse(point_in_polygon((1, 1), [(0, 0), (1, 1)]))


class DistanceTests(unittest.TestCase):
    def test_dist_to_edge(self) -> None:
        self.assertAlmostEqual(dist_to_boundary((5, 1), SQUARE), 1.0)   # 1u od dolnej krawędzi
        self.assertAlmostEqual(dist_to_boundary((2, 5), SQUARE), 2.0)   # 2u od lewej

    def test_near_boundary_margin(self) -> None:
        self.assertTrue(near_boundary((5, 1), SQUARE, margin=2.0))      # blisko krawędzi
        self.assertFalse(near_boundary((5, 5), SQUARE, margin=2.0))     # środek = nie blisko
        self.assertTrue(near_boundary((5, 12), SQUARE, margin=2.0))     # poza = zawsze „blisko"


class SignedDistanceTests(unittest.TestCase):
    """Odległość z kierunkiem: + wewnątrz, − poza wielokątem (rdzeń marginesu świadomego ruchu)."""

    def test_inside_is_positive(self) -> None:
        self.assertAlmostEqual(signed_distance_to_polygon((9.5, 5), SQUARE), 0.5)   # 0.5u od prawej
        self.assertAlmostEqual(signed_distance_to_polygon((5, 5), SQUARE), 5.0)     # środek

    def test_outside_is_negative(self) -> None:
        self.assertAlmostEqual(signed_distance_to_polygon((12, 5), SQUARE), -2.0)   # 2u za prawą


class TurnDecisionTests(unittest.TestCase):
    def test_would_exit(self) -> None:
        self.assertTrue(would_exit((9, 5), (11, 5), SQUARE))   # krok wychodzi w prawo
        self.assertFalse(would_exit((5, 5), (6, 5), SQUARE))   # krok zostaje w środku

    def test_should_turn_when_step_exits(self) -> None:
        self.assertTrue(should_turn_at_boundary((9, 5), (11, 5), SQUARE, margin=1.0))

    def test_should_turn_when_in_margin_toward_edge(self) -> None:
        # w pasie granicy I krok zbliża do krawędzi (9.5→9.8, d 0.5→0.2) → skręć (nadal blokowane)
        self.assertTrue(should_turn_at_boundary((9.5, 5), (9.8, 5), SQUARE, margin=2.0))

    def test_no_turn_when_stepping_inward(self) -> None:
        # PUŁAPKA MARGINESU (NAV_BOUNDARY_LIVELOCK_FIX): w pasie, ale krok DO ŚRODKA
        # (9.5→9.2, d 0.5→0.8 rośnie) → PUŚĆ, inaczej margin>krok zatrzaskuje bota.
        self.assertFalse(should_turn_at_boundary((9.5, 5), (9.2, 5), SQUARE, margin=2.0))

    def test_should_not_turn_when_in_margin_but_step_goes_inward(self) -> None:
        self.assertFalse(should_turn_at_boundary((9.5, 5), (8.0, 5), SQUARE, margin=2.0))

    def test_no_turn_in_open_interior(self) -> None:
        self.assertFalse(should_turn_at_boundary((5, 5), (6, 5), SQUARE, margin=2.0))


class ExpandToIncludeTests(unittest.TestCase):
    """Auto-rozszerzenie granicy gdy bot zobaczy sklep poza nią (samokorekta)."""

    def test_external_point_becomes_inside(self) -> None:
        grown = expand_to_include(SQUARE, (12, 5), margin=1.0)
        self.assertTrue(point_in_polygon((12, 5), grown))   # sklep teraz wewnątrz
        self.assertEqual(len(grown), len(SQUARE) + 1)        # +1 wierzchołek

    def test_interior_point_no_change(self) -> None:
        self.assertEqual(expand_to_include(SQUARE, (5, 5)), SQUARE)

    def test_insert_is_local_keeps_other_vertices(self) -> None:
        grown = expand_to_include(SQUARE, (12, 5), margin=0.0)
        for v in SQUARE:                                     # stare wierzchołki zostają
            self.assertIn(v, grown)


class GrownToIncludeTests(unittest.TestCase):
    BOUNDS = (-5.0, 20.0, -5.0, 20.0)

    def test_grows_on_nearby_external_shop(self) -> None:
        b = FarmBoundary(list(SQUARE), source="perimeter_walk")
        b2, grew = b.grown_to_include((12, 5), bounds=self.BOUNDS, max_jump=10.0)
        self.assertTrue(grew)
        self.assertTrue(b2.contains((12, 5)))
        self.assertEqual(b2.source, "perimeter_walk")        # źródło zachowane

    def test_no_grow_for_interior_shop(self) -> None:
        b = FarmBoundary(list(SQUARE))
        b2, grew = b.grown_to_include((5, 5), bounds=self.BOUNDS)
        self.assertFalse(grew)
        self.assertIs(b2, b)

    def test_sanity_bounds_block_ocr_misread(self) -> None:
        # sklep „na" X=4137 to misread OCR, NIE realny sklep za granicą -> ignoruj
        b = FarmBoundary(list(SQUARE))
        b2, grew = b.grown_to_include((4137, 5), bounds=self.BOUNDS, max_jump=None)
        self.assertFalse(grew)
        self.assertEqual(b2.polygon, SQUARE)

    def test_max_jump_blocks_far_teleport(self) -> None:
        # w kopercie, ale 30u za granicą = nierealny skok -> nie rośnij
        b = FarmBoundary(list(SQUARE))
        b2, grew = b.grown_to_include((19, 5), bounds=self.BOUNDS, max_jump=5.0)
        self.assertFalse(grew)

    def test_max_jump_allows_just_outside(self) -> None:
        b = FarmBoundary(list(SQUARE))
        _, grew = b.grown_to_include((11, 5), bounds=self.BOUNDS, max_jump=5.0)
        self.assertTrue(grew)                                # 1u za granicą = realny nowy sklep


class FarmBoundaryIOTests(unittest.TestCase):
    def test_from_shops_and_contains(self) -> None:
        b = FarmBoundary.from_shops([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
        self.assertEqual(b.source, "hull_of_shops")
        self.assertTrue(b.contains((5, 5)))
        self.assertFalse(b.contains((20, 20)))

    def test_save_load_roundtrip(self) -> None:
        import tempfile
        from pathlib import Path
        b = FarmBoundary(convex_hull([(0, 0), (10, 0), (5, 8)]), source="hand")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "farm_map.json"
            b.save(p)
            b2 = FarmBoundary.load(p)
        self.assertIsNotNone(b2)
        self.assertEqual(b2.source, "hand")
        self.assertEqual(set(b2.polygon), set(b.polygon))

    def test_load_missing_returns_none(self) -> None:
        from pathlib import Path
        self.assertIsNone(FarmBoundary.load(Path("nie_ma_takiego_pliku.json")))


if __name__ == "__main__":
    unittest.main()
