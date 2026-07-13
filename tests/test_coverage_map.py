from __future__ import annotations

import unittest

from scanner.analysis.coverage_map import CoverageMap

# Koperta Glevia Farm (konwencja Claude: x_min, x_max, y_min, y_max).
FARM = (348.0, 501.0, 672.0, 794.0)


class GeometryTests(unittest.TestCase):
    def test_grid_dims(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        # (501-348)/20 = 7.65 -> 8 kolumn; (794-672)/20 = 6.1 -> 7 rzedow
        self.assertEqual(m.total_cells, 8 * 7)

    def test_cell_of_and_center_roundtrip(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        self.assertEqual(m.cell_of((348, 672)), (0, 0))            # rog
        self.assertEqual(m.cell_of((348 + 25, 672 + 25)), (1, 1))  # +25u = komorka (1,1)
        cx, cy = m.cell_of((400, 700))
        cxx, cyy = m.cell_of(m.cell_center((cx, cy)))              # srodek wraca do tej samej
        self.assertEqual((cx, cy), (cxx, cyy))

    def test_out_of_envelope_clamped(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        self.assertEqual(m.cell_of((100, 100)), (0, 0))            # ponizej min
        self.assertEqual(m.cell_of((9999, 9999)), (7, 6))          # powyzej max -> ostatnia

    def test_bad_envelope_raises(self) -> None:
        with self.assertRaises(ValueError):
            CoverageMap((501, 348, 672, 794))   # x_min > x_max
        with self.assertRaises(ValueError):
            CoverageMap(FARM, cell_size=0)


class CoverageStateTests(unittest.TestCase):
    def test_mark_and_is_covered(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        self.assertFalse(m.is_covered((400, 700)))
        m.mark((400, 700))
        self.assertTrue(m.is_covered((400, 700)))
        self.assertTrue(m.is_cell_covered(m.cell_of((400, 700))))  # komorke wprost

    def test_mark_radius_marks_neighbourhood(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        m.mark((400, 700), radius_cells=1)
        c = m.cell_of((400, 700))
        # srodek + do 8 sasiadow (mniej przy krawedzi)
        self.assertGreaterEqual(m.covered_count, 4)
        self.assertTrue(m.is_covered((400 - 20, 700)))

    def test_coverage_fraction_and_complete(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        self.assertAlmostEqual(m.coverage_fraction(), 0.0)
        for c in m.all_cells():
            m.mark(m.cell_center(c))
        self.assertTrue(m.is_complete())
        self.assertAlmostEqual(m.coverage_fraction(), 1.0)
        self.assertEqual(m.remaining(), 0)


class NextTargetTests(unittest.TestCase):
    def test_nearest_uncovered(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        m.mark((355, 678))  # pokryj rog (0,0)
        pos = (355, 678)
        tgt = m.next_target(pos, order="nearest")
        self.assertIsNotNone(tgt)
        # cel to NIE pokryta komorka (0,0); najblizsza to sasiad
        self.assertNotEqual(m.cell_of(tgt), (0, 0))
        # i jest blisko (sasiednia komorka, dystans ~< 2*cell)
        self.assertLess((tgt[0] - pos[0]) ** 2 + (tgt[1] - pos[1]) ** 2, (2 * 20.0) ** 2)

    def test_returns_none_when_complete(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        for c in m.all_cells():
            m.mark(m.cell_center(c))
        self.assertIsNone(m.next_target((400, 700)))

    def test_boustrophedon_serpentine_order(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        # rzad 0 pusty -> pierwszy cel w rzedzie 0, najmniejsze cx (rzad parzysty L->P)
        t0 = m.next_target((400, 700), order="boustrophedon")
        self.assertEqual(m.cell_of(t0), (0, 0))
        # pokryj caly rzad 0 -> nastepny cel w rzedzie 1, od PRAWEJ (nieparzysty)
        for cx in range(8):
            m.mark(m.cell_center((cx, 0)))
        t1 = m.next_target((400, 700), order="boustrophedon")
        self.assertEqual(m.cell_of(t1)[1], 1)
        self.assertEqual(m.cell_of(t1)[0], 7)  # od prawej

    def test_target_drives_toward_uncovered_region(self) -> None:
        # pokryj lewa polowe; cel powinien byc po prawej (wieksze x)
        m = CoverageMap(FARM, cell_size=20.0)
        for c in m.all_cells():
            if c[0] <= 3:
                m.mark(m.cell_center(c))
        tgt = m.next_target((360, 700), order="nearest")
        self.assertGreater(tgt[0], 360)  # ciagnie w prawo, ku niepokrytemu

    def test_path_to_target_routes_around_concave_wall_one_hop_at_a_time(self) -> None:
        m = CoverageMap((0, 100, 0, 80), cell_size=20.0)
        # Tylko (2,0) pozostaje pending. Bez BFS planner wybralby go po
        # dystansie i jechal po prostej przez (1,0), ktore jest sciana.
        for cell in m.all_cells():
            if cell != (2, 0):
                m.mark(m.cell_center(cell))
        m.mark_no_go((1, 0))
        m.mark_no_go((1, 1))

        path = m.path_to_next_target((10, 10))

        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 0))
        self.assertEqual(path[1], (0, 1))
        self.assertEqual(m.next_reachable_hop((10, 10)), m.cell_center((0, 1)))

    def test_path_returns_none_for_pending_cell_in_other_component(self) -> None:
        m = CoverageMap((0, 100, 0, 80), cell_size=20.0)
        for cell in m.all_cells():
            if cell != (2, 0):
                m.mark(m.cell_center(cell))
        for row in range(4):
            m.mark_no_go((1, row))

        self.assertIsNone(m.path_to_next_target((10, 10)))
        self.assertIsNone(m.next_reachable_hop((10, 10)))


class AntiRescanTests(unittest.TestCase):
    """Faza 2: liczniki skanów/duplikatów + decyzja pomiń-klik (anty-reskan)."""

    def test_record_scan_counts_and_covers(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        m.record_scan((400, 700), duplicate=False)
        m.record_scan((400, 700), duplicate=False)
        m.record_scan((400, 700), duplicate=True)
        c = m.cell_of((400, 700))
        self.assertEqual(m.scans_in_cell(c), 2)
        self.assertEqual(m.dups_in_cell(c), 1)
        self.assertTrue(m.is_cell_covered(c))   # skan też pokrywa

    def test_cell_exhausted_by_dup_floor(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        c = m.cell_of((400, 700))
        m.record_scan((400, 700), duplicate=True)
        self.assertFalse(m.cell_exhausted(c, dup_floor=2))  # 1 dup < floor
        m.record_scan((400, 700), duplicate=True)
        self.assertTrue(m.cell_exhausted(c, dup_floor=2))   # 2 dups = przebrane

    def test_cell_exhausted_by_expected(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        c = m.cell_of((400, 700))
        for _ in range(4):
            m.record_scan((400, 700), duplicate=False)
        self.assertTrue(m.cell_exhausted(c, expected=4))
        self.assertFalse(m.cell_exhausted(c, expected=5))

    def test_should_skip_click(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        self.assertFalse(m.should_skip_click(pos))            # świeża komórka — klikaj
        m.record_scan(pos, duplicate=True)
        m.record_scan(pos, duplicate=True)
        self.assertTrue(m.should_skip_click(pos, dup_floor=2))  # pokryta + 2 dups → pomiń

    def test_skip_only_when_both_covered_and_exhausted(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (450, 760)
        m.record_scan(pos, duplicate=False)   # pokryta, ale 0 dups → NIE wyczerpana
        self.assertFalse(m.should_skip_click(pos, dup_floor=2))


class BoundaryAwareTests(unittest.TestCase):
    """Linia akceptowalności KSZTAŁTUJE pokrycie: komórki poza wielokątem wykluczone z next_target."""

    # Mały wielokąt obejmujący tylko lewy-dolny róg koperty (ok. x 348-410, y 672-720).
    SMALL = [(350.0, 674.0), (408.0, 674.0), (408.0, 718.0), (350.0, 718.0)]

    def test_excluded_cells_reduce_farm_count(self) -> None:
        full = CoverageMap(FARM, cell_size=20.0)
        clipped = CoverageMap(FARM, cell_size=20.0, boundary=self.SMALL)
        self.assertEqual(clipped.total_cells, full.total_cells)      # siatka bez zmian
        self.assertLess(clipped.farm_cells, full.total_cells)        # ale farma mniejsza
        self.assertGreater(clipped.farm_cells, 0)

    def test_next_target_never_outside_boundary(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0, boundary=self.SMALL)
        seen = set()
        # przejdź pełny obchód: cel -> oznacz -> kolejny, aż None
        pos = (360, 680)
        for _ in range(200):
            t = m.next_target(pos)
            if t is None:
                break
            self.assertFalse(m.is_excluded(m.cell_of(t)))            # cel ZAWSZE w farmie
            seen.add(m.cell_of(t))
            m.mark(t)
            pos = t
        self.assertTrue(seen)                                        # jakieś cele były

    def test_complete_ignores_outside_cells(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0, boundary=self.SMALL)
        # pokryj TYLKO komórki farmy → kompletne, mimo że siatka ma więcej komórek
        for c in m.all_cells():
            if not m.is_excluded(c):
                m.mark(m.cell_center(c))
        self.assertTrue(m.is_complete())
        self.assertIsNone(m.next_target((360, 680)))
        self.assertAlmostEqual(m.coverage_fraction(), 1.0)

    def test_degenerate_boundary_excludes_nothing(self) -> None:
        # wielokąt nie obejmujący ŻADNEGO środka komórki → ignorowany (pełna siatka, nie zerowa farma)
        tiny = [(348.1, 672.1), (348.2, 672.1), (348.2, 672.2)]
        m = CoverageMap(FARM, cell_size=20.0, boundary=tiny)
        self.assertEqual(m.farm_cells, m.total_cells)               # fallback: nic nie wykluczone

    def test_boundary_none_backward_compatible(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        self.assertEqual(m.farm_cells, m.total_cells)
        for c in m.all_cells():
            self.assertFalse(m.is_excluded(c))


class BoundaryMarginTests(unittest.TestCase):
    """Pas przygraniczny (boundary_margin) = NIE-cel, gdy pokryty ze standu o rząd głębiej.

    Router (BFS) i guard ruchu (should_turn_at_boundary, margines) muszą mieć JEDNĄ
    osiągalność — inaczej router celuje w pas, którego guard nie wpuści → pętla.
    """

    # Prostokąt obejmujący środki komórek cx 0..5, cy 0..4.
    RECT = [(350.0, 674.0), (470.0, 674.0), (470.0, 774.0), (350.0, 774.0)]

    def test_margin_zero_is_backward_compatible(self) -> None:
        base = CoverageMap(FARM, cell_size=20.0, boundary=self.RECT)
        margin0 = CoverageMap(FARM, cell_size=20.0, boundary=self.RECT, boundary_margin=0.0)
        self.assertEqual(base.farm_cells, margin0.farm_cells)   # 0 = jak dawniej
        self.assertFalse(margin0.is_excluded((1, 0)))           # przygraniczna NIE wykluczona

    def test_margin_excludes_near_edge_with_deep_neighbor(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0, boundary=self.RECT, boundary_margin=10.0)
        self.assertTrue(m.is_excluded((1, 0)))      # dolny pas, ma głębszego sąsiada (1,1)
        self.assertTrue(m.is_done((1, 0)))          # wykluczona = od razu „done"

    def test_margin_keeps_deep_interior(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0, boundary=self.RECT, boundary_margin=10.0)
        self.assertFalse(m.is_excluded((2, 2)))     # głęboko w środku = nadal cel

    def test_margin_does_not_orphan_cornerless_cell(self) -> None:
        # róg (0,0): wszyscy sąsiedzi też w pasie (brak głębszego) → ZOSTAJE celem (zero osierocenia)
        m = CoverageMap(FARM, cell_size=20.0, boundary=self.RECT, boundary_margin=10.0)
        self.assertFalse(m.is_excluded((0, 0)))

    def test_margin_shrinks_targets_and_target_never_in_margin(self) -> None:
        base = CoverageMap(FARM, cell_size=20.0, boundary=self.RECT)
        m = CoverageMap(FARM, cell_size=20.0, boundary=self.RECT, boundary_margin=10.0)
        self.assertLess(m.farm_cells, base.farm_cells)          # pas odjęty z celów
        pos = (400, 722)
        for _ in range(200):
            t = m.next_target(pos)
            if t is None:
                break
            self.assertFalse(m.is_excluded(m.cell_of(t)))       # cel NIGDY w pasie/poza
            m.mark(t)
            pos = t


class MarkUnreachableTests(unittest.TestCase):
    """Cel nieosiągalny w biegu: NIE-cel (is_done), ale PRZEJEZDNY w BFS → zero fragmentacji.

    To naprawa „pętli w i s": router przestaje wybierać cel za granicą, ale graf się nie rozpada
    (inaczej niż `no_go`, który blokuje też przejazd).
    """

    def test_unreachable_is_done_but_not_blocked(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        c = (3, 2)
        m.mark_unreachable(c)
        self.assertTrue(m.is_done(c))             # przestaje być celem
        self.assertFalse(m.is_blocked(c))         # ale BFS nadal przez nią przechodzi
        self.assertNotIn(c, m.uncovered_cells())  # zniknęła z celów

    def test_unreachable_not_repicked_by_next_target(self) -> None:
        m = CoverageMap(FARM, cell_size=20.0)
        c = m.cell_of((418, 722))                 # cel z realnej „pętli w i s"
        m.mark_unreachable(c)
        seen, pos = set(), (418, 722)
        for _ in range(300):
            t = m.next_target(pos)
            if t is None:
                break
            seen.add(m.cell_of(t))
            m.mark(t)
            pos = t
        self.assertNotIn(c, seen)                 # NIGDY nie wybrany ponownie

    def test_unreachable_does_not_fragment_corridor(self) -> None:
        # korytarz 1-rzędowy: (0,0)..(5,0); cel ZA przeszkodą osiągalny TYLKO przez nią
        strip = [(350.0, 674.0), (470.0, 674.0), (470.0, 690.0), (350.0, 690.0)]
        m = CoverageMap(FARM, cell_size=20.0, boundary=strip)
        m.mark_unreachable((2, 0))                # NIE-cel w środku korytarza
        m.mark(m.cell_center((0, 0)))
        m.mark(m.cell_center((1, 0)))             # jedyne cele są ZA (2,0)
        path = m.path_to_next_target((358, 682))  # start w (0,0)
        self.assertIsNotNone(path)                # cel nadal osiągalny → brak fragmentacji
        self.assertIn((2, 0), path)               # ścieżka przechodzi PRZEZ nieosiągalny cel

    def test_no_go_would_fragment_same_corridor(self) -> None:
        # kontrast: no_go (is_blocked) ROZCINA korytarz — dowód, że mark_unreachable jest inne
        strip = [(350.0, 674.0), (470.0, 674.0), (470.0, 690.0), (350.0, 690.0)]
        m = CoverageMap(FARM, cell_size=20.0, boundary=strip)
        m.mark_no_go((2, 0))
        m.mark(m.cell_center((0, 0)))
        m.mark(m.cell_center((1, 0)))
        self.assertIsNone(m.path_to_next_target((358, 682)))   # no_go odcina cele za sobą


class DoneSaturationTests(unittest.TestCase):
    """C6 (§6b): DONE = wysycenie, nie dotknięcie — anti-przedwczesne-przejście."""

    def test_touched_cell_still_pending_until_exhausted(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        c = m.cell_of(pos)
        m.record_scan(pos, duplicate=False)             # dotknięta, ale 0 dups
        self.assertFalse(m.is_done(c, dup_floor=2))     # NIE gotowa (niewysycona)
        # tryb "covered" już ją pomija (stare zachowanie)...
        self.assertNotIn(c, [m.cell_of(t) for t in [m.next_target(pos, until="covered")] if t])
        # ...ale tryb "done" wciąż ją wskazuje jako cel (bot ma dokończyć sektor)
        self.assertIn(c, m.pending_cells(dup_floor=2))

    def test_cell_done_after_saturation(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        c = m.cell_of(pos)
        m.record_scan(pos, duplicate=True)
        m.record_scan(pos, duplicate=True)              # 2 dups = wyczerpana
        self.assertTrue(m.is_done(c, dup_floor=2))
        self.assertNotIn(c, m.pending_cells(dup_floor=2))

    def test_next_target_done_returns_unsaturated_covered_cell(self):
        # jedyna „dotknięta ale niewysycona" komórka → until="done" ją zwraca, until="covered" nie
        m = CoverageMap(FARM, cell_size=20.0)
        for c in m.all_cells():                          # wszystko pokryte+wysycone…
            center = m.cell_center(c)
            m.record_scan(center, duplicate=True)
            m.record_scan(center, duplicate=True)
        # …poza jedną, którą tylko dotknięto (1 skan, 0 dups)
        target_cell = (3, 3)
        m._dups.pop(target_cell, None)
        m._dup_streak.pop(target_cell, None)   # G3: saturacja po serii → wyzeruj serię
        m._scans[target_cell] = 1
        self.assertIsNone(m.next_target((400, 700), until="covered"))   # wszystko „pokryte"
        t = m.next_target(m.cell_center(target_cell), until="done")
        self.assertEqual(m.cell_of(t), target_cell)                     # done-mode wraca dokończyć

    def test_all_done_only_when_every_cell_saturated(self):
        m = CoverageMap(FARM, cell_size=20.0)
        for c in m.all_cells():
            center = m.cell_center(c)
            m.record_scan(center, duplicate=True)
            m.record_scan(center, duplicate=True)
        self.assertTrue(m.all_done(dup_floor=2))
        # cofnij jedną komórkę do „niewysyconej" → already not all_done
        m._dup_streak[(0, 0)] = 0   # G3: saturacja po serii kolejnych duplikatów
        self.assertFalse(m.all_done(dup_floor=2))

    def test_excluded_cells_count_as_done(self):
        small = [(350.0, 674.0), (408.0, 674.0), (408.0, 718.0), (350.0, 718.0)]
        m = CoverageMap(FARM, cell_size=20.0, boundary=small)
        # komórka poza granicą jest „done" (nie blokuje all_done)
        outside = next(c for c in m.all_cells() if m.is_excluded(c))
        self.assertTrue(m.is_done(outside))
        self.assertNotIn(outside, m.pending_cells())


class BlockLogTests(unittest.TestCase):
    """C2/C3: logi blokad na mapie → no_go → border_adjustments (douczanie granicy)."""

    def test_record_and_count_blocks(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        c = m.cell_of(pos)
        m.record_block(pos, "failed_open")
        m.record_block(pos, "failed_open")
        m.record_block(pos, "goto_fail")
        self.assertEqual(m.blocks_in_cell(c, "failed_open"), 2)
        self.assertEqual(m.blocks_in_cell(c, "goto_fail"), 1)
        self.assertEqual(m.blocks_in_cell(c), 3)                 # łącznie
        self.assertEqual(m.is_cell_covered(c), False)            # blokada NIE pokrywa komórki

    def test_no_go_excluded_from_targets_and_done(self):
        m = CoverageMap(FARM, cell_size=20.0)
        c = (3, 3)
        self.assertIn(c, m.uncovered_cells())
        m.mark_no_go(c)
        self.assertTrue(m.is_no_go(c))
        self.assertNotIn(c, m.uncovered_cells())                 # nie jest celem
        self.assertNotIn(c, m.pending_cells())
        self.assertTrue(m.is_done(c))                            # no_go liczy się jak done
        self.assertLess(m.farm_cells, m.total_cells)             # zmniejsza mianownik

    def test_clear_no_go_preserves_coverage_and_scan_history(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        m.record_scan(pos, duplicate=False)
        blocked = (3, 3)
        m.mark_no_go(blocked)

        self.assertEqual(m.clear_no_go(), 1)
        self.assertFalse(m.is_no_go(blocked))
        self.assertTrue(m.is_covered(pos))
        self.assertEqual(m.scans_in_cell(m.cell_of(pos)), 1)

    def test_border_adjustments_flags_persistent_failure_no_shop(self):
        m = CoverageMap(FARM, cell_size=20.0)
        empty = (400, 700)                                       # ciągłe porażki, 0 sklepów
        ce = m.cell_of(empty)
        for _ in range(3):
            m.record_block(empty, "failed_open")
        # druga komórka: też porażki, ALE znalazła sklep → NIE no_go
        busy = (460, 740)
        cb = m.cell_of(busy)
        for _ in range(3):
            m.record_block(busy, "goto_fail")
        m.record_scan(busy, duplicate=False)
        cands = m.border_adjustments(fail_floor=3)
        self.assertIn(ce, cands)                                 # pusta + ciągłe porażki
        self.assertNotIn(cb, cands)                              # znalazła sklep → zostaje celem

    def test_border_adjustments_respects_floor(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        m.record_block(pos, "failed_open")
        m.record_block(pos, "failed_open")                       # 2 < floor 3
        self.assertEqual(m.border_adjustments(fail_floor=3), [])

    def test_goto_failure_never_teaches_no_go(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        for _ in range(5):
            m.record_block(pos, "goto_fail")

        self.assertEqual(m.border_adjustments(fail_floor=3), [])

    def test_blocks_and_no_go_persist(self):
        import tempfile
        from pathlib import Path
        m = CoverageMap(FARM, cell_size=20.0)
        m.record_block((400, 700), "failed_open")
        m.mark_no_go((5, 5))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "coverage.json"
            m.save(p)
            m2 = CoverageMap.load(p)
        self.assertEqual(m2.blocks_in_cell(m2.cell_of((400, 700)), "failed_open"), 1)
        self.assertTrue(m2.is_no_go((5, 5)))


class PersistenceTests(unittest.TestCase):
    """C1: mapa żyje między biegami — save/load coverage.json."""

    def _populated(self) -> CoverageMap:
        m = CoverageMap(FARM, cell_size=20.0)
        m.record_scan((400, 700), duplicate=False)
        m.record_scan((400, 700), duplicate=True)
        m.mark((460, 740))
        return m

    def test_roundtrip_preserves_state(self) -> None:
        import tempfile
        from pathlib import Path
        m = self._populated()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "coverage.json"
            m.save(p)
            m2 = CoverageMap.load(p)
        self.assertIsNotNone(m2)
        self.assertEqual(m2.envelope, m.envelope)
        self.assertEqual(m2.cell_size, m.cell_size)
        self.assertEqual(m2._covered, m._covered)               # te same komórki pokryte
        c = m.cell_of((400, 700))
        self.assertEqual(m2.scans_in_cell(c), 1)
        self.assertEqual(m2.dups_in_cell(c), 1)

    def test_save_creates_parent_dir(self) -> None:
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "market_map" / "glevia_market" / "coverage.json"
            self._populated().save(p)                            # tworzy katalogi
            self.assertTrue(p.exists())

    def test_load_missing_returns_none(self) -> None:
        from pathlib import Path
        self.assertIsNone(CoverageMap.load(Path("nie_ma_coverage.json")))

    def test_load_recomputes_excluded_with_boundary(self) -> None:
        # granica podana przy load → _excluded liczone na NIEJ (nie serializowane)
        import tempfile
        from pathlib import Path
        small = [(350.0, 674.0), (408.0, 674.0), (408.0, 718.0), (350.0, 718.0)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "coverage.json"
            CoverageMap(FARM, cell_size=20.0).save(p)
            loaded = CoverageMap.load(p, boundary=small)
        self.assertLess(loaded.farm_cells, loaded.total_cells)   # granica zastosowana po load


class SaturationStreakG3(unittest.TestCase):
    """G3: saturacja po SERII kolejnych duplikatów, nie po sumie (gęsta komórka ~17 sklepów)."""

    def test_new_shop_resets_dup_streak(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700); c = m.cell_of(pos)
        m.record_scan(pos, duplicate=True)
        m.record_scan(pos, duplicate=True)            # seria = 2
        self.assertEqual(m.dup_streak_in_cell(c), 2)
        m.record_scan(pos, duplicate=False)           # NOWY sklep → seria zerowana
        self.assertEqual(m.dup_streak_in_cell(c), 0)
        self.assertEqual(m.dups_in_cell(c), 2)        # suma dups zostaje

    def test_dense_cell_not_exhausted_by_scattered_dups(self):
        # nowe-przeplatane-duplikatami: suma dups rośnie, ale seria nie → NIE wyczerpana
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700); c = m.cell_of(pos)
        for _ in range(8):
            m.record_scan(pos, duplicate=False)       # nowy
            m.record_scan(pos, duplicate=True)        # duplikat (seria zawsze wraca do 1)
        self.assertEqual(m.dups_in_cell(c), 8)        # suma duża…
        self.assertFalse(m.cell_exhausted(c, dup_floor=2))   # …ale seria=1 < floor → produktywna
        self.assertIn(c, m.pending_cells(dup_floor=2))

    def test_consecutive_dups_exhaust(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700); c = m.cell_of(pos)
        m.record_scan(pos, duplicate=False)
        m.record_scan(pos, duplicate=True)
        m.record_scan(pos, duplicate=True)            # 2 znane Z RZĘDU → przebrane
        self.assertTrue(m.cell_exhausted(c, dup_floor=2))
        self.assertTrue(m.is_done(c, dup_floor=2))


class KnownFreshSaturationG2(unittest.TestCase):
    """G2: durable dedup (`duplicate_known_fresh`) liczy się do saturacji jak dup w-biegu."""

    def test_known_fresh_counts_toward_done(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700); c = m.cell_of(pos)
        m.record_known_fresh(pos)
        m.record_known_fresh(pos)                     # komórka pełna ZNANYCH (z innego biegu)
        self.assertTrue(m.is_done(c, dup_floor=2))    # → done, bot jej nie zapętla
        self.assertNotIn(c, m.pending_cells(dup_floor=2))

    def test_known_fresh_then_new_resets(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700); c = m.cell_of(pos)
        m.record_known_fresh(pos)
        m.record_scan(pos, duplicate=False)           # jednak nowy sklep → seria reset
        self.assertEqual(m.dup_streak_in_cell(c), 0)
        self.assertFalse(m.is_done(c, dup_floor=2))


class DupStreakPersistence(unittest.TestCase):
    def test_round_trip_preserves_streak(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700); c = m.cell_of(pos)
        m.record_scan(pos, duplicate=True)
        m.record_scan(pos, duplicate=True)
        m2 = CoverageMap.from_dict(m.to_dict())
        self.assertEqual(m2.dup_streak_in_cell(c), 2)
        self.assertTrue(m2.is_done(c, dup_floor=2))

    def test_legacy_map_without_streak_migrates_from_dups(self):
        # stara mapa (bez 'dup_streak') → seria przyjmuje sumę dups, saturacja zachowana
        legacy = {
            "envelope": list(FARM), "cell_size": 20.0,
            "covered": [[3, 1]], "scans": [], "dups": [[3, 1, 2]], "no_go": [],
            "blocks": [],
        }
        m = CoverageMap.from_dict(legacy)
        self.assertEqual(m.dup_streak_in_cell((3, 1)), 2)
        self.assertTrue(m.is_done((3, 1), dup_floor=2))


class CoveragePathG4(unittest.TestCase):
    """G4: pełny plan zamiatania (wężyk) + dynamiczny driver dowozi CAŁĄ farmę."""

    def test_path_covers_every_farm_cell_once(self):
        m = CoverageMap(FARM, cell_size=20.0)
        path = m.coverage_path()
        self.assertEqual(len(path), m.farm_cells)
        self.assertEqual(len(set(path)), len(path))            # bez duplikatów
        self.assertEqual(set(path), set(m.all_cells()))        # bez granicy = wszystkie

    def test_path_is_serpentine(self):
        m = CoverageMap(FARM, cell_size=20.0)
        path = m.coverage_path()
        row0 = [c[0] for c in path if c[1] == 0]
        row1 = [c[0] for c in path if c[1] == 1]
        self.assertEqual(row0, sorted(row0))                   # rząd 0: L→P
        self.assertEqual(row1, sorted(row1, reverse=True))     # rząd 1: P→L

    def test_path_excludes_no_go_and_boundary(self):
        small = [(350.0, 674.0), (450.0, 674.0), (450.0, 740.0), (350.0, 740.0)]
        m = CoverageMap(FARM, cell_size=20.0, boundary=small)
        m.mark_no_go((0, 0))
        path = m.coverage_path()
        self.assertNotIn((0, 0), path)                         # no_go pominięte
        self.assertTrue(all(not m.is_excluded(c) for c in path))   # poza-granicą pominięte

    def test_dynamic_until_done_covers_all_in_snake(self):
        # iterowanie next_target(until="done") + saturacja każdej → all_done, pokryte = cała farma
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (350, 674)
        visited = set()
        for _ in range(10_000):
            t = m.next_target(pos, order="boustrophedon", until="done")
            if t is None:
                break
            c = m.cell_of(t)
            visited.add(c)
            pos = t
            m.record_scan(t, duplicate=False)                  # 1 nowy
            m.record_scan(t, duplicate=True)
            m.record_scan(t, duplicate=True)                   # 2 znane z rzędu → saturacja
        self.assertTrue(m.all_done())
        self.assertEqual(visited, set(m.coverage_path()))


class TextureGuardG6(unittest.TestCase):
    """G6: stall→no_go (tekstura), edge_hit zdrowe; blocked_ahead = NIE PCHAJ w teksturę/poza kopertę."""

    def test_stall_promotes_no_go(self):
        m = CoverageMap(FARM, cell_size=20.0)
        stuck = (400, 700); c = m.cell_of(stuck)
        for _ in range(3):
            m.record_block(stuck, "stall")             # zacięcie = nieprzechodni obiekt
        self.assertIn(c, m.border_adjustments(fail_floor=3))

    def test_edge_hit_alone_is_not_texture(self):
        m = CoverageMap(FARM, cell_size=20.0)
        edge = (400, 700); c = m.cell_of(edge)
        for _ in range(5):
            m.record_block(edge, "edge_hit")           # zdrowe zawracanie na granicy
        self.assertNotIn(c, m.border_adjustments(fail_floor=3))

    def test_mixed_signals_sum_to_floor(self):
        m = CoverageMap(FARM, cell_size=20.0)
        p = (400, 700); c = m.cell_of(p)
        m.record_block(p, "stall"); m.record_block(p, "goto_fail"); m.record_block(p, "failed_open")
        self.assertNotIn(c, m.border_adjustments(fail_floor=3))
        m.record_block(p, "failed_open")
        self.assertIn(c, m.border_adjustments(fail_floor=3))   # stall + 2 failed_open

    def test_blocked_ahead_detects_no_go_cell(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        ahead_cell = m.cell_of((400 + m.cell_size, 700))       # komórka o krok w +X
        m.mark_no_go(ahead_cell)
        self.assertTrue(m.blocked_ahead(pos, (460, 700)))      # cel w +X → wbiega w no_go
        self.assertFalse(m.blocked_ahead(pos, (400, 660)))     # inny kierunek, czysto

    def test_blocked_ahead_detects_envelope_exit(self):
        m = CoverageMap(FARM, cell_size=20.0)
        x_min, x_max, y_min, y_max = FARM
        near_bottom = (400, y_min + 2)                         # tuż przy dolnej krawędzi
        self.assertTrue(m.blocked_ahead(near_bottom, (400, y_min - 50)))   # krok poza kopertę
        self.assertFalse(m.blocked_ahead((400, 730), (400, 760)))         # w głąb farmy → ok

    def test_blocked_ahead_clear_path(self):
        m = CoverageMap(FARM, cell_size=20.0)
        self.assertFalse(m.blocked_ahead((400, 700), (420, 720)))


class ForwardTargetG8(unittest.TestCase):
    """„Skanujemy przód, nie wokół": PUSTE komórki pierwsze + przednie przed tylnymi w next_target."""

    def test_prefer_uncovered_picks_empty_over_unsaturated_covered(self):
        # pokryta-ale-niewysycona komórka BLISKO vs pusta DALEJ → prefer_uncovered wybiera PUSTĄ
        m = CoverageMap(FARM, cell_size=20.0)
        near = (400, 700)
        m.record_scan(near, duplicate=False)              # dotknięta, niewysycona (0 dups)
        cnear = m.cell_of(near)
        # bez prefer: until="done" wraca do bliskiej niewysyconej
        t_default = m.next_target(near, until="done")
        self.assertEqual(m.cell_of(t_default), cnear)
        # z prefer_uncovered: omija ją, celuje w PUSTĄ (inną komórkę)
        t_pref = m.next_target(near, until="done", prefer_uncovered=True)
        self.assertNotEqual(m.cell_of(t_pref), cnear)
        self.assertNotIn(m.cell_of(t_pref), m._covered)

    def test_prefer_uncovered_falls_back_to_pending_when_no_empty(self):
        # brak pustych (wszystko dotknięte) → prefer_uncovered wraca do niewysyconej (domknięcie)
        m = CoverageMap(FARM, cell_size=20.0)
        for c in m.all_cells():
            m.record_scan(m.cell_center(c), duplicate=False)   # wszystko pokryte, nic wysycone
        t = m.next_target((400, 700), until="done", prefer_uncovered=True)
        self.assertIsNotNone(t)                            # nie None — domyka pokryte-niewysycone
        self.assertIn(m.cell_of(t), m.pending_cells())

    def test_heading_prefers_forward_cell_over_nearer_rear(self):
        # tylna komórka odrobinę bliżej, przednia dalej → heading wybiera PRZEDNIĄ
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (420.0, 700.0)
        # zostaw tylko dwie niepokryte: jedną z przodu (+x), jedną z tyłu (-x) bliżej
        keep_front = m.cell_of((460.0, 700.0))
        keep_rear = m.cell_of((400.0, 700.0))
        for c in m.all_cells():
            if c not in (keep_front, keep_rear):
                m.mark(m.cell_center(c))                   # pokryj resztę
        t = m.next_target(pos, until="covered", heading=(1.0, 0.0))
        self.assertEqual(m.cell_of(t), keep_front)         # przód mimo że tył bliżej
        # bez heading: wybiera bliższą (tył)
        t2 = m.next_target(pos, until="covered")
        self.assertEqual(m.cell_of(t2), keep_rear)

    def test_heading_none_matches_legacy_nearest(self):
        m = CoverageMap(FARM, cell_size=20.0)
        pos = (400, 700)
        self.assertEqual(
            m.next_target(pos, until="covered", heading=None),
            m.next_target(pos, until="covered"),
        )


class CoverageReentry(unittest.TestCase):
    def test_blocked_start_reenters_nearest_farm_cell_instead_of_none(self):
        boundary = [
            (380.0, 680.0),
            (460.0, 680.0),
            (460.0, 760.0),
            (380.0, 760.0),
        ]
        m = CoverageMap(FARM, cell_size=20.0, boundary=boundary)
        outside_pos = (500.0, 760.0)

        self.assertTrue(m.is_blocked(m.cell_of(outside_pos)))
        reentry = m.reentry_path(outside_pos)

        self.assertIsNotNone(reentry)
        assert reentry is not None
        self.assertFalse(m.is_blocked(reentry[-1]))

    def test_path_from_blocked_start_can_still_find_global_pending_target(self):
        boundary = [
            (380.0, 680.0),
            (460.0, 680.0),
            (460.0, 760.0),
            (380.0, 760.0),
        ]
        m = CoverageMap(FARM, cell_size=20.0, boundary=boundary)
        outside_pos = (500.0, 760.0)

        path = m.path_to_next_target(
            outside_pos,
            until="done",
            prefer_uncovered=True,
            dup_floor=4,
        )

        self.assertIsNotNone(path)
        assert path is not None
        self.assertFalse(m.is_blocked(path[0]))
        self.assertFalse(m.is_done(path[-1], dup_floor=4))


if __name__ == "__main__":
    unittest.main()
