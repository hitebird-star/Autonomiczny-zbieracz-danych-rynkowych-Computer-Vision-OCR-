"""Testy offline analizatora jakości capture (działka Claude)."""

from __future__ import annotations

import unittest

from scanner.analysis import capture_quality as cq
from scanner.models import ItemObservation, ShopScan


def _scan(scan_id: str, cells: dict[tuple[int, int], bool]) -> ShopScan:
    """Zbuduj skan: {(column, row): czy_ma_klatke}. True=hit, False=miss."""

    slots: dict[int, ItemObservation] = {}
    for (column, row), hit in cells.items():
        slot = row * 10 + column
        slots[slot] = ItemObservation(
            slot=slot,
            row=row,
            column=column,
            images=["tooltips/x.png"] if hit else [],
        )
    return ShopScan(
        scan_id=scan_id,
        occupied_slots=len(slots),
        captured_slots=sum(cells.values()),
        slots=slots,
    )


class ScanStatsTests(unittest.TestCase):
    def test_counts_hits_and_misses_per_cell(self) -> None:
        scan = _scan("s1", {(0, 0): True, (1, 0): False, (2, 0): True})
        stats = cq.scan_stats(scan)
        self.assertEqual(stats.occupied, 3)
        self.assertEqual(stats.hits, 2)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.hit_cells, ((0, 0), (2, 0)))
        self.assertEqual(stats.miss_cells, ((1, 0),))
        self.assertAlmostEqual(stats.hit_rate, 2 / 3)

    def test_empty_scan_has_zero_rate(self) -> None:
        stats = cq.scan_stats(ShopScan(scan_id="empty"))
        self.assertEqual(stats.occupied, 0)
        self.assertEqual(stats.hit_rate, 0.0)


class CellReliabilityTests(unittest.TestCase):
    def test_flipping_cell_is_evidence_of_nondeterminism(self) -> None:
        # Ta sama komórka (0,0): hit w run A, miss w run B -> mieszająca.
        a = _scan("a", {(0, 0): True, (5, 5): True})
        b = _scan("b", {(0, 0): False, (5, 5): True})
        reliability = cq.cell_reliability([cq.scan_stats(a), cq.scan_stats(b)])
        self.assertEqual(reliability[(0, 0)], (1, 2))
        self.assertEqual(reliability[(5, 5)], (2, 2))
        self.assertEqual(cq.flipping_cells(reliability), [(0, 0)])

    def test_chronic_miss_is_not_flipping(self) -> None:
        a = _scan("a", {(3, 3): False})
        b = _scan("b", {(3, 3): False})
        reliability = cq.cell_reliability([cq.scan_stats(a), cq.scan_stats(b)])
        self.assertEqual(cq.flipping_cells(reliability), [])
        chronic = [
            cell for cell, (hits, total) in reliability.items()
            if total >= 2 and hits == 0
        ]
        self.assertEqual(chronic, [(3, 3)])

    def test_single_appearance_is_not_flipping(self) -> None:
        # Jedno wystąpienie nie wystarcza, by orzec niedeterminizm.
        reliability = cq.cell_reliability([cq.scan_stats(_scan("a", {(7, 7): False}))])
        self.assertEqual(cq.flipping_cells(reliability), [])


class ReportTests(unittest.TestCase):
    def test_random_verdict_when_cells_flip(self) -> None:
        a = _scan("a", {(0, 0): True, (1, 0): False})
        b = _scan("b", {(0, 0): False, (1, 0): True})
        report = cq.format_report([cq.scan_stats(a), cq.scan_stats(b)])
        self.assertIn("LOSOWE", report)
        self.assertIn("zawodny input", report)

    def test_spatial_verdict_when_cells_chronic(self) -> None:
        a = _scan("a", {(0, 0): True, (9, 9): False})
        b = _scan("b", {(0, 0): True, (9, 9): False})
        report = cq.format_report([cq.scan_stats(a), cq.scan_stats(b)])
        self.assertIn("PRZESTRZENNE", report)

    def test_empty_report_is_safe(self) -> None:
        self.assertIn("Brak skanów", cq.format_report([]))


if __name__ == "__main__":
    unittest.main()
