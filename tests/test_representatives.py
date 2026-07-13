"""Testy Phase B (wybór reprezentantów stosu) — funkcja czysta, bez gry.

Pokrywa: reps_for_size (floor/big_quorum/granice), select_representatives
(singleton, brak icon_group, podział reps/deferred, materiał na konsensus),
oraz pomiar oszczędności (savings_table, group_sizes_from_manifest)."""

from __future__ import annotations

import unittest
from collections import Counter

from scanner.analysis.representatives import (
    CONSENSUS_MIN,
    DEFAULT_QUORUM,
    GroupSelection,
    group_sizes_from_manifest,
    reps_for_size,
    savings_table,
    select_representatives,
)


class RepsForSizeTest(unittest.TestCase):
    def test_maly_stos_hoveruje_wszystkich(self):
        self.assertEqual(reps_for_size(1), 1)
        self.assertEqual(reps_for_size(2), 2)
        self.assertEqual(reps_for_size(3), 3)

    def test_duzy_stos_ucina_do_kworum(self):
        self.assertEqual(reps_for_size(20, quorum=3), 3)
        self.assertEqual(reps_for_size(40, quorum=3), 3)

    def test_pusta_grupa_to_zero(self):
        self.assertEqual(reps_for_size(0), 0)

    def test_big_quorum_czyta_wiecej_dla_duzych(self):
        # poniżej progu: zwykłe kworum; od progu: wyższe kworum
        self.assertEqual(reps_for_size(5, quorum=3, big_threshold=8, big_quorum=5), 3)
        self.assertEqual(reps_for_size(20, quorum=3, big_threshold=8, big_quorum=5), 5)

    def test_plan_polityka_1_2(self):
        # plan rynku: 1 reprezentant, 2 dla grup >=8
        kw = {"quorum": 1, "big_threshold": 8, "big_quorum": 2}
        self.assertEqual(reps_for_size(4, **kw), 1)
        self.assertEqual(reps_for_size(8, **kw), 2)
        self.assertEqual(reps_for_size(40, **kw), 2)

    def test_quorum_zero_blad(self):
        with self.assertRaises(ValueError):
            reps_for_size(5, quorum=0)

    def test_big_quorum_ponizej_floor_blad(self):
        with self.assertRaises(ValueError):
            reps_for_size(10, quorum=3, big_threshold=8, big_quorum=2)


class SelectTest(unittest.TestCase):
    def test_singleton_jest_wlasnym_reprezentantem(self):
        sel = select_representatives({7: [5]})
        self.assertEqual(sel[7].representatives, (5,))
        self.assertEqual(sel[7].deferred, ())
        self.assertFalse(sel[7].has_consensus_material)

    def test_brak_icon_group_kazdy_slot_osobno(self):
        sel = select_representatives({None: [3, 1, 2]})
        # None: hoveruj wszystkie (brak dedupu bez klucza), posortowane
        self.assertEqual(sel[None].representatives, (1, 2, 3))
        self.assertEqual(sel[None].deferred, ())

    def test_duza_grupa_reps_i_deferred(self):
        members = list(range(10, 30))  # 20 slotów
        sel = select_representatives({3: members}, quorum=3)
        g = sel[3]
        self.assertEqual(g.representatives, (10, 11, 12))  # 3 najniższe
        self.assertEqual(len(g.deferred), 17)
        self.assertTrue(g.has_consensus_material)

    def test_reps_sa_posortowane_deterministycznie(self):
        sel = select_representatives({1: [30, 10, 20, 40]}, quorum=2)
        self.assertEqual(sel[1].representatives, (10, 20))
        self.assertEqual(sel[1].deferred, (30, 40))

    def test_reps_i_deferred_rozlaczne_i_pelne(self):
        members = list(range(50))
        sel = select_representatives({9: members}, quorum=3)
        g = sel[9]
        self.assertEqual(set(g.representatives) | set(g.deferred), set(members))
        self.assertEqual(set(g.representatives) & set(g.deferred), set())


class SavingsTest(unittest.TestCase):
    def test_group_sizes_z_manifestu_grupuje_po_icon_group(self):
        manifest = {"slots": {
            "0": {"icon_group": 1, "images": ["a.png"]},
            "1": {"icon_group": 1, "images": ["b.png"]},
            "2": {"icon_group": 2, "images": ["c.png"]},
            "3": {"icon_group": None, "images": ["d.png"]},   # singleton
            "4": {"icon_group": 1, "images": []},              # failed capture -> pomiń
        }}
        sizes = sorted(group_sizes_from_manifest(manifest))
        # grupa1=2 (sloty 0,1; slot4 bez obrazu nie liczony), grupa2=1, singleton=1
        self.assertEqual(sizes, [1, 1, 2])

    def test_savings_floor1_odslania_wszystkie_naddatki(self):
        # 1 grupa rozmiaru 10
        counts = Counter({10: 1})
        rows = {r.policy: r for r in savings_table(
            counts, [("f1", {"quorum": 1}), ("f3", {"quorum": 3})])}
        # floor=1: 1 hover z 10 -> 90% oszczędności, ale 9 slotów odsłoniętych
        self.assertEqual(rows["f1"].hovers, 1)
        self.assertEqual(rows["f1"].exposed_slots, 9)
        self.assertEqual(rows["f1"].groups_no_consensus, 1)
        # floor=3: 3 hovery z 10 -> 70% oszczędności, 0 odsłoniętych (konsensus jest)
        self.assertEqual(rows["f3"].hovers, 3)
        self.assertEqual(rows["f3"].exposed_slots, 0)
        self.assertEqual(rows["f3"].groups_no_consensus, 0)

    def test_baseline_to_suma_slotow(self):
        counts = Counter({2: 3, 5: 1})  # 3 grupy po 2 + 1 grupa po 5 = 11 slotów
        rows = savings_table(counts, [("f3", {"quorum": 3})])
        self.assertEqual(rows[0].baseline, 11)

    def test_singletony_nie_daja_oszczednosci_ani_ryzyka(self):
        counts = Counter({1: 5})  # 5 singletonów
        rows = savings_table(counts, [("f1", {"quorum": 1}), ("f3", {"quorum": 3})])
        for r in rows:
            self.assertEqual(r.hovers, 5)          # singleton zawsze hoverowany
            self.assertEqual(r.saved_pct, 0.0)
            self.assertEqual(r.exposed_slots, 0)   # rozmiar 1 < CONSENSUS_MIN


if __name__ == "__main__":
    unittest.main()
