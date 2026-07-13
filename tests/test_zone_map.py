"""Testy stref (Etap 3) — kafelkowanie koperty, lokalizacja punktu, binowanie
pokrycia, nasycenie K, DONE=oba sygnały, najbliższa-niedokończona, round-trip.
Bez gry. Koperta z planu: x 348-501, y 672-794, grid 3x3."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scanner.analysis.zone_map import (
    DEFAULT_SATURATION_K,
    Zone,
    ZoneMap,
    build_zones,
)

ENVELOPE = (348, 672, 501, 794)  # zmierzona pełna koperta rynku


class BuildZonesTest(unittest.TestCase):
    def test_3x3_daje_9_stref_z_nazwami(self):
        zones = build_zones(ENVELOPE, (3, 3))
        self.assertEqual(len(zones), 9)
        self.assertEqual([z.zone_id for z in zones],
                         [f"Z{i}" for i in range(1, 10)])

    def test_kafle_pokrywaja_koperte_bez_dziur(self):
        zones = build_zones(ENVELOPE, (3, 3))
        # lewy-górny zaczyna się w rogu koperty, prawy-dolny kończy w przeciwległym
        self.assertEqual(zones[0].box[:2], (348, 672))
        self.assertEqual(zones[-1].box[2:], (501, 794))

    def test_sasiednie_kafle_dziela_krawedz(self):
        zones = build_zones(ENVELOPE, (3, 3))
        z1, z2 = zones[0], zones[1]  # Z1 i Z2 w tym samym wierszu
        self.assertEqual(z1.box[2], z2.box[0])  # prawa krawędź Z1 = lewa Z2

    def test_1x1_to_cala_koperta(self):
        zones = build_zones(ENVELOPE, (1, 1))
        self.assertEqual(zones[0].box, ENVELOPE)

    def test_odwrocona_koperta_blad(self):
        with self.assertRaises(ValueError):
            build_zones((501, 794, 348, 672))

    def test_zerowa_siatka_blad(self):
        with self.assertRaises(ValueError):
            build_zones(ENVELOPE, (0, 3))


class ZoneForTest(unittest.TestCase):
    def setUp(self):
        self.zmap = ZoneMap(ENVELOPE, grid=(3, 3))

    def test_rog_lewy_gorny_to_Z1(self):
        self.assertEqual(self.zmap.zone_for(348, 672).zone_id, "Z1")

    def test_rog_prawy_dolny_to_Z9(self):
        self.assertEqual(self.zmap.zone_for(501, 794).zone_id, "Z9")

    def test_srodek_to_Z5(self):
        self.assertEqual(self.zmap.zone_for(424, 733).zone_id, "Z5")

    def test_punkt_poza_koperta_to_none(self):
        self.assertIsNone(self.zmap.zone_for(300, 672))
        self.assertIsNone(self.zmap.zone_for(348, 900))

    def test_kazdy_punkt_ma_dokladnie_jedna_strefe(self):
        # gęsta siatka próbek — każda trafia w dokładnie jedną strefę
        for x in range(348, 502, 7):
            for y in range(672, 795, 7):
                self.assertIsNotNone(self.zmap.zone_for(x, y))


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.zmap = ZoneMap(ENVELOPE, grid=(3, 3), subgrid=(2, 2))

    def test_jedna_pozycja_nie_pokrywa_calej_strefy(self):
        self.zmap.record_position(350, 674)  # róg Z1
        self.assertFalse(self.zmap.coverage_complete("Z1"))

    def test_wszystkie_podkomorki_pokrywaja(self):
        z1 = self.zmap.zone_for(348, 672)
        x0, y0, x1, y1 = z1.box
        # 2x2 subgrid: po jednym punkcie w każdej ćwiartce boxu Z1
        for fx in (0.25, 0.75):
            for fy in (0.25, 0.75):
                self.zmap.record_position(x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)
        self.assertTrue(self.zmap.coverage_complete("Z1"))

    def test_powtorna_pozycja_w_tej_samej_podkomorce_nie_liczy(self):
        self.assertTrue(self.zmap.record_position(350, 674))   # nowa
        self.assertFalse(self.zmap.record_position(351, 675))  # ta sama podkomórka

    def test_pozycja_poza_koperta_ignorowana(self):
        self.assertFalse(self.zmap.record_position(10, 10))


class SaturationTest(unittest.TestCase):
    def setUp(self):
        self.zmap = ZoneMap(ENVELOPE, grid=(3, 3), saturation_k=3)

    def test_nowy_fingerprint_zeruje_licznik(self):
        for _ in range(2):
            self.zmap.record_open("Z1", is_new_fingerprint=False)
        self.assertEqual(self.zmap.get_state("Z1").saturation_counter, 2)
        self.zmap.record_open("Z1", is_new_fingerprint=True)
        self.assertEqual(self.zmap.get_state("Z1").saturation_counter, 0)

    def test_saturacja_po_K_powtorkach(self):
        self.assertFalse(self.zmap.saturated("Z1"))
        for _ in range(3):
            self.zmap.record_open("Z1", is_new_fingerprint=False)
        self.assertTrue(self.zmap.saturated("Z1"))

    def test_last_new_shop_ts_znaczony(self):
        self.zmap.record_open("Z1", is_new_fingerprint=True, ts="2026-06-21T18:00:00.000+00:00")
        self.assertEqual(self.zmap.get_state("Z1").last_new_shop_ts,
                         "2026-06-21T18:00:00.000+00:00")


class DoneTest(unittest.TestCase):
    def setUp(self):
        self.zmap = ZoneMap(ENVELOPE, grid=(3, 3), subgrid=(2, 2), saturation_k=2)

    def _cover_zone(self, zone_id):
        z = self.zmap._by_id[zone_id]
        x0, y0, x1, y1 = z.box
        for fx in (0.25, 0.75):
            for fy in (0.25, 0.75):
                self.zmap.record_position(x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)

    def test_samo_pokrycie_nie_wystarcza(self):
        self._cover_zone("Z1")
        self.assertTrue(self.zmap.coverage_complete("Z1"))
        self.assertFalse(self.zmap.is_done("Z1"))  # brak nasycenia

    def test_samo_nasycenie_nie_wystarcza(self):
        for _ in range(2):
            self.zmap.record_open("Z1", is_new_fingerprint=False)
        self.assertTrue(self.zmap.saturated("Z1"))
        self.assertFalse(self.zmap.is_done("Z1"))  # brak pokrycia rogów

    def test_oba_sygnaly_to_DONE(self):
        self._cover_zone("Z1")
        for _ in range(2):
            self.zmap.record_open("Z1", is_new_fingerprint=False)
        self.assertTrue(self.zmap.is_done("Z1"))
        self.assertEqual(self.zmap.state_of("Z1"), "DONE")

    def test_done_jest_latchem(self):
        self._cover_zone("Z1")
        for _ in range(2):
            self.zmap.record_open("Z1", is_new_fingerprint=False)
        self.assertTrue(self.zmap.is_done("Z1"))
        # nowy fingerprint zeruje licznik, ale latch trzyma DONE
        self.zmap.record_open("Z1", is_new_fingerprint=True)
        self.assertTrue(self.zmap.is_done("Z1"))

    def test_stany_pending_active_done(self):
        self.assertEqual(self.zmap.state_of("Z9"), "PENDING")
        self.zmap.record_open("Z5", is_new_fingerprint=True)
        self.assertEqual(self.zmap.state_of("Z5"), "ACTIVE")


class NextZoneTest(unittest.TestCase):
    def setUp(self):
        self.zmap = ZoneMap(ENVELOPE, grid=(3, 3), subgrid=(2, 2), saturation_k=1)

    def _finish(self, zone_id):
        z = self.zmap._by_id[zone_id]
        x0, y0, x1, y1 = z.box
        for fx in (0.25, 0.75):
            for fy in (0.25, 0.75):
                self.zmap.record_position(x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)
        self.zmap.record_open(zone_id, is_new_fingerprint=False)  # K=1 => od razu nasycona
        assert self.zmap.is_done(zone_id)

    def test_najblizsza_niedokonczona_od_pozycji(self):
        # stoję w rogu Z1; najbliższa niedokończona to Z1
        nz = self.zmap.next_zone(350, 674)
        self.assertEqual(nz.zone_id, "Z1")

    def test_pomija_done_idzie_do_kolejnej(self):
        self._finish("Z1")
        nz = self.zmap.next_zone(350, 674)  # Z1 done => następna najbliższa
        self.assertNotEqual(nz.zone_id, "Z1")

    def test_wszystkie_done_to_none(self):
        for i in range(1, 10):
            self._finish(f"Z{i}")
        self.assertIsNone(self.zmap.next_zone(424, 733))
        self.assertEqual(self.zmap.remaining(), [])

    def test_progress_liczy_stany(self):
        self._finish("Z1")
        prog = self.zmap.progress()
        self.assertEqual(prog["DONE"], 1)
        self.assertEqual(prog["PENDING"], 8)
        self.assertEqual(sum(prog.values()), 9)


class PersistenceTest(unittest.TestCase):
    def test_round_trip_zachowuje_stan(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            zmap = ZoneMap(ENVELOPE, grid=(3, 3), subgrid=(2, 2), saturation_k=2, directory=d)
            zmap.record_position(350, 674)
            zmap.record_open("Z1", is_new_fingerprint=True, ts="2026-06-21T18:00:00.000+00:00")
            zmap.record_open("Z1", is_new_fingerprint=False)
            zmap.save()

            back = ZoneMap.load(d)
            self.assertEqual(back.envelope, ENVELOPE)
            self.assertEqual(back.grid, (3, 3))
            self.assertEqual(back.saturation_k, 2)
            s = back.get_state("Z1")
            self.assertEqual(s.opens, 2)
            self.assertEqual(s.saturation_counter, 1)
            self.assertEqual(s.last_new_shop_ts, "2026-06-21T18:00:00.000+00:00")
            self.assertIn((0, 0), s.covered_cells)

    def test_done_latch_przezywa_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            zmap = ZoneMap(ENVELOPE, grid=(3, 3), subgrid=(1, 1), saturation_k=1, directory=d)
            zmap.record_position(350, 674)
            zmap.record_open("Z1", is_new_fingerprint=False)
            self.assertTrue(zmap.is_done("Z1"))
            zmap.save()
            self.assertTrue(ZoneMap.load(d).is_done("Z1"))


if __name__ == "__main__":
    unittest.main()
