"""Testy rejestru sklepów (Etap 1) — dedup po fingerprincie, lokalizacja
nullable, by_id/nearest/in_zone, TTL, partycja, round-trip trwałości. Bez gry."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scanner.analysis.shop_registry import (
    DEFAULT_TTL_HOURS,
    ShopRecord,
    ShopRegistry,
    partition_key,
)


def manifest(
    *,
    scan_id: str,
    fingerprint: str | None,
    seller: str = "",
    game_position=None,
    map_name=None,
    channel=None,
    created_at: str = "2026-06-21T18:00:00.000+00:00",
    updated_at: str | None = None,
) -> dict:
    """Minimalny manifest w realnym kształcie (pola z models/shop_scan.py)."""

    return {
        "scan_id": scan_id,
        "shop_fingerprint": fingerprint,
        "seller": seller,
        "game_position": game_position,
        "map_name": map_name,
        "channel": channel,
        "created_at": created_at,
        "updated_at": updated_at or created_at,
    }


class PartitionKeyTest(unittest.TestCase):
    def test_brak_mapy_i_kanalu(self):
        self.assertEqual(partition_key(None, None), "unknown_CHx")

    def test_mapa_i_kanal(self):
        self.assertEqual(partition_key("Glevia2 Farm", 1), "Glevia2_Farm_CH1")

    def test_kanal_zero_nie_jest_none(self):
        self.assertEqual(partition_key("X", 0), "X_CH0")


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.reg = ShopRegistry(Path("memory"))  # katalog nieistotny bez save()

    def test_pierwszy_sklep_dostaje_shop_id_1(self):
        rec = self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa", seller="MANDAT"))
        self.assertIsNotNone(rec)
        self.assertEqual(rec.shop_id, 1)
        self.assertEqual(rec.fingerprint, "aaa")
        self.assertEqual(rec.seller, "MANDAT")
        self.assertEqual(rec.scan_ids, ["s1"])

    def test_shop_id_sekwencyjny_rosnacy(self):
        a = self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa"))
        b = self.reg.ingest(manifest(scan_id="s2", fingerprint="bbb"))
        self.assertEqual((a.shop_id, b.shop_id), (1, 2))
        self.assertEqual(len(self.reg), 2)

    def test_manifest_bez_fingerprintu_pomijany(self):
        rec = self.reg.ingest(manifest(scan_id="s1", fingerprint=None, seller="X"))
        self.assertIsNone(rec)
        self.assertEqual(len(self.reg), 0)

    def test_dedup_po_fingerprincie_nie_tworzy_drugiego(self):
        a = self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa"))
        b = self.reg.ingest(manifest(scan_id="s2", fingerprint="aaa"))
        self.assertEqual(a.shop_id, b.shop_id)
        self.assertEqual(len(self.reg), 1)
        self.assertEqual(sorted(b.scan_ids), ["s1", "s2"])

    def test_dedup_ten_sam_scan_nie_dubluje(self):
        self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa"))
        rec = self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa"))
        self.assertEqual(rec.scan_ids, ["s1"])

    def test_lokalizacja_nullable_potem_dopelniona(self):
        rec = self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa"))
        self.assertIsNone(rec.location)  # brak game_position => None, bez błędu
        rec2 = self.reg.ingest(
            manifest(scan_id="s2", fingerprint="aaa", game_position=[460, 720],
                     map_name="Glevia2 Farm", channel=1)
        )
        self.assertEqual(rec2.location, (460, 720))
        self.assertEqual(rec2.map_name, "Glevia2 Farm")
        self.assertEqual(rec2.channel, 1)

    def test_pusty_seller_nie_nadpisuje_znanego(self):
        self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa", seller="MANDAT"))
        rec = self.reg.ingest(manifest(scan_id="s2", fingerprint="aaa", seller=""))
        self.assertEqual(rec.seller, "MANDAT")

    def test_first_last_seen_rosnie_we_wlasciwa_strone(self):
        self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa",
                                 created_at="2026-06-21T10:00:00.000+00:00"))
        rec = self.reg.ingest(manifest(scan_id="s2", fingerprint="aaa",
                                       created_at="2026-06-21T09:00:00.000+00:00",
                                       updated_at="2026-06-21T12:00:00.000+00:00"))
        self.assertEqual(rec.first_seen, "2026-06-21T09:00:00.000+00:00")  # wcześniejszy
        self.assertEqual(rec.last_seen, "2026-06-21T12:00:00.000+00:00")   # późniejszy

    def test_nowa_pozycja_nie_kasuje_starej_gdy_brak(self):
        self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa", game_position=[460, 720]))
        rec = self.reg.ingest(manifest(scan_id="s2", fingerprint="aaa", game_position=None))
        self.assertEqual(rec.location, (460, 720))  # None nie nadpisuje znanej

    def test_ingest_all_liczy_tylko_z_pk(self):
        n = self.reg.ingest_all([
            manifest(scan_id="s1", fingerprint="aaa"),
            manifest(scan_id="s2", fingerprint=None),
            manifest(scan_id="s3", fingerprint="bbb"),
        ])
        self.assertEqual(n, 2)


class QueryTest(unittest.TestCase):
    def setUp(self):
        self.reg = ShopRegistry(Path("memory"))
        self.reg.ingest(manifest(scan_id="s1", fingerprint="aaa", game_position=[400, 700]))
        self.reg.ingest(manifest(scan_id="s2", fingerprint="bbb", game_position=[460, 720]))
        self.reg.ingest(manifest(scan_id="s3", fingerprint="ccc", game_position=[500, 790]))
        self.reg.ingest(manifest(scan_id="s4", fingerprint="ddd"))  # bez lokalizacji

    def test_by_id(self):
        self.assertEqual(self.reg.by_id(2).fingerprint, "bbb")
        self.assertIsNone(self.reg.by_id(999))

    def test_by_fingerprint(self):
        self.assertEqual(self.reg.by_fingerprint("ccc").shop_id, 3)

    def test_nearest_zwraca_najblizszy_zlokalizowany(self):
        near = self.reg.nearest(455, 718)
        self.assertEqual(near.fingerprint, "bbb")

    def test_nearest_pomija_bez_lokalizacji(self):
        near = self.reg.nearest(405, 702)
        self.assertEqual(near.fingerprint, "aaa")  # ddd bez (x,y) ignorowany

    def test_nearest_remis_nizszy_shop_id(self):
        reg = ShopRegistry(Path("memory"))
        reg.ingest(manifest(scan_id="a", fingerprint="f1", game_position=[100, 100]))
        reg.ingest(manifest(scan_id="b", fingerprint="f2", game_position=[120, 100]))
        near = reg.nearest(110, 100)  # równa odległość 10 do obu
        self.assertEqual(near.shop_id, 1)

    def test_nearest_pusty_rejestr(self):
        self.assertIsNone(ShopRegistry(Path("memory")).nearest(0, 0))

    def test_in_zone_inclusive_i_sortowanie(self):
        # box wyklucza aaa krawędzią: x0=401 > aaa.x=400, więc zostają bbb, ccc
        zone = self.reg.in_zone((401, 672, 501, 794))
        self.assertEqual([r.fingerprint for r in zone], ["bbb", "ccc"])
        # szerszy box łapie wszystkie trzy zlokalizowane, posortowane po shop_id
        zone_full = self.reg.in_zone((350, 690, 510, 800))
        self.assertEqual([r.shop_id for r in zone_full], [1, 2, 3])

    def test_in_zone_granica_inclusive(self):
        # dokładnie na rogu aaa(400,700) => wpada (inclusive z obu stron)
        self.assertEqual(
            [r.fingerprint for r in self.reg.in_zone((400, 700, 400, 700))], ["aaa"]
        )

    def test_in_zone_pomija_bez_lokalizacji(self):
        zone = self.reg.in_zone((0, 0, 9999, 9999))
        self.assertNotIn("ddd", [r.fingerprint for r in zone])

    def test_in_zone_normalizuje_odwrocony_box(self):
        a = self.reg.in_zone((501, 794, 399, 672))
        b = self.reg.in_zone((399, 672, 501, 794))
        self.assertEqual([r.shop_id for r in a], [r.shop_id for r in b])


class FreshnessTest(unittest.TestCase):
    def setUp(self):
        self.reg = ShopRegistry(Path("memory"))
        self.now = datetime(2026, 6, 21, 18, 0, tzinfo=timezone.utc)

    def _seen(self, fp, hours_ago):
        ts = (self.now - timedelta(hours=hours_ago)).isoformat(timespec="milliseconds")
        self.reg.ingest(manifest(scan_id=fp, fingerprint=fp, created_at=ts, updated_at=ts))

    def test_swiezy_w_oknie_ttl(self):
        self._seen("aaa", hours_ago=2)
        self.assertTrue(self.reg.is_known_fresh("aaa", now=self.now))

    def test_przeterminowany_poza_ttl(self):
        self._seen("bbb", hours_ago=DEFAULT_TTL_HOURS + 1)
        self.assertFalse(self.reg.is_known_fresh("bbb", now=self.now))

    def test_nieznany_fingerprint_nie_jest_swiezy(self):
        self.assertFalse(self.reg.is_known_fresh("zzz", now=self.now))

    def test_fresh_zwraca_tylko_swieze(self):
        self._seen("aaa", hours_ago=1)
        self._seen("bbb", hours_ago=DEFAULT_TTL_HOURS + 5)
        fresh = self.reg.fresh(now=self.now)
        self.assertEqual([r.fingerprint for r in fresh], ["aaa"])


class PersistenceTest(unittest.TestCase):
    def test_round_trip_zachowuje_dane_i_next_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = ShopRegistry.open(root, partition="glevia_market")
            reg.ingest(manifest(scan_id="s1", fingerprint="aaa", seller="MANDAT",
                                game_position=[460, 720], map_name="Glevia2 Farm", channel=1))
            reg.ingest(manifest(scan_id="s2", fingerprint="bbb"))
            reg.save()

            reloaded = ShopRegistry.open(root, partition="glevia_market")
            self.assertEqual(len(reloaded), 2)
            a = reloaded.by_fingerprint("aaa")
            self.assertEqual(a.shop_id, 1)
            self.assertEqual(a.seller, "MANDAT")
            self.assertEqual(a.location, (460, 720))
            self.assertEqual(a.map_name, "Glevia2 Farm")
            self.assertEqual(a.channel, 1)
            # nowy sklep dostaje id 3 (next_id = max+1, nigdy nie reużywa)
            c = reloaded.ingest(manifest(scan_id="s3", fingerprint="ccc"))
            self.assertEqual(c.shop_id, 3)

    def test_partycja_to_osobny_katalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ch1 = ShopRegistry.open(root, map_name="Glevia2 Farm", channel=1)
            ch2 = ShopRegistry.open(root, map_name="Glevia2 Farm", channel=2)
            ch1.ingest(manifest(scan_id="s1", fingerprint="aaa"))
            ch2.ingest(manifest(scan_id="s2", fingerprint="aaa"))  # ten sam fp, inny kanał
            ch1.save()
            ch2.save()
            self.assertNotEqual(ch1.path, ch2.path)
            self.assertTrue(ch1.path.exists())
            self.assertTrue(ch2.path.exists())
            # izolacja: każdy worek widzi swój jeden sklep
            self.assertEqual(len(ShopRegistry.open(root, map_name="Glevia2 Farm", channel=1)), 1)

    def test_open_brak_katalogu_to_pusty_rejestr(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ShopRegistry.open(Path(tmp) / "nie_ma", partition="x")
            self.assertEqual(len(reg), 0)

    def test_save_jest_atomowy_bez_tmp_po_zapisie(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ShopRegistry.open(Path(tmp), partition="p")
            reg.ingest(manifest(scan_id="s1", fingerprint="aaa"))
            reg.save()
            leftovers = list(reg.directory.glob("*.tmp"))
            self.assertEqual(leftovers, [])


class RecordTest(unittest.TestCase):
    def test_to_from_dict_round_trip(self):
        rec = ShopRecord(shop_id=7, fingerprint="aaa", seller="X", x=460, y=720,
                         map_name="M", channel=2, scan_ids=["s1", "s2"])
        back = ShopRecord.from_dict(rec.to_dict())
        self.assertEqual(back, rec)

    def test_location_none_gdy_brak_wspolrzednej(self):
        self.assertIsNone(ShopRecord(shop_id=1, fingerprint="a", x=460, y=None).location)

    def test_legacy_record_without_pos_samples_seeds_from_xy(self):
        # stary shops.jsonl (brak klucza pos_samples) -> zalążkuj, pozycja zachowana
        back = ShopRecord.from_dict({"shop_id": 1, "fingerprint": "a", "x": 460, "y": 720})
        self.assertEqual(back.location, (460, 720))
        self.assertEqual(len(back.pos_samples), 1)
        self.assertEqual(back.pos_samples[0]["source"], "legacy")


class IdealLocationTest(unittest.TestCase):
    """C5 (COVERAGE_MAP_CORE §5b): idealna lokacja = mediana próbek OCR, nie last-write."""

    def test_ideal_is_median_of_ocr_not_last_write(self):
        rec = ShopRecord(shop_id=1, fingerprint="a")
        for x, y in [(460, 720), (462, 722), (461, 719)]:
            rec.add_pos_sample(x, y, source="ocr")
        self.assertEqual(rec.location, (461, 720))      # mediana per-oś, nie ostatnia (461,719)
        self.assertEqual(rec.pos_conf["n_ocr"], 3)

    def test_dead_reckoning_ignored_when_ocr_present(self):
        rec = ShopRecord(shop_id=1, fingerprint="a")
        rec.add_pos_sample(460, 720, source="ocr")
        rec.add_pos_sample(999, 999, source="dead_reckoning")   # śmieciowy DR
        self.assertEqual(rec.location, (460, 720))      # OCR wygrywa, DR pominięty
        self.assertEqual(rec.pos_conf["n_ocr"], 1)

    def test_dead_reckoning_used_when_no_ocr_low_conf(self):
        rec = ShopRecord(shop_id=1, fingerprint="a")
        rec.add_pos_sample(450, 700, source="dead_reckoning")
        rec.add_pos_sample(454, 704, source="dead_reckoning")
        self.assertEqual(rec.location, (452, 702))      # mediana DR (lepsze niż nic)
        self.assertEqual(rec.pos_conf["n_ocr"], 0)      # ale conf mówi: brak OCR

    def test_ingest_tags_source_and_aggregates(self):
        reg = ShopRegistry(Path("memory"))
        reg.ingest(manifest(scan_id="s1", fingerprint="aaa", game_position=[460, 720]))
        reg.ingest(manifest(scan_id="s2", fingerprint="aaa", game_position=[462, 722]))
        rec = reg.by_fingerprint("aaa")
        self.assertEqual(len(rec.pos_samples), 2)       # akumulacja, nie nadpisanie
        self.assertEqual(rec.location, (461, 721))      # mediana

    def test_registry_add_pos_sample_and_reaggregate(self):
        reg = ShopRegistry(Path("memory"))
        reg.ingest(manifest(scan_id="s1", fingerprint="aaa", game_position=[460, 720]))
        reg.add_pos_sample("aaa", 500, 760, source="ocr")
        self.assertEqual(reg.add_pos_sample("nieznany", 1, 1), None)
        reg.reaggregate()
        self.assertEqual(reg.by_fingerprint("aaa").pos_conf["n_ocr"], 1)  # tylko 1 OCR (drugi unknown)

    def test_samples_capped(self):
        rec = ShopRecord(shop_id=1, fingerprint="a")
        for i in range(40):
            rec.add_pos_sample(460 + i % 3, 720, source="ocr")
        self.assertLessEqual(len(rec.pos_samples), 25)  # MAX_POS_SAMPLES

    def test_pos_samples_round_trip(self):
        rec = ShopRecord(shop_id=1, fingerprint="a")
        rec.add_pos_sample(460, 720, source="ocr")
        back = ShopRecord.from_dict(rec.to_dict())
        self.assertEqual(back, rec)                     # pos_samples + pos_conf przetrwały


if __name__ == "__main__":
    unittest.main()
