"""Testy mapowania w tle (Etap 2) — backfill, wykrywanie zmian, idempotencja,
odporność na obcięty manifest, trwałość. Bez gry (czyste pliki + temp dir)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scanner.analysis.map_watcher import MapWatcher, SyncResult
from scanner.analysis.shop_registry import ShopRegistry


def write_manifest(scans_dir: Path, scan_id: str, *, fingerprint=None, seller="",
                   game_position=None, mtime: float | None = None) -> Path:
    """Zapisz scans/<scan_id>/manifest.json w realnym kształcie."""

    d = scans_dir / scan_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "manifest.json"
    path.write_text(json.dumps({
        "scan_id": scan_id,
        "shop_fingerprint": fingerprint,
        "seller": seller,
        "game_position": game_position,
        "map_name": None,
        "channel": None,
        "created_at": "2026-06-21T18:00:00.000+00:00",
        "updated_at": "2026-06-21T18:00:00.000+00:00",
    }, ensure_ascii=False), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class WatcherTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.scans = self.root / "scans"
        self.scans.mkdir()
        self.registry = ShopRegistry.open(self.root / "market_map", partition="glevia_market")
        self.watcher = MapWatcher(self.registry, self.scans)

    def tearDown(self):
        self._tmp.cleanup()


class BackfillTest(WatcherTestBase):
    def test_pusty_katalog_nic_nie_robi(self):
        result = self.watcher.sync_once()
        self.assertEqual((result.scanned, result.changed, result.new_shops), (0, 0, 0))

    def test_brak_katalogu_scans_nie_wywala(self):
        w = MapWatcher(self.registry, self.root / "nie_ma")
        self.assertEqual(w.sync_once().scanned, 0)

    def test_backfill_ingestuje_istniejace(self):
        write_manifest(self.scans, "s1", fingerprint="aaa", seller="MANDAT")
        write_manifest(self.scans, "s2", fingerprint="bbb")
        result = self.watcher.sync_once()
        self.assertEqual(result.scanned, 2)
        self.assertEqual(result.changed, 2)
        self.assertEqual(result.ingested, 2)
        self.assertEqual(result.new_shops, 2)
        self.assertEqual(len(self.registry), 2)

    def test_manifest_bez_fingerprintu_obejrzany_nie_zliczony(self):
        write_manifest(self.scans, "s1", fingerprint=None)
        result = self.watcher.sync_once()
        self.assertEqual(result.changed, 1)   # plik zauważony
        self.assertEqual(result.ingested, 0)  # ale bez PK nie wszedł do rejestru
        self.assertEqual(len(self.registry), 0)


class ChangeDetectionTest(WatcherTestBase):
    def test_drugie_przejscie_bez_zmian_nic_nie_robi(self):
        write_manifest(self.scans, "s1", fingerprint="aaa", mtime=1000.0)
        self.watcher.sync_once()
        result = self.watcher.sync_once()
        self.assertEqual(result.scanned, 1)
        self.assertEqual(result.changed, 0)  # mtime ten sam => pominięty
        self.assertEqual(result.new_shops, 0)

    def test_zmieniony_manifest_jest_reingestowany(self):
        # manifest dojrzewa: najpierw bez fingerprintu, potem z nim (mtime rośnie)
        write_manifest(self.scans, "s1", fingerprint=None, mtime=1000.0)
        r1 = self.watcher.sync_once()
        self.assertEqual(r1.ingested, 0)
        write_manifest(self.scans, "s1", fingerprint="aaa", seller="MANDAT", mtime=2000.0)
        r2 = self.watcher.sync_once()
        self.assertEqual(r2.changed, 1)
        self.assertEqual(r2.ingested, 1)
        self.assertEqual(len(self.registry), 1)
        self.assertEqual(self.registry.by_fingerprint("aaa").seller, "MANDAT")

    def test_nowy_sklep_w_kolejnym_przejsciu(self):
        write_manifest(self.scans, "s1", fingerprint="aaa", mtime=1000.0)
        self.watcher.sync_once()
        write_manifest(self.scans, "s2", fingerprint="bbb", mtime=1000.0)
        result = self.watcher.sync_once()
        self.assertEqual(result.new_shops, 1)
        self.assertEqual(len(self.registry), 2)

    def test_obciety_manifest_pomijany_potem_dochodzi(self):
        d = self.scans / "s1"
        d.mkdir()
        (d / "manifest.json").write_text("{ obciety", encoding="utf-8")  # nie-JSON
        result = self.watcher.sync_once()
        self.assertEqual(result.changed, 0)   # nie policzony jako zmiana (skip)
        self.assertEqual(len(self.registry), 0)
        # teraz plik domknięty -> kolejne przejście go łyka
        write_manifest(self.scans, "s1", fingerprint="aaa")
        self.assertEqual(self.watcher.sync_once().ingested, 1)


class PersistenceTest(WatcherTestBase):
    def test_sync_zapisuje_rejestr_na_dysk(self):
        write_manifest(self.scans, "s1", fingerprint="aaa")
        self.watcher.sync_once()
        self.assertTrue(self.registry.path.exists())
        reloaded = ShopRegistry.open(self.root / "market_map", partition="glevia_market")
        self.assertEqual(len(reloaded), 1)

    def test_bez_zmian_nie_zapisuje(self):
        # pusty katalog => changed=0 => brak zapisu (plik nie powstaje)
        self.watcher.sync_once()
        self.assertFalse(self.registry.path.exists())


class WatchLoopTest(WatcherTestBase):
    def test_watch_max_passes_i_callback(self):
        write_manifest(self.scans, "s1", fingerprint="aaa")
        seen: list[SyncResult] = []
        total = self.watcher.watch(interval=0, max_passes=2, on_pass=seen.append)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0].new_shops, 1)   # 1. przejście ingestuje
        self.assertEqual(seen[1].changed, 0)     # 2. przejście nic nowego
        self.assertEqual(total, 1)


if __name__ == "__main__":
    unittest.main()
