"""Testy modelu Atlasu: ingest rejestru, rzut metryczny per sklep, dedup, persist."""

from __future__ import annotations

import os
import tempfile
import unittest

from scanner.atlas.calibration import GroundProjection
from scanner.atlas.atlas import AtlasShop, MarketAtlas
from scanner.atlas.config import AtlasConfig


class MarketAtlasTests(unittest.TestCase):
    A_TRUE = [[0.030, -0.030], [0.016, 0.016]]

    def _registry(self):
        # dwa sklepy dzielące JEDEN coord postaci (problem, który Atlas ma rozwiązać)
        return [
            {"shop_id": 1, "fingerprint": "aaa", "seller": "Alice", "x": 486, "y": 740, "scan_ids": ["s1"]},
            {"shop_id": 2, "fingerprint": "bbb", "seller": "Bob", "x": 486, "y": 740, "scan_ids": ["s2"]},
        ]

    def test_upsert_from_registry(self):
        atlas = MarketAtlas()
        added = atlas.upsert_from_registry(self._registry())
        self.assertEqual(added, 2)
        self.assertEqual(len(atlas.shops()), 2)
        # game_position współdzielone (zgrubne) — atlas_position jeszcze puste
        self.assertEqual(atlas.located_shops(), [])
        self.assertIsNone(atlas.shops()[0].atlas_position)

    def test_upsert_merges_same_fingerprint(self):
        atlas = MarketAtlas()
        atlas.upsert_from_registry([{"fingerprint": "aaa", "x": 486, "y": 740, "scan_ids": ["s1"]}])
        atlas.upsert_from_registry(
            [{"fingerprint": "aaa", "seller": "Alice", "x": 487, "y": 741, "scan_ids": ["s9"]}]
        )
        self.assertEqual(len(atlas.shops()), 1)
        shop = atlas.shops()[0]
        self.assertEqual(shop.seller, "Alice")
        self.assertIn("s9", shop.scan_ids)
        self.assertEqual(shop.game_position, (487.0, 741.0))

    def test_place_shop_gives_distinct_metric_positions(self):
        # DWA sklepy z tego samego postoju, ale RÓŻNE piksele → różne atlas_position
        atlas = MarketAtlas()
        atlas.upsert_from_registry(self._registry())
        proj = GroundProjection(self.A_TRUE, version="v1")
        player = (486.0, 740.0)
        a = atlas.place_shop("aaa", player, (900.0, 500.0), proj)
        b = atlas.place_shop("bbb", player, (1050.0, 560.0), proj)
        self.assertIsNotNone(a.atlas_position)
        self.assertIsNotNone(b.atlas_position)
        self.assertNotEqual(a.atlas_position, b.atlas_position)  # rozdzielone!
        self.assertEqual(a.transform_version, "v1")
        self.assertEqual(len(atlas.located_shops()), 2)

    def test_place_shop_ema_stabilizes(self):
        atlas = MarketAtlas(position_ema=0.5)
        proj = GroundProjection(self.A_TRUE)
        atlas.upsert_from_registry([{"fingerprint": "aaa", "x": 486, "y": 740}])
        p1 = atlas.place_shop("aaa", (486.0, 740.0), (900.0, 500.0), proj).atlas_position
        p2 = atlas.place_shop("aaa", (486.0, 740.0), (902.0, 502.0), proj).atlas_position
        self.assertNotEqual(p1, p2)  # druga próbka przesunęła pozycję (EMA)
        self.assertEqual(len(atlas.shops()), 1)

    def test_nearest(self):
        atlas = MarketAtlas()
        proj = GroundProjection(self.A_TRUE)
        atlas.upsert_from_registry(self._registry())
        atlas.place_shop("aaa", (486.0, 740.0), (900.0, 500.0), proj)
        atlas.place_shop("bbb", (486.0, 740.0), (1050.0, 560.0), proj)
        target = atlas._by_fp["aaa"].atlas_position
        nearest = atlas.nearest(target, k=1)
        self.assertEqual(nearest[0].fingerprint, "aaa")

    def test_persist_round_trip(self):
        atlas = MarketAtlas(boundary=[(1.0, 2.0), (3.0, 4.0)])
        proj = GroundProjection(self.A_TRUE)
        atlas.upsert_from_registry(self._registry())
        atlas.place_shop("aaa", (486.0, 740.0), (900.0, 500.0), proj)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "atlas.json")
            atlas.save(p)
            restored = MarketAtlas.load(p)
        self.assertIsNotNone(restored)
        self.assertEqual(len(restored.shops()), 2)
        self.assertEqual(restored.boundary, [(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(len(restored.located_shops()), 1)

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(MarketAtlas.load(os.path.join(d, "brak.json")))

    def test_shop_dict_round_trip(self):
        shop = AtlasShop(
            shop_id="1", fingerprint="aaa", seller="Alice", scan_ids=["s1"],
            game_position=(486.0, 740.0), atlas_position=(455.5, 725.2),
            screen_position=(900.0, 500.0), position_confidence=0.8, transform_version="v1",
        )
        back = AtlasShop.from_dict(shop.to_dict())
        self.assertEqual(back.to_dict(), shop.to_dict())


class AtlasConfigTests(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        cfg = AtlasConfig()
        back = AtlasConfig.from_dict(cfg.to_dict())
        self.assertEqual(back.anchor, cfg.anchor)
        self.assertEqual(back.calib_keys, cfg.calib_keys)
        self.assertEqual(back.server_port, cfg.server_port)

    def test_load_missing_gives_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = AtlasConfig.load(os.path.join(d, "brak.json"))
            self.assertEqual(cfg.anchor, (960.0, 540.0))

    def test_save_load_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "atlas_config.json")
            cfg = AtlasConfig(server_port=9999, dedup_radius_units=2.5)
            cfg.save(p)
            back = AtlasConfig.load(p)
            self.assertEqual(back.server_port, 9999)
            self.assertEqual(back.dedup_radius_units, 2.5)

    def test_ignores_unknown_keys(self):
        cfg = AtlasConfig.from_dict({"server_port": 8000, "nieznane_pole": 123})
        self.assertEqual(cfg.server_port, 8000)


if __name__ == "__main__":
    unittest.main()
