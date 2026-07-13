"""Testy warstwy web Atlasu: payload render (czysty), build z plików, ustawienia, smoke HTTP."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.request

from scanner.atlas.atlas import MarketAtlas
from scanner.atlas.calibration import GroundProjection
from scanner.atlas.config import AtlasConfig
from scanner.atlas.render import build_render_payload
from scanner.atlas.server import AtlasState, build_atlas_from_sources, make_server

A_TRUE = [[0.030, -0.030], [0.016, 0.016]]


class RenderPayloadTests(unittest.TestCase):
    def test_payload_marks_source_and_stats(self):
        atlas = MarketAtlas(boundary=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        atlas.upsert_from_registry([
            {"fingerprint": "aaa", "seller": "Alice", "x": 486, "y": 740},
            {"fingerprint": "bbb", "seller": "Bob", "x": 486, "y": 740},
        ])
        # jeden sklep dostaje metryczną pozycję (atlas), drugi zostaje na coord postoju (game)
        atlas.place_shop("aaa", (486.0, 740.0), (900.0, 500.0), GroundProjection(A_TRUE))
        p = build_render_payload(atlas)
        self.assertEqual(p["stats"]["total"], 2)
        self.assertEqual(p["stats"]["located"], 1)
        self.assertEqual(p["stats"]["on_stand"], 1)
        sources = {s["fingerprint"]: s["source"] for s in p["shops"]}
        self.assertEqual(sources["aaa"], "atlas")
        self.assertEqual(sources["bbb"], "game")
        self.assertIsNotNone(p["bounds"])
        self.assertEqual(len(p["boundary"]), 3)

    def test_payload_empty_atlas(self):
        p = build_render_payload(MarketAtlas())
        self.assertEqual(p["shops"], [])
        self.assertIsNone(p["bounds"])


class BuildFromSourcesTests(unittest.TestCase):
    def _cfg(self, d):
        reg = os.path.join(d, "shops.jsonl")
        with open(reg, "w", encoding="utf-8") as f:
            f.write(json.dumps({"shop_id": 1, "fingerprint": "aaa", "seller": "Al", "x": 486, "y": 740}) + "\n")
            f.write(json.dumps({"shop_id": 2, "fingerprint": "bbb", "seller": "Bo", "x": 486, "y": 740}) + "\n")
        bnd = os.path.join(d, "farm_map.json")
        with open(bnd, "w", encoding="utf-8") as f:
            json.dump({"polygon": [[380, 669], [505, 715], [456, 761]], "source": "walk"}, f)
        return AtlasConfig(
            shops_registry=reg, boundary_file=bnd,
            atlas_store=os.path.join(d, "atlas.json"),
            calibration_file=os.path.join(d, "cal.json"),
        )

    def test_build_reads_registry_and_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            atlas = build_atlas_from_sources(self._cfg(d))
            self.assertEqual(len(atlas.shops()), 2)
            self.assertEqual(len(atlas.boundary), 3)


class ConfigUpdateTests(unittest.TestCase):
    def test_update_config_persists_to_given_path(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "atlas_config.json")
            state = AtlasState(AtlasConfig(server_port=0), config_path=cfg_path)
            new = state.update_config({"server_port": 8123, "dedup_radius_units": 2.0})
            self.assertEqual(new.server_port, 8123)
            self.assertTrue(os.path.exists(cfg_path))
            self.assertEqual(AtlasConfig.load(cfg_path).dedup_radius_units, 2.0)


class ServerSmokeTests(unittest.TestCase):
    def test_http_endpoints(self):
        with tempfile.TemporaryDirectory() as d:
            reg = os.path.join(d, "shops.jsonl")
            with open(reg, "w", encoding="utf-8") as f:
                f.write(json.dumps({"fingerprint": "aaa", "seller": "Al", "x": 486, "y": 740}) + "\n")
            cfg = AtlasConfig(
                server_host="127.0.0.1", server_port=0, shops_registry=reg,
                atlas_store=os.path.join(d, "atlas.json"),
                calibration_file=os.path.join(d, "cal.json"),
                boundary_file=os.path.join(d, "brak.json"),
            )
            httpd, state = make_server(cfg)
            port = httpd.server_address[1]
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            try:
                base = f"http://127.0.0.1:{port}"
                atlas = json.loads(urllib.request.urlopen(base + "/api/atlas", timeout=5).read())
                self.assertEqual(atlas["stats"]["total"], 1)
                self.assertIn("has_calibration", atlas)
                conf = json.loads(urllib.request.urlopen(base + "/api/config", timeout=5).read())
                self.assertIn("server_port", conf)
                index = urllib.request.urlopen(base + "/", timeout=5)
                self.assertEqual(index.status, 200)
                self.assertIn(b"Atlas", index.read())
            finally:
                httpd.shutdown()
                httpd.server_close()
                t.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
