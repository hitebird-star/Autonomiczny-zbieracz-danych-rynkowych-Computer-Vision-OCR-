
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scanner.analysis.shop_registry import ShopRegistry
from scanner.analysis.zone_map import ZoneMap
from scanner.navigation.map_navigator import MapSynchronizedNavigator

FARM = (348, 672, 501, 794)


class MapNavigatorTests(unittest.TestCase):

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.market = Path(self.tmp.name)
        self.registry = ShopRegistry(directory=self.market)
        self.zmap = ZoneMap(FARM, directory=self.market)
        self.nav = MapSynchronizedNavigator(self.zmap, self.registry)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fallback_when_no_position(self):
        step = self.nav.next_step()
        self.assertIsNotNone(step)
        self.assertIn(step.key, ("a", "d", "s"))
        self.assertFalse(self.nav.is_finished())

    def test_stamps_position_and_enters_zone(self):
        self.nav.stamp_position((420, 730))
        self.assertIsNotNone(self.nav.current_zone_id)
        step = self.nav.next_step()
        self.assertIsNotNone(step)
        # Should be a zone step, not fallback
        self.assertIn(step.kind, ("horizontal", "lane_change"))

    def test_bounce_on_right_edge(self):
        # Place near right edge of the zone
        self.nav.stamp_position((490, 730))
        step = self.nav.next_step()
        # Near right edge + direction=right -> should bounce (lane_change)
        self.assertEqual(step.kind, "lane_change")

    def test_bounce_on_left_edge(self):
        self.nav.stamp_position((355, 730))
        self.nav._direction = "left"
        step = self.nav.next_step()
        self.assertEqual(step.kind, "lane_change")

    def test_zone_done_triggers_transition(self):
        self.nav.stamp_position((420, 730))
        zid = self.nav.current_zone_id
        z = self.zmap._by_id[zid]
        x0, y0, x1, y1 = z.box
        # Mark all 9 subcells as covered
        sc, sr = self.zmap.subgrid
        sw = (x1 - x0) / sc
        sh = (y1 - y0) / sr
        for ci in range(sc):
            for ri in range(sr):
                cx = x0 + (ci + 0.5) * sw
                cy = y0 + (ri + 0.5) * sh
                self.zmap.record_position(cx, cy)
        self.assertTrue(self.zmap.coverage_complete(zid))
        # Mark saturation: K consecutive non-new opens
        for _ in range(self.zmap.saturation_k + 1):
            self.zmap.record_open(zid, is_new_fingerprint=False)
        self.assertTrue(self.zmap.saturated(zid))
        self.assertTrue(self.zmap.is_done(zid))
        step = self.nav.next_step()
        self.assertIsNotNone(step)
        # Should be a transition step (navigate_x or navigate_y)
        self.assertIn(step.kind, ("navigate_x", "navigate_y"))

    def test_all_done_finishes(self):
        # Mark all 9 zones as DONE with full coverage + saturation
        sc, sr = self.zmap.subgrid
        for zone in self.zmap.zones:
            x0, y0, x1, y1 = zone.box
            sw = (x1 - x0) / sc
            sh = (y1 - y0) / sr
            for ci in range(sc):
                for ri in range(sr):
                    self.zmap.record_position(x0 + (ci + 0.5) * sw, y0 + (ri + 0.5) * sh)
            for _ in range(self.zmap.saturation_k + 1):
                self.zmap.record_open(zone.zone_id, is_new_fingerprint=False)
        self.nav.stamp_position((420, 730))
        step = self.nav.next_step()
        self.assertIsNone(step)
        self.assertTrue(self.nav.is_finished())


if __name__ == "__main__":
    unittest.main()
