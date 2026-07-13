"""Testy indeksu ofert (offer_index.py)."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scanner.analysis.offer_index import OfferEntry, OfferIndex, _now_iso


class OfferEntryTest(unittest.TestCase):
    def test_round_trip(self):
        entry = OfferEntry(
            shop_id=42,
            item="Kamien Duszy",
            unit_price=120000,
            quantity=5,
            scanned_at="2026-06-22T14:30:00",
            scan_id="20260622_143000_Gracz1",
        )
        data = entry.to_dict()
        restored = OfferEntry.from_dict(data)
        self.assertEqual(restored.shop_id, 42)
        self.assertEqual(restored.item, "Kamien Duszy")
        self.assertEqual(restored.unit_price, 120000)
        self.assertEqual(restored.quantity, 5)

    def test_defaults(self):
        entry = OfferEntry(shop_id=1, item="Miecz", unit_price=1000)
        self.assertEqual(entry.quantity, 1)
        self.assertIsNotNone(entry.scanned_at)


class OfferIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.idx = OfferIndex(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _add_sample(self):
        self.idx.add(OfferEntry(
            shop_id=42, item="Kamien Duszy", unit_price=120000, quantity=5,
            scanned_at=_now_iso(),
        ))
        self.idx.add(OfferEntry(
            shop_id=117, item="Kamien Duszy", unit_price=110000, quantity=10,
            scanned_at=_now_iso(),
        ))
        self.idx.add(OfferEntry(
            shop_id=308, item="Kamien Duszy", unit_price=125000, quantity=3,
            scanned_at=_now_iso(),
        ))
        self.idx.add(OfferEntry(
            shop_id=42, item="Miecz+9", unit_price=3500000, quantity=1,
            scanned_at=_now_iso(),
        ))

    def test_add_and_cheapest(self):
        self._add_sample()
        cheapest = self.idx.cheapest("Kamien Duszy")
        self.assertIsNotNone(cheapest)
        self.assertEqual(cheapest.shop_id, 117)
        self.assertEqual(cheapest.unit_price, 110000)

    def test_all_for_sorted(self):
        self._add_sample()
        all_offers = self.idx.all_for("Kamien Duszy")
        self.assertEqual(len(all_offers), 3)
        prices = [e.unit_price for e in all_offers]
        self.assertEqual(prices, sorted(prices))

    def test_cheapest_missing_item(self):
        self._add_sample()
        self.assertIsNone(self.idx.cheapest("Nieistniejacy Przedmiot"))

    def test_items_list(self):
        self._add_sample()
        items = self.idx.items()
        self.assertIn("Kamien Duszy", items)
        self.assertIn("Miecz+9", items)

    def test_contains(self):
        self._add_sample()
        self.assertIn("Kamien Duszy", self.idx)
        self.assertNotIn("Nieistniejacy", self.idx)

    def test_length(self):
        self._add_sample()
        self.assertEqual(len(self.idx), 4)

    def test_save_and_load_roundtrip(self):
        self._add_sample()
        path = self.idx.save()
        self.assertTrue(path.exists())

        idx2 = OfferIndex(self.tmp.name).load()
        self.assertEqual(len(idx2), 4)
        cheapest = idx2.cheapest("Kamien Duszy")
        self.assertIsNotNone(cheapest)
        self.assertEqual(cheapest.shop_id, 117)

    def test_load_empty_directory(self):
        idx = OfferIndex("/nonexistent/path/12345").load()
        self.assertEqual(len(idx), 0)

    def test_ttl_expired(self):
        # Dodaj stara oferte (25h temu)
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        self.idx.add(OfferEntry(
            shop_id=99, item="Stary Item", unit_price=100,
            scanned_at=old_ts,
        ))
        # Dodaj swieza
        self.idx.add(OfferEntry(
            shop_id=100, item="Stary Item", unit_price=200,
            scanned_at=_now_iso(),
        ))
        cheapest = self.idx.cheapest("Stary Item", ttl_hours=24)
        self.assertIsNotNone(cheapest)
        self.assertEqual(cheapest.shop_id, 100)  # tylko swieza

    def test_ttl_all_expired(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        self.idx.add(OfferEntry(
            shop_id=99, item="Stary Item", unit_price=100,
            scanned_at=old_ts,
        ))
        self.assertIsNone(self.idx.cheapest("Stary Item", ttl_hours=24))


if __name__ == "__main__":
    unittest.main()