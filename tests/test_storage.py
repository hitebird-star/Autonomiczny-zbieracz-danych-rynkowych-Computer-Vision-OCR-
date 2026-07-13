from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scanner.models import ItemObservation, ScanStatus, ShopScan
import scanner.storage.scan_repository as scan_repository_module
from scanner.storage import CSVExporter, ScanRepository


class ScanRepositoryTests(unittest.TestCase):
    def test_manifest_events_images_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            scan = ShopScan(scan_id="scan-001", seller="Kocur")

            repository.create(scan)
            relative = repository.save_tooltip_image(
                scan.scan_id, 17, 1, Image.new("RGB", (20, 10), "black")
            )
            repository.append_event(scan.scan_id, "slot_captured", slot=17)

            restored = repository.load(scan.scan_id)
            events = [
                json.loads(line)
                for line in (
                    repository.scan_dir(scan.scan_id) / "raw_events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(restored.seller, "Kocur")
            self.assertEqual(relative, "tooltips/slot_017_1.png")
            self.assertEqual(events[-1]["slot"], 17)
            self.assertEqual([item.scan_id for item in repository.pending()], ["scan-001"])
            self.assertFalse(
                (repository.scan_dir(scan.scan_id) / ".manifest.json.tmp").exists()
            )

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(temp)
            with self.assertRaises(ValueError):
                repository.scan_dir("../outside")

    def test_save_manifest_retries_transient_windows_replace_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            scan = ShopScan(scan_id="scan-retry", seller="Retry")
            real_replace = os.replace
            attempts = {"count": 0}

            def flaky_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
                attempts["count"] += 1
                if attempts["count"] <= 2:
                    error = PermissionError("plik chwilowo zajęty")
                    error.winerror = 5
                    raise error
                real_replace(src, dst)

            with patch(
                "scanner.storage.scan_repository.os.replace", side_effect=flaky_replace
            ), patch("scanner.storage.scan_repository.time.sleep") as sleep:
                manifest = repository.save_manifest(scan)

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["scan_id"], "scan-retry")
            self.assertEqual(attempts["count"], 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertFalse((manifest.parent / ".manifest.json.tmp").exists())

    def test_save_manifest_propagates_persistent_windows_replace_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            scan = ShopScan(scan_id="scan-locked", seller="Locked")

            def locked_replace(
                src: str | os.PathLike[str], dst: str | os.PathLike[str]
            ) -> None:
                error = PermissionError("plik nadal zajęty")
                error.winerror = 5
                raise error

            with patch(
                "scanner.storage.scan_repository.os.replace", side_effect=locked_replace
            ) as replace, patch("scanner.storage.scan_repository.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    repository.save_manifest(scan)

            self.assertEqual(
                replace.call_count, scan_repository_module._REPLACE_RETRIES + 1
            )
            self.assertEqual(sleep.call_count, scan_repository_module._REPLACE_RETRIES)


class CSVExporterTests(unittest.TestCase):
    def test_exports_only_verified_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ceny.csv"
            verified = ItemObservation(
                slot=1,
                row=0,
                column=1,
                status=ScanStatus.VERIFIED,
                ai={
                    "item": "Odłamek Metina",
                    "unit_price": 7_000_000,
                    "quantity": 10,
                },
                validation={"quantity": 10, "unit_price": 7_000_000},
            )
            provisional = ItemObservation(
                slot=2,
                row=0,
                column=2,
                status=ScanStatus.PROVISIONAL,
                ai={"item": "Niepewny", "unit_price": 1, "quantity": 1},
            )
            scan = ShopScan(
                scan_id="scan-2",
                seller="Kocur",
                status=ScanStatus.VERIFIED,
                occupied_slots=2,
                captured_slots=2,
                slots={1: verified, 2: provisional},
            )
            exporter = CSVExporter(path)

            self.assertEqual(exporter.export(scan), 1)
            self.assertEqual(exporter.export(scan), 0)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["item"], "Odłamek Metina")
            self.assertEqual(rows[0]["price"], "7000000")

    def test_aggregates_identical_offers_within_shop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ceny.csv"
            slots = {}
            for slot in (1, 2, 3):
                slots[slot] = ItemObservation(
                    slot=slot,
                    row=0,
                    column=slot,
                    status=ScanStatus.VERIFIED,
                    ai={
                        "item": "Przepustka Twierdzy Razadora (2)",
                        "unit_price": 9_000_000,
                        "quantity": 25,
                    },
                    validation={"quantity": 25, "unit_price": 9_000_000},
                )
            scan = ShopScan(
                scan_id="aggregate",
                seller="DeanW",
                status=ScanStatus.VERIFIED,
                occupied_slots=3,
                captured_slots=3,
                slots=slots,
            )

            self.assertEqual(CSVExporter(path).export(scan), 1)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(row["quantity"], "75")
            self.assertEqual(row["stack_count"], "3")

    def test_upgrades_old_csv_without_inventing_historical_stack_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ceny.csv"
            path.write_text(
                "item,price,quantity,timestamp,source,seller\n"
                "Stara oferta,100,25,2026-01-01T00:00:00Z,scanner,Handlarz\n",
                encoding="utf-8-sig",
            )
            scan = ShopScan(
                scan_id="new-offer",
                seller="Kocur",
                status=ScanStatus.VERIFIED,
                occupied_slots=1,
                captured_slots=1,
                slots={
                    1: ItemObservation(
                        slot=1, row=0, column=1, status=ScanStatus.VERIFIED,
                        validation={"item": "Nowa oferta", "unit_price": 100, "quantity": 10},
                    )
                },
            )

            self.assertEqual(CSVExporter(path).export(scan), 1)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["stack_count"], "")
            self.assertEqual(rows[1]["stack_count"], "1")

    def test_rejects_extreme_unit_price(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ceny.csv"
            scan = ShopScan(
                scan_id="outlier",
                seller="",
                status=ScanStatus.VERIFIED,
                occupied_slots=1,
                captured_slots=1,
                slots={
                    1: ItemObservation(
                        slot=1,
                        row=0,
                        column=1,
                        status=ScanStatus.VERIFIED,
                        validation={
                            "item": "Pierścień Ludzi+O",
                            "unit_price": 10_000_000_000_000,
                            "quantity": 1,
                        },
                    )
                },
            )

            self.assertEqual(CSVExporter(path).export(scan), 0)
            self.assertFalse(path.exists())

    def test_empty_seller_falls_back_to_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ceny.csv"
            scan = ShopScan(
                scan_id="seller-fallback",
                seller="",
                status=ScanStatus.VERIFIED,
                occupied_slots=1,
                captured_slots=1,
                shop_fingerprint="abc123",
                slots={
                    1: ItemObservation(
                        slot=1,
                        row=0,
                        column=1,
                        status=ScanStatus.VERIFIED,
                        validation={
                            "item": "Odłamek Metina",
                            "unit_price": 7_000_000,
                            "quantity": 10,
                        },
                    )
                },
            )

            self.assertEqual(CSVExporter(path).export(scan), 1)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(row["seller"], "abc123")


if __name__ == "__main__":
    unittest.main()
