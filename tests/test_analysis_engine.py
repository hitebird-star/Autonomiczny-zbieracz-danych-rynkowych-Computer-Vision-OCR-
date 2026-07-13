"""Test silnika analizy z podmienionym readerem VLM (bez Ollamy)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scanner.analysis import ollama_reader
from scanner.analysis.engine import VlmAnalysisEngine
from scanner.models import ItemObservation, ScanStatus, ShopScan
from scanner.storage import CSVExporter, ScanRepository

SMOKA = {"item": "Skrzynia Komnaty Smoka", "total_price": 230_000_000,
         "unit_price": 23_000_000, "quantity": 10}
KSIEGA = {"item": "Księga Przebicia", "total_price": 15_000_000,
          "unit_price": 15_000_000, "quantity": 1}
MISS = {"item": "Odłamek Metina", "total_price": 70_000_000,
        "unit_price": None, "quantity": None}


def _reading(data):
    base = {"seconds": 0.1, "eval_count": 50, "error": None, "source": "vlm"}
    return {**base, **data}


class EngineTests(unittest.TestCase):
    def _build(self, repo: ScanRepository) -> ShopScan:
        scan = ShopScan(scan_id="t1", seller="Kocur", status=ScanStatus.ANALYZING)
        repo.create(scan)
        for slot_no, frames in ((1, 2), (2, 1), (3, 2)):
            paths = []
            for index in range(1, frames + 1):
                image = Image.new("RGB", (20, 20), "white")
                paths.append(
                    repo.save_tooltip_image(scan.scan_id, slot_no, index, image))
            scan.slots[slot_no] = ItemObservation(
                slot=slot_no, row=0, column=slot_no, images=paths,
                status=ScanStatus.CAPTURED)
            scan.occupied_slots += 1
            scan.captured_slots += 1
        return scan

    def test_engine_assigns_statuses_and_exports_only_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = ScanRepository(root / "scans")
            scan = self._build(repo)

            # Kolejność wywołań: slot1 f1,f2 | slot2 f1 | slot3 f1,f2
            sequence = [_reading(SMOKA), _reading(SMOKA), _reading(KSIEGA),
                        _reading(MISS), _reading(MISS)]
            engine = VlmAnalysisEngine(use_ocr=False)
            with patch.object(ollama_reader, "read_tooltip", side_effect=sequence):
                result = engine.analyze(scan, repo)

            self.assertEqual(result.slots[1].status, ScanStatus.VERIFIED)
            self.assertEqual(result.slots[2].status, ScanStatus.PROVISIONAL)
            self.assertEqual(result.slots[3].status, ScanStatus.REVIEW)
            self.assertEqual(result.slots[3].validation["reason"],
                             "unit_price_missing")
            # slot1 potwierdzony przez zgodną drugą klatkę
            self.assertIn("vlm_frame_2", result.slots[1].evidence)
            # skan: mieszane -> PROVISIONAL
            self.assertEqual(result.status, ScanStatus.PROVISIONAL)

            csv_path = root / "ceny.csv"
            exported = CSVExporter(csv_path).export(result)
            self.assertEqual(exported, 1)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["item"], "Skrzynia Komnaty Smoka")
            self.assertEqual(rows[0]["price"], "23000000")
            self.assertEqual(rows[0]["quantity"], "10")

    @staticmethod
    def _shop_reader(items):
        def reader(_image):
            return {"items": items, "error": None, "seconds": 0.5}
        return reader

    def _single_verified_scan(self, repo: ScanRepository) -> ShopScan:
        """Sklep 1-slotowy, zweryfikowany dymkiem (occupied=1, assigned=1)."""
        scan = ShopScan(scan_id="solo", seller="Kocur", status=ScanStatus.ANALYZING)
        repo.create(scan)
        paths = [repo.save_tooltip_image(scan.scan_id, 1, i, Image.new("RGB", (20, 20), "white"))
                 for i in (1, 2)]
        scan.slots[1] = ItemObservation(slot=1, row=0, column=1, images=paths,
                                        status=ScanStatus.CAPTURED)
        scan.occupied_slots = 1
        scan.captured_slots = 1
        engine = VlmAnalysisEngine(use_ocr=False)
        with patch.object(ollama_reader, "read_tooltip",
                          side_effect=[_reading(SMOKA), _reading(SMOKA)]):
            engine.analyze(scan, repo)
        return scan

    def _run_default(self, repo: ScanRepository) -> ShopScan:
        # _build: 3 zajęte sloty; tylko slot1 (Smoka) -> VERIFIED -> assigned=1.
        scan = self._build(repo)
        sequence = [_reading(SMOKA), _reading(SMOKA), _reading(KSIEGA),
                    _reading(MISS), _reading(MISS)]
        engine = VlmAnalysisEngine(use_ocr=False)
        with patch.object(ollama_reader, "read_tooltip", side_effect=sequence):
            engine.analyze(scan, repo)
        return scan

    def test_inventory_audit_complete_when_all_slots_assigned(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = ScanRepository(root / "scans")
            scan = self._single_verified_scan(repo)

            audit = scan.inventory_audit
            self.assertEqual(audit["occupied_slots"], 1)
            self.assertEqual(audit["pipeline_stack_count"], 1)
            self.assertEqual(audit["unassigned_slots"], 0)
            self.assertEqual(audit["audit_status"], "complete")
            self.assertNotIn("vlm", audit)   # domyślnie BEZ VLM

            csv_path = root / "ceny.csv"
            self.assertEqual(CSVExporter(csv_path).export(scan), 1)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["occupied_slots"], "1")
            self.assertEqual(rows[0]["unassigned_slots"], "0")
            self.assertEqual(rows[0]["audit_status"], "complete")

    def test_inventory_audit_partial_counts_unassigned_without_naming(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = ScanRepository(root / "scans")
            scan = self._run_default(repo)

            audit = scan.inventory_audit
            self.assertEqual(audit["occupied_slots"], 3)
            self.assertEqual(audit["pipeline_stack_count"], 1)   # tylko Smoka VERIFIED
            self.assertEqual(audit["unassigned_slots"], 2)       # uczciwy unassigned
            self.assertEqual(audit["audit_status"], "partial")
            # nieprzypisane sloty NIE doklejone do żadnej nazwy
            self.assertEqual([o["item"] for o in audit["offers"]],
                             ["Skrzynia Komnaty Smoka"])

            csv_path = root / "ceny.csv"
            self.assertEqual(CSVExporter(csv_path).export(scan), 1)  # NIE blokuje
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["unassigned_slots"], "2")
            self.assertEqual(rows[0]["audit_status"], "partial")

    def test_experimental_vlm_shop_audit_is_diagnostics_only(self) -> None:
        # VLM nazywa ikony z wyglądu — nic nie pasuje do nazwy z dymka (matched=0),
        # ale audyt kompletności i tak jest deterministyczny.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = ScanRepository(root / "scans")
            scan = self._build(repo)
            repo.save_shop_image(scan.scan_id, Image.new("RGB", (40, 40), "black"))
            sequence = [_reading(SMOKA), _reading(SMOKA), _reading(KSIEGA),
                        _reading(MISS), _reading(MISS)]
            engine = VlmAnalysisEngine(
                use_ocr=False, use_vlm_shop_audit=True,
                shop_inventory_reader=self._shop_reader(
                    [{"item": "złota skrzynia", "slots": 1}, {"item": "kostur", "slots": 2}]),
            )
            with patch.object(ollama_reader, "read_tooltip", side_effect=sequence):
                engine.analyze(scan, repo)

            audit = scan.inventory_audit
            self.assertEqual(audit["audit_status"], "partial")   # deterministyczny
            self.assertIn("vlm", audit)
            self.assertTrue(audit["vlm"]["available"])
            self.assertEqual(audit["vlm"]["matched"], [])        # nazwy nie pasują
            self.assertEqual(len(audit["vlm"]["vlm_only"]), 2)

    def test_manifest_roundtrip_after_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = ScanRepository(Path(temp) / "scans")
            scan = self._build(repo)
            sequence = [_reading(SMOKA), _reading(SMOKA), _reading(KSIEGA),
                        _reading(MISS), _reading(MISS)]
            engine = VlmAnalysisEngine(use_ocr=False)
            with patch.object(ollama_reader, "read_tooltip", side_effect=sequence):
                engine.analyze(scan, repo)
            repo.save_manifest(scan)
            restored = repo.load(scan.scan_id)
            self.assertEqual(restored.slots[1].status, ScanStatus.VERIFIED)
            self.assertEqual(restored.slots[1].validation["unit_price"], 23_000_000)


if __name__ == "__main__":
    unittest.main()
