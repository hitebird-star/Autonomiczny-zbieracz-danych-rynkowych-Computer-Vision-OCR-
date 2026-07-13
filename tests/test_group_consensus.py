from __future__ import annotations

import unittest
import csv
import tempfile
from pathlib import Path

from scanner.analysis.group_consensus import apply_group_consensus
from scanner.analysis.engine import VlmAnalysisEngine
from scanner.models import ItemObservation, ScanStatus, ShopScan
from scanner.storage import CSVExporter


def _representative(
    slot: int,
    *,
    status: ScanStatus,
    item: str = "Odłamek Metina",
    unit_price: int = 7_000_000,
    quantity: int = 100,
) -> ItemObservation:
    return ItemObservation(
        slot=slot,
        row=slot // 10,
        column=slot % 10,
        icon_group=1,
        images=[f"tooltips/slot_{slot:03d}_1.png"],
        status=status,
        validation={
            "status": status.value,
            "item": item,
            "unit_price": unit_price,
            "quantity": quantity,
        },
    )


def _deferred(slot: int) -> ItemObservation:
    return ItemObservation(
        slot=slot,
        row=slot // 10,
        column=slot % 10,
        icon_group=1,
        status=ScanStatus.CAPTURED,
        evidence=["stack_representative"],
    )


class GroupConsensusTests(unittest.TestCase):
    def test_verified_agreeing_representatives_cover_all_deferred_slots(self) -> None:
        scan = ShopScan(
            scan_id="group-ok",
            slots={
                0: _representative(0, status=ScanStatus.VERIFIED),
                1: _representative(1, status=ScanStatus.PROVISIONAL),
                2: _deferred(2),
                3: _deferred(3),
                4: _deferred(4),
            },
        )

        (decision,) = apply_group_consensus(scan)

        self.assertTrue(decision.applied)
        self.assertEqual(decision.representative_slots, (0, 1))
        self.assertEqual(decision.inherited_slots, (1, 2, 3, 4))
        for slot in (1, 2, 3, 4):
            observation = scan.slots[slot]
            self.assertIs(observation.status, ScanStatus.VERIFIED)
            self.assertEqual(observation.validation["item"], "Odłamek Metina")
            self.assertEqual(observation.validation["quantity"], 100)
            self.assertIn("group_consensus_of:0", observation.evidence)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ceny.csv"
            self.assertEqual(CSVExporter(path).export(scan), 1)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["stack_count"], "5")
            self.assertEqual(row["quantity"], "500")

    def test_mismatched_representatives_do_not_propagate(self) -> None:
        scan = ShopScan(
            scan_id="group-mismatch",
            slots={
                0: _representative(0, status=ScanStatus.VERIFIED),
                1: _representative(1, status=ScanStatus.PROVISIONAL, quantity=50),
                2: _deferred(2),
            },
        )

        (decision,) = apply_group_consensus(scan)

        self.assertFalse(decision.applied)
        self.assertEqual(decision.reason, "representative_mismatch")
        self.assertIs(scan.slots[2].status, ScanStatus.CAPTURED)

    def test_agreeing_provisional_representatives_verify_the_group(self) -> None:
        scan = ShopScan(
            scan_id="group-provisional",
            slots={
                0: _representative(0, status=ScanStatus.PROVISIONAL),
                1: _representative(1, status=ScanStatus.PROVISIONAL),
                2: _deferred(2),
            },
        )

        (decision,) = apply_group_consensus(scan)

        self.assertTrue(decision.applied)
        self.assertTrue(all(obs.status is ScanStatus.VERIFIED for obs in scan.slots.values()))

    def test_one_representative_never_multiplies_to_the_whole_group(self) -> None:
        scan = ShopScan(
            scan_id="group-singleton-proof",
            slots={
                0: _representative(0, status=ScanStatus.PROVISIONAL),
                1: _deferred(1),
                2: _deferred(2),
            },
        )

        (decision,) = apply_group_consensus(scan)

        self.assertFalse(decision.applied)
        self.assertEqual(decision.reason, "insufficient_read_representatives")
        self.assertIs(scan.slots[1].status, ScanStatus.CAPTURED)

    def test_engine_logs_applied_group_and_exports_whole_stack(self) -> None:
        class Repository:
            def __init__(self) -> None:
                self.events = []

            def append_event(self, scan_id, event, **data) -> None:
                self.events.append((scan_id, event, data))

        scan = ShopScan(
            scan_id="group-engine",
            slots={
                0: _representative(0, status=ScanStatus.VERIFIED),
                1: _representative(1, status=ScanStatus.PROVISIONAL),
                2: _deferred(2),
            },
        )
        repository = Repository()

        VlmAnalysisEngine._apply_group_consensus(scan, repository)

        self.assertEqual(repository.events[0][1], "group_consensus_applied")
        self.assertEqual(repository.events[0][2]["inherited_count"], 2)
        self.assertTrue(all(obs.status is ScanStatus.VERIFIED for obs in scan.slots.values()))


if __name__ == "__main__":
    unittest.main()
