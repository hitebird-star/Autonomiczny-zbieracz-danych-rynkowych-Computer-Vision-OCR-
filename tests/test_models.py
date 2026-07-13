from __future__ import annotations

import unittest

from scanner.models import ItemObservation, ScanError, ScanStatus, ShopScan


class ShopScanTests(unittest.TestCase):
    def test_happy_path_reaches_provisional_then_verified(self) -> None:
        scan = ShopScan(scan_id="scan-1", occupied_slots=1)
        for status in (
            ScanStatus.APPROACHING,
            ScanStatus.OPENING,
            ScanStatus.OPENED,
            ScanStatus.CAPTURING,
            ScanStatus.CAPTURED,
            ScanStatus.QUEUED,
            ScanStatus.ANALYZING,
            ScanStatus.PROVISIONAL,
            ScanStatus.VERIFIED,
        ):
            scan.transition(status)

        self.assertEqual(scan.status, ScanStatus.VERIFIED)
        self.assertEqual(len(scan.status_history), 9)

    def test_cannot_skip_provisional(self) -> None:
        scan = ShopScan(scan_id="scan-2", status=ScanStatus.ANALYZING)
        with self.assertRaisesRegex(ValueError, "niedozwolone przejście"):
            scan.transition(ScanStatus.VERIFIED)

    def test_failed_requires_structured_error(self) -> None:
        scan = ShopScan(scan_id="scan-3")
        with self.assertRaisesRegex(ValueError, "wymaga ScanError"):
            scan.transition(ScanStatus.FAILED)

        error = ScanError(
            failed_stage="opening",
            reason="shop_window_not_detected",
            retry_count=2,
            recoverable=True,
        )
        scan.transition(ScanStatus.FAILED, error=error)
        self.assertEqual(scan.error, error)

    def test_manifest_roundtrip(self) -> None:
        observation = ItemObservation(
            slot=17,
            row=1,
            column=7,
            images=["tooltips/slot_017_1.png"],
            status=ScanStatus.PROVISIONAL,
            ai={
                "item": "Odłamek Metina",
                "total_price": 70_000_000,
                "unit_price": 7_000_000,
                "quantity": 10,
            },
            validation={"quantity": 10, "status": "provisional"},
            evidence=["vlm_frame_1"],
        )
        scan = ShopScan(
            scan_id="scan-4",
            seller="Kocur",
            status=ScanStatus.ANALYZING,
            occupied_slots=1,
            captured_slots=1,
            slots={17: observation},
            game_position=(532, 418),
        )

        restored = ShopScan.from_dict(scan.to_dict())

        self.assertEqual(restored.to_dict(), scan.to_dict())
        self.assertEqual(restored.slots[17].ai["total_price"], 70_000_000)


class ItemObservationTests(unittest.TestCase):
    def test_rejects_coordinates_outside_grid(self) -> None:
        with self.assertRaises(ValueError):
            ItemObservation(slot=100, row=0, column=0)
        with self.assertRaises(ValueError):
            ItemObservation(slot=0, row=10, column=0)

    def test_rejects_game_only_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "nieprawidłowy status"):
            ItemObservation(
                slot=0,
                row=0,
                column=0,
                status=ScanStatus.APPROACHING,
            )


if __name__ == "__main__":
    unittest.main()

