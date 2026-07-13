"""Złoty zestaw walidatora (D-016/3) — przypadki z realnych dymków Glevii."""

from __future__ import annotations

import unittest

from scanner.analysis import validator


def _ai(item="Skrzynia Komnaty Smoka", total=230_000_000, unit=23_000_000):
    return {"item": item, "total_price": total, "unit_price": unit}


class ValidatorTests(unittest.TestCase):
    def test_single_consistent_read_is_provisional_not_verified(self) -> None:
        # Sedno D-018: spójny pojedynczy odczyt to PROVISIONAL, nie VERIFIED.
        result = validator.validate(_ai())
        self.assertEqual(result["status"], validator.PROVISIONAL)
        self.assertEqual(result["quantity"], 10)
        self.assertEqual(result["evidence"], ["vlm_primary"])

    def test_second_frame_agreement_verifies(self) -> None:
        result = validator.validate(
            _ai(),
            confirmations=[{"source": "vlm_frame_2",
                            "total_price": 230_000_000, "unit_price": 23_000_000}],
        )
        self.assertEqual(result["status"], validator.VERIFIED)
        self.assertIn("vlm_frame_2", result["evidence"])

    def test_total_only_confirmation_does_not_verify(self) -> None:
        # Zgodność samego total nie dotyka pola ryzyka (unit/qty) -> nadal prowizja.
        result = validator.validate(
            _ai(),
            confirmations=[{"source": "ocr", "total_price": 230_000_000}],
        )
        self.assertEqual(result["status"], validator.PROVISIONAL)

    def test_ocr_unit_agreement_verifies(self) -> None:
        result = validator.validate(
            _ai(),
            confirmations=[{"source": "ocr", "total_price": 230_000_000,
                            "unit_price": 23_000_000}],
        )
        self.assertEqual(result["status"], validator.VERIFIED)

    def test_grid_quantity_agreement_verifies(self) -> None:
        result = validator.validate(_ai(), grid_quantity=10)
        self.assertEqual(result["status"], validator.VERIFIED)
        self.assertIn("grid", result["evidence"])

    def test_dangerous_unit_equals_total_is_provisional(self) -> None:
        # Księga Przebicia: AI gubi linię 1.5kk -> unit=total, qty=1. Spójne,
        # ale FAŁSZYWE. Walidator NIE może tego zweryfikować z jednego źródła.
        result = validator.validate(
            {"item": "Księga Przebicia", "total_price": 15_000_000,
             "unit_price": 15_000_000})
        self.assertEqual(result["status"], validator.PROVISIONAL)
        self.assertEqual(result["quantity"], 1)

    def test_grid_conflict_sends_to_review(self) -> None:
        result = validator.validate(
            {"item": "Księga Przebicia", "total_price": 15_000_000,
             "unit_price": 15_000_000},
            grid_quantity=10)
        self.assertEqual(result["status"], validator.REVIEW)
        self.assertEqual(result["reason"], "source_conflict")

    def test_frame_conflict_sends_to_review(self) -> None:
        result = validator.validate(
            _ai(),
            confirmations=[{"source": "vlm_frame_2",
                            "total_price": 230_000_000, "unit_price": 1_500_000}])
        self.assertEqual(result["status"], validator.REVIEW)
        self.assertEqual(result["reason"], "source_conflict")

    def test_missing_unit_is_review(self) -> None:
        result = validator.validate({"item": "Dobry Item", "total_price": 70_000_000,
                                     "unit_price": None})
        self.assertEqual(result["status"], validator.REVIEW)
        self.assertEqual(result["reason"], "unit_price_missing")

    def test_missing_total_is_review(self) -> None:
        result = validator.validate({"item": "Dobry Item", "total_price": None,
                                     "unit_price": 7_000_000})
        self.assertEqual(result["reason"], "total_price_missing")

    def test_missing_item_is_review(self) -> None:
        result = validator.validate({"item": None, "total_price": 70_000_000,
                                     "unit_price": 7_000_000})
        self.assertEqual(result["reason"], "item_missing")

    def test_inconsistent_total_is_review(self) -> None:
        # total nie dzieli się przez unit w granicach tolerancji.
        result = validator.validate({"item": "Dobry Item", "total_price": 100_000_000,
                                     "unit_price": 23_000_000})
        self.assertEqual(result["status"], validator.REVIEW)
        self.assertEqual(result["reason"], "inconsistent_total")

    def test_quantity_out_of_range_is_review(self) -> None:
        result = validator.validate({"item": "Dobry Item", "total_price": 10**11,
                                     "unit_price": 10_000})
        self.assertEqual(result["reason"], "quantity_out_of_range")

    def test_empty_confirmation_is_ignored(self) -> None:
        # Źródło, które nic nie odczytało, nie psuje ani nie potwierdza.
        result = validator.validate(
            _ai(),
            confirmations=[{"source": "ocr", "total_price": None,
                            "unit_price": None, "quantity": None}])
        self.assertEqual(result["status"], validator.PROVISIONAL)

    def test_implausible_low_price_is_review(self) -> None:
        # Pierwszy run: "Mityczny Apsik" 250/250 qty 1 -> nie może być VERIFIED.
        result = validator.validate(
            {"item": "Mityczny Apsik", "total_price": 250, "unit_price": 250},
            confirmations=[{"source": "vlm_frame_2", "total_price": 250,
                            "unit_price": 250}])
        self.assertEqual(result["status"], validator.REVIEW)
        self.assertEqual(result["reason"], "implausible_price")

    def test_numeric_name_is_review(self) -> None:
        result = validator.validate(
            {"item": "250", "total_price": 50_000_000, "unit_price": 5_000_000})
        self.assertEqual(result["reason"], "implausible_name")

    def test_short_name_is_review(self) -> None:
        result = validator.validate(
            {"item": "X", "total_price": 50_000_000, "unit_price": 5_000_000})
        self.assertEqual(result["reason"], "implausible_name")

    def test_price_at_floor_passes(self) -> None:
        result = validator.validate(
            {"item": "Tani Item", "total_price": 100_000, "unit_price": 1_000})
        self.assertEqual(result["status"], validator.PROVISIONAL)
        self.assertEqual(result["quantity"], 100)

    def test_real_cases_compute_quantity(self) -> None:
        cases = [
            ("Cor Draconis (antyczne)", 131_250_000, 750_000, 175),
            ("Tęczowa Perła", 170_000_000, 1_700_000, 100),
            ("Skrzynia Ks. Zwierzaka", 70_000_000, 350_000, 200),
            ("Sakwa Kamieni Duszy +9", 88_000_000, 11_000_000, 8),
            ("Kryształ Burzowy (Przetop)", 125_000_000, 25_000_000, 5),
        ]
        for item, total, unit, qty in cases:
            result = validator.validate(
                {"item": item, "total_price": total, "unit_price": unit})
            self.assertEqual(result["quantity"], qty, item)
            self.assertEqual(result["status"], validator.PROVISIONAL, item)


if __name__ == "__main__":
    unittest.main()
