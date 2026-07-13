"""Testy deterministycznego audytu kompletności sklepu (inventory_audit.py).

Bez VLM: nazwa przedmiotu nie istnieje na shop.png. Mierzymy occupied vs
bezpiecznie przypisane (= unassigned) i status complete/partial.
"""

import unittest

from scanner.analysis.inventory_audit import (
    COMPLETE,
    PARTIAL,
    audit_shop,
    build_vlm_counts,
    normalize_item,
    vlm_diagnostics,
)


class NormalizeItemTest(unittest.TestCase):
    def test_collapses_whitespace_and_casefolds(self):
        self.assertEqual(
            normalize_item("  Odłamek   Metina "), normalize_item("odłamek metina")
        )

    def test_distinct_names_stay_distinct(self):
        self.assertNotEqual(normalize_item("Kamień Duszy"), normalize_item("Kamień Mocy"))


class AuditShopTest(unittest.TestCase):
    def _offer(self, item, *, stack_count, quantity):
        return {"item": item, "stack_count": stack_count, "quantity": quantity}

    def test_complete_when_all_slots_assigned(self):
        offers = [
            self._offer("Odłamek Metina", stack_count=5, quantity=500),
            self._offer("Kamień Duszy", stack_count=3, quantity=30),
        ]
        audit = audit_shop(8, offers)
        self.assertEqual(audit["occupied_slots"], 8)
        self.assertEqual(audit["pipeline_stack_count"], 8)
        self.assertEqual(audit["unassigned_slots"], 0)
        self.assertEqual(audit["audit_status"], COMPLETE)
        self.assertEqual(len(audit["offers"]), 2)

    def test_partial_when_slots_unassigned(self):
        # 10 wykrytych, ale tylko 6 bezpiecznie przypisanych (np. grupa bez konsensusu).
        offers = [self._offer("Odłamek Metina", stack_count=6, quantity=600)]
        audit = audit_shop(10, offers)
        self.assertEqual(audit["pipeline_stack_count"], 6)
        self.assertEqual(audit["unassigned_slots"], 4)
        self.assertEqual(audit["audit_status"], PARTIAL)

    def test_no_offers_with_occupied_is_partial(self):
        audit = audit_shop(5, [])
        self.assertEqual(audit["unassigned_slots"], 5)
        self.assertEqual(audit["audit_status"], PARTIAL)

    def test_empty_shop_is_complete(self):
        audit = audit_shop(0, [])
        self.assertEqual(audit["occupied_slots"], 0)
        self.assertEqual(audit["unassigned_slots"], 0)
        self.assertEqual(audit["audit_status"], COMPLETE)

    def test_assigned_never_exceeds_occupied(self):
        # Gdyby detekcja zajętości zaniżyła, unassigned nie schodzi poniżej zera.
        offers = [self._offer("X", stack_count=9, quantity=9)]
        audit = audit_shop(7, offers)
        self.assertEqual(audit["unassigned_slots"], 0)
        self.assertEqual(audit["audit_status"], COMPLETE)

    def test_does_not_attribute_unassigned_to_a_name(self):
        # Nieprzypisane sloty NIE doklejają się do żadnej oferty po nazwie.
        offers = [self._offer("Pióro", stack_count=3, quantity=30)]
        audit = audit_shop(8, offers)
        (offer,) = audit["offers"]
        self.assertEqual(offer["stack_count"], 3)         # tylko potwierdzone
        self.assertEqual(audit["unassigned_slots"], 5)    # reszta = uczciwy unassigned


class VlmDiagnosticsTest(unittest.TestCase):
    """Eksperymentalna diagnostyka VLM — tylko podgląd, nigdy autorytet."""

    def test_build_vlm_counts_sums_and_normalizes(self):
        counts = build_vlm_counts([
            {"item": "kostur", "slots": 2},
            {"item": "Kostur", "slots": 1},
            {"item": "", "slots": 9},
            {"item": "zła", "slots": 0},
        ])
        self.assertEqual(counts, {normalize_item("kostur"): 3})

    def test_vlm_only_surfaces_unmatched_names(self):
        offers = [{"item": "Odłamek Metina", "stack_count": 1, "quantity": 100}]
        # VLM nazywa ikony z wyglądu — nic nie pasuje do nazwy z dymka.
        counts = build_vlm_counts([
            {"item": "złota skrzynia", "slots": 1},
            {"item": "kostur", "slots": 2},
        ])
        diag = vlm_diagnostics(offers, counts)
        self.assertEqual(diag["matched"], [])
        self.assertEqual(
            diag["vlm_only"],
            [(normalize_item("kostur"), 2), (normalize_item("złota skrzynia"), 1)],
        )

    def test_vlm_matched_when_name_happens_to_align(self):
        offers = [{"item": "Kostur", "stack_count": 2, "quantity": 2}]
        counts = build_vlm_counts([{"item": "kostur", "slots": 2}])
        diag = vlm_diagnostics(offers, counts)
        self.assertEqual(diag["matched"], [("Kostur", 2)])
        self.assertEqual(diag["vlm_only"], [])


if __name__ == "__main__":
    unittest.main()
