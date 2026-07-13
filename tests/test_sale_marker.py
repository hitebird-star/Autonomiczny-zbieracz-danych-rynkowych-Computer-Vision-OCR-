"""Regresja markera „Cena sprzedaży" na REALNYCH stringach OCR z shop-00008 (bieg 18:37).

Każdy fixture to dosłowny odczyt `win_ocr` z zaakceptowanego panelu dymka slotu, który
STARY matcher odrzucał (slot padał `tooltip_not_detected`). Patrz `sale_marker.py`.
"""

from __future__ import annotations

import re
import unittest

from scanner.analysis.sale_marker import has_sale_marker, normalize_marker

# --- realne linie OCR markera z nieudanych slotów (unicode dosłownie jak czytał win_ocr) ---
# „Cena" mylone, `ż` zjadane; pełne panele skrócone do linii istotnych dla matchera.
REAL_FAILED_PANELS = {
    0: ["[Szacowana cena jednej sztuki)", "2.3kk Yang za sztuke", "[Cena sprŽedaŽy]•", "230.000.000"],
    1: ["[Szacowana cena jednej sztuki)", "[CenS sprŽedaŽy]", "0 10"],
    4: ["[Szacpwana cena jednejșztuki]", "[CenS spŕŽedaŽy]•", "22 01.000.00"],
    12: ["Krysztal Burzowy", "0.03 Won za sztuke", "(Cyja sprzedaży]", "50.000.000"],
    14: ["Z tego Hejnału", "[Cenę sprzedaży]"],
}

# klatki BEZ dymka (tło świata gry) — nie wolno dać false-positive
NO_TOOLTIP_LINES = [
    ["105", "Sklep Offline", "100", "100", "100"],
    ["Glevia", "HP", "MP", "EXP"],
    [""],
    [],
]


class SaleMarkerRecall(unittest.TestCase):
    def test_recovers_real_mangled_panels(self):
        for slot, lines in REAL_FAILED_PANELS.items():
            with self.subTest(slot=slot):
                self.assertTrue(has_sale_marker(lines), f"slot {slot} powinien matchować marker")

    def test_clean_phrase_matches(self):
        # niezmielony przypadek też musi przejść
        self.assertTrue(has_sale_marker(["[Cena sprzedazy]", "230.000.000"]))
        self.assertTrue(has_sale_marker(["[Cena sprzedaży]"]))

    def test_marker_split_across_lines(self):
        # „Cena" mylone na „CenS", więc poleganie na samym rdzeniu z innej linii
        self.assertTrue(has_sale_marker(["[CenS", "sprŽedaŽy]"]))


class SaleMarkerSpecificity(unittest.TestCase):
    def test_no_false_positive_without_marker(self):
        for lines in NO_TOOLTIP_LINES:
            with self.subTest(lines=lines):
                self.assertFalse(has_sale_marker(lines))

    def test_estimated_price_line_alone_is_not_sale_marker(self):
        # „Szacowana cena jednej sztuki" ma „cena" ale NIE „sprzedaży" → nie marker
        self.assertFalse(has_sale_marker(["[Szacowana cena jednej sztuki]", "0.00 Won za sztuke"]))


class NormalizeMarker(unittest.TestCase):
    def test_strips_to_az(self):
        self.assertEqual(normalize_marker("[Cena 230]•"), "cena")
        self.assertEqual(normalize_marker(None), "")

    def test_mangled_sprzedazy_keeps_core(self):
        # po normalizacji „sprŽedaŽy" -> „spreday", a rdzeń sp.{0,4}eda go łapie
        n = normalize_marker("sprŽedaŽy")
        self.assertTrue(re.search(r"sp.{0,4}eda", n))


if __name__ == "__main__":
    unittest.main()
