from __future__ import annotations

import unittest

from scanner.analysis.buy_dialog import parse_buy_dialog


class BuyDialogTests(unittest.TestCase):
    def test_real_ocr_x3(self):
        # prawdziwe wyjście win_ocr (x3) z popupa Glevii
        d = parse_buy_dialog(["Czy chcesz kupić Księga Władcy x2?", "Cena wynosi: 1.800.000"])
        self.assertIsNotNone(d)
        self.assertEqual(d.name, "Księga Władcy")
        self.assertEqual(d.quantity, 2)
        self.assertEqual(d.price, 1_800_000)

    def test_real_ocr_x2_with_button_noise(self):
        d = parse_buy_dialog(
            ["Czy chcesz kupić Księga Władcy x2?", "Cena wynosi: 1.800.000", "r—- r-ts--"]
        )
        self.assertEqual(d.name, "Księga Władcy")
        self.assertEqual(d.quantity, 2)
        self.assertEqual(d.price, 1_800_000)

    def test_degraded_x1_kupic_dropped_price_garbage(self):
        # x1: brak „kupić", cena to śmieć „I.BOO.OOO" → nazwa+ilość OK, cena None
        d = parse_buy_dialog(["Czy chcesz Księga Władcy x2", "I.BOO.OOO'", "Cena wynosi:"])
        self.assertEqual(d.name, "Księga Władcy")
        self.assertEqual(d.quantity, 2)
        self.assertIsNone(d.price)        # nie zgaduj — DeepSeek re-PPM/upscale

    def test_single_item_no_quantity(self):
        d = parse_buy_dialog(["Czy chcesz kupić Miecz Treningowy?", "Cena wynosi: 50.000"])
        self.assertEqual(d.name, "Miecz Treningowy")
        self.assertEqual(d.quantity, 1)   # brak „xN" → 1
        self.assertEqual(d.price, 50_000)

    def test_comma_separators(self):
        d = parse_buy_dialog(["Czy chcesz kupić Eliksir x10?", "Cena wynosi: 1,250,000"])
        self.assertEqual(d.name, "Eliksir")
        self.assertEqual(d.quantity, 10)
        self.assertEqual(d.price, 1_250_000)

    def test_space_separators(self):
        d = parse_buy_dialog(["Czy chcesz kupić Ruda x99?", "Cena wynosi: 1 800 000"])
        self.assertEqual(d.quantity, 99)
        self.assertEqual(d.price, 1_800_000)

    def test_price_on_same_blob_no_context_word(self):
        # gdyby „wynosi/cena" zgubione — fallback: token z największą liczbą cyfr
        d = parse_buy_dialog(["Czy chcesz kupić Pierścień x3?", "2.500.000"])
        self.assertEqual(d.quantity, 3)
        self.assertEqual(d.price, 2_500_000)

    def test_not_a_buy_dialog(self):
        self.assertIsNone(parse_buy_dialog(["Cena sprzedaży", "jakiś tekst"]))
        self.assertIsNone(parse_buy_dialog([]))
        self.assertIsNone(parse_buy_dialog(["", "   "]))

    def test_multiword_name(self):
        d = parse_buy_dialog(["Czy chcesz kupić Kamień Magii Życia x5?", "Cena wynosi: 320.000"])
        self.assertEqual(d.name, "Kamień Magii Życia")
        self.assertEqual(d.quantity, 5)
        self.assertEqual(d.price, 320_000)

    def test_ocr_x1_as_xl_real_garble(self):
        # realny garble z biegu 133123: OCR czyta „x1"→„xl"; ucinamy z nazwy, qty=1
        d = parse_buy_dialog(["Czy chcesz kupić Łuk Burzowy+l xl?", "Cena wynosi: 400.000.000"])
        self.assertEqual(d.name, "Łuk Burzowy+l")     # „+l" w nazwie zostaje, „ xl?" ucięte
        self.assertEqual(d.quantity, 1)               # „xl" → 1
        self.assertEqual(d.price, 400_000_000)

    def test_real_x30_stack(self):
        d = parse_buy_dialog(["Czy chcesz kupić Księga Opiekuna x30?", "Cena wynosi: 3.022.500.000"])
        self.assertEqual(d.name, "Księga Opiekuna")
        self.assertEqual(d.quantity, 30)
        self.assertEqual(d.price, 3_022_500_000)

    def test_xI_uppercase_and_pipe(self):
        self.assertEqual(parse_buy_dialog(["kupić Eliksir xI?", "wynosi 999"]).quantity, 1)
        self.assertEqual(parse_buy_dialog(["kupić Eliksir x|?", "wynosi 999"]).quantity, 1)


if __name__ == "__main__":
    unittest.main()
