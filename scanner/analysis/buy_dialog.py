"""Parser popupa kupna (PPM na item) → nazwa + ilość + cena (Claude, pure).

**Po co.** PPM na item w sklepie offline otwiera dialog na ŚRODKU ekranu: „Czy chcesz kupić
<NAZWA> x<ILOŚĆ>? / Cena wynosi: <CENA>". Stała pozycja, ciemne tło, jasny tekst → OCR niemal
idealny (vs kruchy hover-dymek ~50% yield + zasłanianie przez tłum). Ten moduł parsuje surowe
linie OCR popupa na strukturę. Czysta logika: zero gry/OCR/kliknięć — DeepSeek robi PPM+grab+ESC
i podaje tu linie. Odporny na szum OCR jak [[tooltip-marker-ocr-yield]] (`sale_marker`).

Bezpieczeństwo (po stronie DeepSeeka, nie tu): popup to dialog KUPNA — zawsze ESC/✗, NIGDY ✓.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BuyDialog:
    name: str
    quantity: int
    price: int | None     # None = cena nieczytelna (np. OCR x1 zwrócił śmieć) → DeepSeek: re-PPM/upscale


_BUY_LINE = re.compile(r"kup|chcesz", re.IGNORECASE)        # linia pytania kupna
_VERB_KUP = re.compile(r"kup\w*\s+", re.IGNORECASE)         # „kupić " tuż przed nazwą (preferowane)
_VERB_CHCESZ = re.compile(r"chcesz\s+", re.IGNORECASE)      # fallback gdy OCR zgubił „kupić"
# OCR myli 1↔l/I/|, 0↔o/O (np. „x1"→„xl", widziane na żywo). W LICZNOŚCI (tylko po końcowym „x")
# normalizujemy te znaki na cyfry; reszta nazwy nietknięta (np. „+l" w nazwie itemu zostaje).
_QTY_CHARS = r"[\dlI|oO]"
_OCR_TO_DIGIT = str.maketrans({"l": "1", "I": "1", "|": "1", "o": "0", "O": "0"})
_QTY_END = re.compile(rf"[xX]\s*({_QTY_CHARS}{{1,3}})\s*\??\s*$")    # „xN?" na końcu = liczność stacka
_QTY_ANY = re.compile(rf"[xX]\s*({_QTY_CHARS}{{1,3}})(?!\w)")
_PRICE_CTX = re.compile(r"(?:wyn\w*|cena)\D{0,6}(\d[\d.,\s]*\d|\d)", re.IGNORECASE)  # po „wynosi/cena"
_NUM_TOKEN = re.compile(r"\d[\d.,\s]*\d|\d")


def _to_int(text: str) -> int | None:
    digits = re.sub(r"\D", "", text or "")
    return int(digits) if digits else None


def parse_buy_dialog(lines: Iterable[str]) -> BuyDialog | None:
    """Surowe linie OCR popupa → `BuyDialog(name, quantity, price)`. `None` = to nie popup kupna.

    - quantity: „xN" na końcu pytania (domyślnie 1, gdy item niestackowalny — bez „xN").
    - price: liczba po „wynosi/cena"; fallback = token z NAJWIĘKSZĄ liczbą cyfr (cena ma ich więcej
      niż liczność). `None` gdy nic czytelnego (degradacja łagodna — nie zgaduj).
    - name: tekst po „kupić/chcesz" bez końcowego „xN?".
    """

    raw = [str(line) for line in lines if line is not None and str(line).strip()]
    if not raw:
        return None
    qline = next((line for line in raw if _BUY_LINE.search(line)), None)
    if qline is None:
        return None

    # --- ilość ---
    m = _QTY_END.search(qline) or _QTY_ANY.search(qline)
    quantity = int(m.group(1).translate(_OCR_TO_DIGIT)) if m else 1

    # --- nazwa ---
    name = qline
    vm = _VERB_KUP.search(qline) or _VERB_CHCESZ.search(qline)   # „kupić" przed „chcesz"
    if vm:
        name = qline[vm.end():]
    name = re.sub(rf"\s*[xX]\s*{_QTY_CHARS}{{1,3}}\s*\??\s*$", "", name)   # utnij „ x2?"/„ xl?"
    name = name.strip(" ?\t").strip()
    if not name:
        return None

    # --- cena ---
    blob = " ".join(raw)
    price: int | None = None
    pm = _PRICE_CTX.search(blob)
    if pm:
        price = _to_int(pm.group(1))
    if price is None:
        # fallback: token z największą liczbą cyfr (cena >> liczność)
        best = ""
        for tok in _NUM_TOKEN.findall(blob):
            if len(re.sub(r"\D", "", tok)) > len(re.sub(r"\D", "", best)):
                best = tok
        price = _to_int(best)
        # nie myl liczności z ceną: jeśli „cena" = qty i to jedyna liczba, odrzuć
        if price is not None and price == quantity and len(re.sub(r"\D", "", best)) <= 2:
            price = None

    return BuyDialog(name=name, quantity=quantity, price=price)
