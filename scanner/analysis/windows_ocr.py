"""Pomocniczy, NIEZALEŻNY odczyt OCR (Windows.Media.Ocr) do potwierdzania VLM.

To drugi silnik (inny niż model wizyjny), więc jego zgodność liczbowa promuje
rekord PROVISIONAL -> VERIFIED (AI_DATA_CONTRACT §5). Parser jest celowo lekki:
to koroborator, nie główne źródło prawdy. NIE importuje legacy `shop_scanner`
(scanner/ pozostaje odsprzężone od gry).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from PIL import Image

# win_ocr.py leży w katalogu głównym repo (obok pakietu scanner/).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import win_ocr
    AVAILABLE = bool(getattr(win_ocr, "AVAILABLE", False))
except Exception:
    win_ocr = None
    AVAILABLE = False

_GROUPED = re.compile(r"\d{1,3}(?:[.\s ]\d{3})+")
_UNIT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kkk|kk|k)\b", re.IGNORECASE)
_MULT = {"k": 1_000, "kk": 1_000_000, "kkk": 1_000_000_000}


def _norm(text: str) -> str:
    return (text or "").lower().replace(" ", " ")


def read_tooltip(image: Image.Image) -> dict[str, Any]:
    """Niezależny odczyt total/unit z dymka. Zwraca:

        {source:'ocr', total_price, unit_price, quantity, raw, error}

    Pola nieczytelne = None (wtedy walidator po prostu nie liczy tego głosu).
    """
    out: dict[str, Any] = {
        "source": "ocr", "total_price": None, "unit_price": None,
        "quantity": None, "raw": "", "error": None,
    }
    if not AVAILABLE:
        out["error"] = "win_ocr_unavailable"
        return out
    try:
        # x2 poprawia odczyt małego fontu; współrzędne nieistotne dla parsera.
        scaled = image.convert("RGB")
        scaled = scaled.resize((scaled.width * 2, scaled.height * 2), Image.LANCZOS)
        lines = win_ocr.recognize(scaled)
        out["raw"] = " | ".join(l["text"] for l in lines)

        marker_y = None
        for line in lines:
            if "sprzeda" in _norm(line["text"]):
                marker_y = line["box"][1]
                break

        # total = pogrupowana liczba poniżej markera [Cena sprzedaży]
        total = None
        best_y = None
        for line in lines:
            y0 = line["box"][1]
            if marker_y is not None and y0 < marker_y:
                continue
            match = _GROUPED.search(line["text"])
            if not match:
                continue
            value = int(re.sub(r"\D", "", match.group(0)))
            if 1000 <= value < 10**12 and (best_y is None or y0 < best_y):
                total, best_y = value, y0
        out["total_price"] = total

        # unit = z linii "X kk/k Yang za sztukę" (pomijamy linię "Won")
        for line in lines:
            norm = _norm(line["text"])
            if "za sztuk" not in norm or "won" in norm:
                continue
            match = _UNIT.search(line["text"])
            if match:
                number = float(match.group(1).replace(",", "."))
                out["unit_price"] = int(round(number * _MULT[match.group(2).lower()]))
                break
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
