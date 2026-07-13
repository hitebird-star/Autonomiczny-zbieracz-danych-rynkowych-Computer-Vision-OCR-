"""Odczyt sprzedawcy z paska tytułu otwartego sklepu."""

from __future__ import annotations

import re
from collections.abc import Callable

from PIL import Image

from scanner.config import GridGeometry
from scanner.runtime import ScreenBackend


_SELLER = re.compile(
    r"(?i)sklep\s*offline\s*[\(\[]\s*([^\)\]]+?)\s*[\)\]]"
)
_PAREN_SELLER = re.compile(r"[\(\[]\s*([^\)\]]{1,32}?)\s*[\)\]]")


def parse_seller(text: str) -> str:
    match = _SELLER.search(text or "")
    if match:
        return match.group(1).strip()
    # Windows OCR często gubi pierwszą literę albo myli środek frazy
    # "Sklep Offline", ale nick w nawiasie pozostaje czytelny. Reader dostaje
    # wyłącznie ciasny pasek tytułu, więc zawartość nawiasu jest bezpiecznym
    # fallbackiem i nie pochodzi z opisu przedmiotu.
    match = _PAREN_SELLER.search(text or "")
    return match.group(1).strip() if match else ""


class SellerReader:
    def __init__(
        self,
        screen: ScreenBackend,
        geometry: GridGeometry,
        *,
        recognizer: Callable[[Image.Image], list[dict]] | None = None,
    ) -> None:
        self.screen = screen
        self.geometry = geometry
        self._recognizer = recognizer

    def read(self) -> str:
        if self._recognizer is None:
            try:
                import win_ocr

                if not getattr(win_ocr, "AVAILABLE", False):
                    return ""
                recognizer = win_ocr.recognize
            except Exception:
                return ""
        else:
            recognizer = self._recognizer

        origin_x, origin_y = self.geometry.origin
        width = self.geometry.offset[0] + self.geometry.columns * self.geometry.cell
        image = self.screen.grab((origin_x + 20, origin_y, max(260, width - 30), 28))
        image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
        try:
            lines = recognizer(image)
        except Exception:
            return ""
        for line in lines:
            seller = parse_seller(str(line.get("text") or ""))
            if seller:
                return seller
        return parse_seller(" ".join(str(line.get("text") or "") for line in lines))
