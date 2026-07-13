"""Rozpoznanie markera dymka sklepu „Cena sprzedaży" odporne na OCR (Claude, offline-pure).

Root cause biegu 18:37 (shop-00008: 24 nieudane sloty, ~85 s zmarnowane, recovery 0/6):
`_tooltip_bbox` POPRAWNIE wykrywa panel dymka (14/14), ale `_has_sales_marker` odrzucał go,
bo żądał dosłownego podłańcucha „sprze" w JEDNEJ linii po `re.sub(r"[^a-z]","")`. OCR gry:
  * myli „Cena" → „CenS" / „Cyja" / „Cenę"  → linia markera nie zawiera „cena",
  * zjada polskie `ż`/`r` w „sprzeda**ż**y"   → „sprze" rozpada się na „spre"/„sped".
Skutek: czytelny dymek („[Cena sprzedaży] 230.000.000") odrzucany → poll do timeoutu × próby.

Fix: dopasuj DYSTYNKTYWNY rdzeń słowa „sprzedaży" na poziomie CAŁEGO panelu (nie linii),
tolerując zjedzone znaki: `sp.{0,4}eda`. Walidacja offline na 14 realnych pudłach shop-00008:
  stary matcher: 0/14 · `sp.{0,4}eda`: 13/14 · false-pozytywy na 14 klatkach bez dymka: 0/14.

Czysta logika tekstu — `tooltip_capture._has_sales_marker` ma wołać `has_sale_marker(lines)`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# „sprzedaży": OCR gubi `ż` (->kasowane przez normalize) i czasem `r`, więc rdzeń to „sp…eda".
# Dystynktywne dla markera sprzedaży (zwykła „cena" w „Szacowana cena" nie ma „…eda").
_SALE_CORE = re.compile(r"sp.{0,4}eda")


def normalize_marker(text: str) -> str:
    """Casefold + usunięcie wszystkiego poza a-z (jak legacy `_normalize_marker`).

    To samo zachowanie co dotychczas — `ż`,`ł`,cyfry,spacje,nawiasy znikają. Różnica jest
    DOPIERO w dopasowaniu: rdzeń zamiast sztywnego „cena"+„sprze".
    """

    return re.sub(r"[^a-z]", "", (text or "").casefold())


def has_sale_marker(lines: Iterable[str]) -> bool:
    """Czy odczytane linie OCR zawierają marker „Cena sprzedaży" (odpornie na mielenie).

    Łączy WSZYSTKIE linie w jeden znormalizowany blob (marker bywa rozbity między linie,
    a „Cena" mylone na „CenS"/„Cyja") i szuka rdzenia `sp.{0,4}eda`. Panel jest już
    potwierdzony jako ciemny dymek przez `_tooltip_bbox`, więc rdzeń nie daje fałszywek.
    """

    blob = normalize_marker(" ".join(str(line) for line in lines))
    return bool(_SALE_CORE.search(blob))
