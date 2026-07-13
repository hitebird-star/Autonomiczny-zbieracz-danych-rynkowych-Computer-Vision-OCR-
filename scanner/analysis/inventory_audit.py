"""Audyt KOMPLETNOŚCI sklepu — deterministyczny, bez VLM.

CZYSTA logika (bez I/O, sieci, gry) — w pełni testowalna offline, jak `validator.py`.

Dlaczego deterministyczny, a nie VLM. Nazwa przedmiotu NIE istnieje na `shop.png`
(jest tylko w dymku na hover), więc VLM patrzący na całość uczciwie widzi ikonę
("kostur", "złota skrzynia"), a nie "Odłamek Metina" — pomiar live dał 0/96
dopasowań nazw. Tożsamość daje wyłącznie dymek + bezpieczny konsensus grupy
(`group_consensus`). Audyt mierzy więc to, co FAKTYCZNIE widać:

- ``occupied_slots``       — wszystkie wykryte sloty sklepu (piksele, deterministyczne)
- ``pipeline_stack_count`` — sloty BEZPIECZNIE przypisane (Phase B + group_consensus)
- ``unassigned_slots``     — ``occupied - pipeline`` = najważniejsza metryka kompletności
- ``audit_status``         — ``complete`` / ``partial`` (partial ⇔ zostały nieprzypisane sloty)

Per oferta: ``stack_count`` (sloty potwierdzonej grupy) i ``quantity`` (suma sztuk).
NIE przypisujemy nieprzypisanych slotów do żadnej nazwy — dla nierozpoznanej grupy
nie wiemy, którego itemu brakuje; uczciwy wynik to ``unassigned_slots``.

VLM całego widoku może DODATKOWO dorzucić surową diagnostykę (``vlm_diagnostics``),
ale TYLKO za flagą eksperymentalną — nigdy jako autorytet nazw/liczb.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# Werdykt kompletności sklepu (string, by storage/CSV mapował 1:1).
COMPLETE = "complete"   # każdy wykryty slot bezpiecznie przypisany
PARTIAL = "partial"     # zostały sloty bez reprezentanta / z rozbieżnym konsensusem


def normalize_item(name: str) -> str:
    """Nazwa do porównań: zwinięte spacje, bez ogona, case-fold (jak CSV)."""

    return re.sub(r"\s+", " ", str(name)).strip().casefold()


def _non_negative(value: object) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return result if result > 0 else 0


def audit_shop(
    occupied_slots: object, offers: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    """Deterministyczny audyt kompletności sklepu (bez VLM).

    ``occupied_slots`` — liczba wykrytych zajętych slotów (``scan.occupied_slots``).
    ``offers`` — oferty po tożsamości (z dymka + konsensusu), każda:
        ``{item, stack_count, quantity}``.

    Zwraca dict zapisywany w ``ShopScan.inventory_audit``:
        ``{occupied_slots, pipeline_stack_count, unassigned_slots, audit_status, offers}``.
    ``pipeline_stack_count`` = suma ``stack_count`` (sloty bezpiecznie przypisane);
    ``unassigned_slots`` = ``occupied - pipeline`` (nie przypisujemy ich do nazw).
    """

    normalized: list[dict[str, object]] = []
    assigned = 0
    for offer in offers:
        stack_count = _non_negative(offer.get("stack_count"))
        assigned += stack_count
        normalized.append({
            "item": str(offer.get("item") or ""),
            "stack_count": stack_count,
            "quantity": _non_negative(offer.get("quantity")),
        })

    occupied = _non_negative(occupied_slots)
    unassigned = max(0, occupied - assigned)
    return {
        "occupied_slots": occupied,
        "pipeline_stack_count": assigned,
        "unassigned_slots": unassigned,
        "audit_status": PARTIAL if unassigned > 0 else COMPLETE,
        "offers": normalized,
    }


# --- eksperymentalna diagnostyka VLM całego widoku (NIE audyt, NIE autorytet) ---

def build_vlm_counts(raw_items: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Złóż surowe ``[{item, slots}]`` od VLM w ``{znormalizowana_nazwa: suma}``.

    Wyłącznie do diagnostyki eksperymentalnej (``vlm_diagnostics``). Powtórzone
    nazwy sumujemy, nieparsowalne/puste pomijamy.
    """

    counts: dict[str, int] = {}
    for entry in raw_items:
        name = entry.get("item")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            value = int(entry.get("slots"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        key = normalize_item(name)
        counts[key] = counts.get(key, 0) + value
    return counts


def vlm_diagnostics(
    offers: Iterable[Mapping[str, object]], vlm_counts: Mapping[str, int]
) -> dict[str, object]:
    """Surowe porównanie nazw VLM z ofertami — TYLKO diagnostyka (flaga eksperymentalna).

    Zwraca, ile ofert VLM w ogóle nazwał (``matched``) i pozycje, których VLM widzi,
    a pipeline nie zgłosił (``vlm_only``). Pomiar live: matched≈0 (nazwy nie ma na
    obrazie), więc to NIE może gatekeepować CSV — służy tylko do podglądu/benchmarku.
    """

    matched: list[tuple[str, int]] = []
    seen: set[str] = set()
    for offer in offers:
        key = normalize_item(str(offer.get("item") or ""))
        seen.add(key)
        if key in vlm_counts:
            matched.append((str(offer.get("item") or ""), vlm_counts[key]))
    vlm_only = sorted((n, c) for n, c in vlm_counts.items() if n not in seen)
    return {"matched": matched, "vlm_only": vlm_only}
