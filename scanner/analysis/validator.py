"""Twarda walidacja matematyczna odczytów (Python zatwierdza, AI proponuje).

CZYSTA logika: bez I/O, bez gry, bez sieci — w pełni testowalna offline na
złotym zestawie z 26 dymków (D-016/3).

Drabina statusów (AI_DATA_CONTRACT §5, D-018):
- pojedynczy, matematycznie spójny odczyt AI            -> PROVISIONAL
- + niezależne potwierdzenie (2. klatka VLM / OCR / grid) -> VERIFIED
- sprzeczne źródła albo brak danych                      -> REVIEW

Samo równanie `total = unit × ilość` z JEDNEGO odczytu nie jest dowodem:
`total=15M, unit=15M, ilość=1` jest spójne, a prawda to `unit=1.5M, ilość=10`.
Dlatego potwierdzenie musi dotyczyć pola RYZYKA (unit lub quantity), nie samego
total.
"""

from __future__ import annotations

from typing import Any

# Status walidacji = wartości ScanStatus (string), żeby silnik mapował 1:1.
PROVISIONAL = "provisional"
VERIFIED = "verified"
REVIEW = "review"

MIN_QTY = 1
MAX_QTY = 9999
TOLERANCE_REL = 0.02   # tolerancja na zaokrąglenie unit (np. 1.7kk)

# Bramki plausibilności (L1) — przeciw fałszywym VERIFIED z błędnych crops.
# Pierwszy żywy run: "Mityczny Apsik" 200-250 Yang / qty 1 (VLM wyłuskał małą
# liczbę ze sceny bez dymka) przeszło na VERIFIED, bo dwie klatki tego samego
# modelu zgodziły się na ten sam błąd. Realne oferty rynkowe miały ceny ≥160k.
MIN_PLAUSIBLE_PRICE = 1000   # cena rynkowa poniżej tego = niemal pewny przekłam
MIN_NAME_ALPHA = 3           # nazwa musi mieć min. tyle liter (odsiewa "250")


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _plausible_name(name: str) -> bool:
    """Nazwa przedmiotu musi wyglądać jak nazwa, nie jak liczba/szum."""
    alpha = sum(ch.isalpha() for ch in name)
    return alpha >= MIN_NAME_ALPHA and not name.strip().isdigit()


def _result(status: str, *, item, total, unit, quantity, evidence, reason=None):
    return {
        "status": status,
        "item": item,
        "total_price": total,
        "unit_price": unit,
        "quantity": quantity,
        "evidence": list(evidence),
        "reason": reason,
    }


def validate(
    ai: dict[str, Any],
    *,
    confirmations: list[dict[str, Any]] | None = None,
    grid_quantity: int | None = None,
) -> dict[str, Any]:
    """Zwaliduj jeden odczyt AI względem niezależnych potwierdzeń.

    `ai` — {item, total_price, unit_price, quantity?} (quantity advisory; liczymy).
    `confirmations` — lista niezależnych odczytów (2. klatka VLM, OCR), każdy z
        opcjonalnymi total_price/unit_price/quantity oraz 'source'.
    `grid_quantity` — przyszłe źródło: licznik z gridu (D-007).

    Zwraca dict 'validation' z polem `status` ∈ {provisional, verified, review}.
    """
    item = ai.get("item")
    item = item.strip() if isinstance(item, str) else None
    total = _as_int(ai.get("total_price"))
    unit = _as_int(ai.get("unit_price"))

    if not item:
        return _result(REVIEW, item=item, total=total, unit=unit, quantity=None,
                       evidence=[], reason="item_missing")
    if not _plausible_name(item):
        return _result(REVIEW, item=item, total=total, unit=unit, quantity=None,
                       evidence=[], reason="implausible_name")
    if total is None or total <= 0:
        return _result(REVIEW, item=item, total=total, unit=unit, quantity=None,
                       evidence=[], reason="total_price_missing")
    if unit is None or unit <= 0:
        # brak ceny jednostkowej -> review (D-006)
        return _result(REVIEW, item=item, total=total, unit=unit, quantity=None,
                       evidence=[], reason="unit_price_missing")
    if total < MIN_PLAUSIBLE_PRICE or unit < MIN_PLAUSIBLE_PRICE:
        # absurdalnie niska cena = niemal pewny przekłam (mała liczba ze sceny)
        return _result(REVIEW, item=item, total=total, unit=unit, quantity=None,
                       evidence=[], reason="implausible_price")

    quantity = round(total / unit)
    if not (MIN_QTY <= quantity <= MAX_QTY):
        return _result(REVIEW, item=item, total=total, unit=unit, quantity=quantity,
                       evidence=[], reason="quantity_out_of_range")
    if abs(total - quantity * unit) > max(2, unit * TOLERANCE_REL):
        return _result(REVIEW, item=item, total=total, unit=unit, quantity=quantity,
                       evidence=[], reason="inconsistent_total")

    # Spójny pojedynczy odczyt = co najmniej PROVISIONAL.
    evidence = ["vlm_primary"]
    confirmed = False

    sources: list[dict[str, Any]] = list(confirmations or [])
    if grid_quantity is not None:
        sources.append({"source": "grid", "quantity": int(grid_quantity)})

    for conf in sources:
        ct = _as_int(conf.get("total_price"))
        cu = _as_int(conf.get("unit_price"))
        cq = _as_int(conf.get("quantity"))
        provided = [v for v in (ct, cu, cq) if v is not None]
        if not provided:
            continue  # źródło nic nie odczytało — ignorujemy (brak głosu)
        label = str(conf.get("source") or "confirmation")

        # Sprzeczność na którymkolwiek dostarczonym polu -> review.
        if ((ct is not None and ct != total)
                or (cu is not None and cu != unit)
                or (cq is not None and cq != quantity)):
            return _result(REVIEW, item=item, total=total, unit=unit,
                           quantity=quantity, evidence=evidence + [f"conflict:{label}"],
                           reason="source_conflict")

        # Potwierdzenie liczy się tylko, gdy dotyka POLA RYZYKA (unit lub qty).
        if cu is not None or cq is not None:
            confirmed = True
            evidence.append(label)

    status = VERIFIED if confirmed else PROVISIONAL
    return _result(status, item=item, total=total, unit=unit, quantity=quantity,
                   evidence=evidence, reason=None)
