"""Guard ceny-outliera (Claude, offline-pure): łapie cichy misread przy quorum=1.

**Po co.** Phase B z `quorum=1` (mały stos → 1 hover) NIE ma mniejszości, która
zdemaskowałaby jednolity misread OCR — pojedynczy zły odczyt (klasycznie cena ×1000 z
doklejonej cyfry, albo ×1/1000 z urwanej) dziedziczy CAŁY stos bez ostrzeżenia
(`representatives.py` §kontrakt). Ten guard jest ZEWNĘTRZNYM sprawdzianem zastępującym
brakujący konsensus: porównuje świeży odczyt do **referencji rynkowej** (mediana cen tego
itemu z innych sklepów — `OfferIndex.offers_for(item)`), i flaguje odchylenie do **re-hoveru**
albo kwarantanny. To NIE konsensus wewnątrz stosu (`stack_consensus.py`) — to sanity-check
o jeden poziom wyżej, dla grup bez materiału na konsensus.

Czysta logika — zero gry/OCR/obrazów, testowalna offline jak reszta `scanner/analysis/*`.
Decyzję (próg, czy re-hover czy odrzut) podejmuje DeepSeek/człowiek; moduł ją egzekwuje i mierzy.

Sygnał najmocniejszy = **przesunięcie cyfr** (`decimal_shift`): cena ≈ referencja×10^k to
podpis błędu OCR, nie premium itemu. Por. `representatives.py` (skąd quorum=1),
`offer_index.py` (skąd referencja/mediana), `APPROACH_A_SPEC.md` „DŁUGI BIEG" §3 (mitygacja).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

# Próg odchylenia od referencji (w obie strony): cena ≥ factor× lub ≤ 1/factor× mediany.
# 10× łapie błąd o jedną cyfrę (×10) i wyżej; pod nim mieszczą się realne różnice cen.
DEFAULT_FACTOR = 10.0
# Ile cen referencyjnych potrzeba, by mediana była wiarygodna. Poniżej — NO_REFERENCE
# (nie zgaduj na 1-2 próbkach; brak werdyktu ≠ OK, to „nie wiem, niska pewność").
MIN_REFERENCE_SAMPLES = 3
# Klasyczne przesunięcia cyfr OCR (doklejone/urwane zera). Sprawdzane w obie strony.
DECIMAL_SHIFTS = (10, 100, 1000)
# Tolerancja wokół round power-of-ten (±15%) — zakresy 10/100/1000 się nie nakładają.
SHIFT_TOLERANCE = 0.15


class PriceVerdict(str, Enum):
    """Werdykt guardu. `str`-Enum: porównywalny wprost ze stringiem (jak `NavReason`)."""

    OK = "ok"                    # w granicach — przepuść
    HIGH = "high_outlier"        # podejrzanie drogo (np. ×1000 z doklejonej cyfry)
    LOW = "low_outlier"          # podejrzanie tanio (urwana cyfra)
    INVALID = "invalid"          # cena <= 0 — niemożliwa
    NO_REFERENCE = "no_reference"  # za mało cen referencyjnych — nie da się ocenić

    def __str__(self) -> str:  # czytelne logi/diagnostyka
        return self.value


@dataclass(frozen=True, slots=True)
class PriceCheck:
    """Wynik sprawdzenia jednej ceny względem referencji rynkowej."""

    price: int
    verdict: PriceVerdict
    reference: float | None     # mediana referencyjna (None gdy NO_REFERENCE/INVALID)
    ratio: float | None         # price / reference (None gdy brak referencji)
    shift: int | None           # podpis błędu cyfr: +1000 = cena≈ref×1000, -1000 = ref/1000
    samples: int                # ile cen referencyjnych użyto

    @property
    def is_outlier(self) -> bool:
        """Czy werdykt sygnalizuje podejrzaną cenę (do re-hoveru/kwarantanny)."""

        return self.verdict in (PriceVerdict.HIGH, PriceVerdict.LOW, PriceVerdict.INVALID)

    @property
    def high_confidence(self) -> bool:
        """Outlier + dopasowane przesunięcie cyfr = niemal pewny misread OCR (nie premium)."""

        return self.is_outlier and self.shift is not None


def median(values: Iterable[float | int | None]) -> float | None:
    """Mediana (odporna na outliery — stąd, nie średnia). `None` dla pustego wejścia."""

    nums = sorted(float(v) for v in values if v is not None)
    n = len(nums)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def decimal_shift(
    price: float, reference: float, *, tolerance: float = SHIFT_TOLERANCE
) -> int | None:
    """Czy `price` ≈ `reference × 10^k` (k∈±{1,2,3})? Zwraca podpis przesunięcia.

    Dodatni = cena za duża (doklejone zera, np. +1000), ujemny = za mała (urwane).
    `None` = nie pasuje do żadnego round power-of-ten. To najmocniejszy sygnał misreadu:
    odróżnia błąd OCR (round ×10^k) od realnie drogiego/taniego itemu.
    """

    if reference <= 0 or price <= 0:
        return None
    ratio = price / reference
    for k in DECIMAL_SHIFTS:
        if abs(ratio - k) <= k * tolerance:        # cena ≈ ref × k
            return k
        if abs(ratio - 1.0 / k) <= (1.0 / k) * tolerance:  # cena ≈ ref / k
            return -k
    return None


def check_price(
    price: int,
    reference: float | None,
    *,
    factor: float = DEFAULT_FACTOR,
    samples: int = 0,
) -> PriceCheck:
    """Sklasyfikuj cenę względem gotowej referencji (mediany). Czyste.

    `reference=None` (za mało próbek) → NO_REFERENCE: guard się WSTRZYMUJE, nie zmyśla
    werdyktu. `price<=0` → INVALID. Inaczej test ilorazu w obie strony + podpis cyfr.
    """

    if price <= 0:
        return PriceCheck(price, PriceVerdict.INVALID, reference, None, None, samples)
    if reference is None or reference <= 0:
        return PriceCheck(price, PriceVerdict.NO_REFERENCE, None, None, None, samples)

    ratio = price / reference
    shift = decimal_shift(price, reference, tolerance=SHIFT_TOLERANCE)
    if ratio >= factor:
        verdict = PriceVerdict.HIGH
    elif ratio <= 1.0 / factor:
        verdict = PriceVerdict.LOW
    else:
        verdict = PriceVerdict.OK
    return PriceCheck(price, verdict, reference, ratio, shift, samples)


def check_against_market(
    price: int,
    reference_prices: Sequence[float | int],
    *,
    factor: float = DEFAULT_FACTOR,
    min_samples: int = MIN_REFERENCE_SAMPLES,
) -> PriceCheck:
    """Wygodne: zbuduj referencję (medianę) z cen rynkowych i sprawdź `price`.

    `reference_prices` = ceny tego itemu z INNYCH odczytów (np. `OfferIndex.offers_for`).
    Poniżej `min_samples` → NO_REFERENCE (nie ma z czym porównać — quorum=1 na świeżym
    itemie po prostu nie ma asekuracji; flaguj wtedy inną drogą, np. niska pewność).
    Bieżąca cena nie powinna być w `reference_prices` (inaczej zabrudzi własną medianę).
    """

    usable = [float(p) for p in reference_prices if p is not None and p > 0]
    if len(usable) < min_samples:
        return PriceCheck(price, PriceVerdict.NO_REFERENCE, None, None, None, len(usable))
    return check_price(price, median(usable), factor=factor, samples=len(usable))


def needs_rehover(check: PriceCheck) -> bool:
    """Czy ten odczyt zasługuje na re-hover (więcej reprezentantów) zamiast zaufania.

    Tak dla outlierów (HIGH/LOW/INVALID). NO_REFERENCE → False (re-hover nic nie da bez
    referencji — to sygnał dla innej polityki, nie dla tego guardu).
    """

    return check.is_outlier


# --------------------------------------------------------------------------- #
# Pomiar: ile outlierów siedzi już w offers.jsonl (read-only, zero gry)        #
# --------------------------------------------------------------------------- #

def screen_offers(
    offers: Iterable[tuple[str, int]],
    *,
    factor: float = DEFAULT_FACTOR,
    min_samples: int = MIN_REFERENCE_SAMPLES,
) -> dict[str, list[PriceCheck]]:
    """Per-item: sprawdź każdą cenę względem mediany POZOSTAŁYCH cen tego itemu.

    `offers` = pary `(item, unit_price)`. Leave-one-out: cena nie zabrudza własnej
    referencji. Zwraca `item -> [PriceCheck]` (tylko outliery). Do audytu istniejących
    danych: „ile ×1000 już wpadło do offers.jsonl mimo floor=3?".
    """

    by_item: dict[str, list[int]] = {}
    for item, price in offers:
        by_item.setdefault(item, []).append(price)

    flagged: dict[str, list[PriceCheck]] = {}
    for item, prices in by_item.items():
        hits: list[PriceCheck] = []
        for i, price in enumerate(prices):
            rest = prices[:i] + prices[i + 1:]
            chk = check_against_market(price, rest, factor=factor, min_samples=min_samples)
            if chk.is_outlier:
                hits.append(chk)
        if hits:
            flagged[item] = hits
    return flagged


def _load_offers(path: Path) -> list[tuple[str, int]]:
    import json

    out: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append((str(d["item"]), int(d["unit_price"])))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scanner.analysis.price_guard",
        description="Audyt outlierów ceny w offers.jsonl (read-only).",
    )
    parser.add_argument(
        "--offers", default="market_map/glevia_market/offers.jsonl",
        help="ścieżka do offers.jsonl",
    )
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR)
    parser.add_argument("--min-samples", type=int, default=MIN_REFERENCE_SAMPLES)
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    path = Path(args.offers)
    if not path.exists():
        print(f"Brak pliku {path!r} (offers.jsonl jeszcze pusty?).")
        return 1
    offers = _load_offers(path)
    flagged = screen_offers(offers, factor=args.factor, min_samples=args.min_samples)
    total = sum(len(v) for v in flagged.values())
    print(f"=== Price guard — audyt {len(offers)} ofert, factor=×{args.factor:g} ===")
    print(f"Outlierów: {total} w {len(flagged)} itemach")
    for item in sorted(flagged, key=lambda k: -len(flagged[k]))[:20]:
        for chk in flagged[item][:5]:
            tag = "shift×%+d" % chk.shift if chk.shift else "ratio %.1f" % (chk.ratio or 0)
            print(f"  {item[:30]:30} cena={chk.price:>10} med={chk.reference:>10.0f} "
                  f"[{chk.verdict}] {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
