"""Phase B: wybór reprezentantów stosu do hoveru (budżet capture, pas Claude — spec).

Komplementarne do `stack_consensus.py`:
- **tu** decydujemy, KTÓRE sloty grupy ikon zhoverować (reduktor wejścia, capture),
- **tam** rekonsyliujemy ICH odczyty (konsensus wyjścia, engine).

Czysta logika — zero gry/OCR/obrazów, testowalna offline jak reszta `scanner/analysis/*`.
Pełny kontrakt + podstawa empiryczna: `scanner/analysis/PHASE_B_CONTRACT.md`.

Zasada bezpieczeństwa (z `STACK_CONSENSUS_CONTRACT.md` §4): poniżej kworum
`DEFAULT_QUORUM=3` odczytów konsensus NIE MA mniejszości, która zdemaskuje
jednolity misread (×1000) → pojedynczy reprezentant przepuszcza cichy błąd na CAŁĄ
grupę. Dlatego `quorum` jest FLOOREM, nie celem do zejścia w dół. Wartości
(`quorum`, `big_quorum`) są ARGUMENTAMI — finalną decyzję podejmuje DeepSeek/człowiek;
ten moduł jej nie przesądza, tylko ją egzekwuje i mierzy.

CLI (pomiar oszczędności na realnych manifestach, read-only):
    python -m scanner.analysis.representatives --scans scans
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

# Floor kworum = k0 z STACK_CONSENSUS_CONTRACT §4. Poniżej 3 brak mniejszości
# demaskującej jednolity misread; dlatego nie schodzimy poniżej tej wartości.
DEFAULT_QUORUM = 3
# Próg „dużej" grupy i (opcjonalnie) wyższe kworum dla niej — czytaj więcej, gdy
# stos duży (większa szansa misreadu), wciąż ograniczone wielkością grupy.
DEFAULT_BIG_THRESHOLD = 8
# Minimum slotów, by konsensus w ogóle mógł zadziałać (potrzebna ≥1 mniejszość).
CONSENSUS_MIN = 2


def reps_for_size(
    size: int,
    *,
    quorum: int = DEFAULT_QUORUM,
    big_threshold: int = DEFAULT_BIG_THRESHOLD,
    big_quorum: int | None = None,
) -> int:
    """Ile slotów grupy zhoverować przy danym rozmiarze stosu.

    - `size <= quorum`            → cały stos (i tak mały, hover wszystkich),
    - `size >  quorum`            → `quorum` reprezentantów (floor konsensusu),
    - `size >= big_threshold` i `big_quorum` ustawione → użyj `big_quorum`.
    Wynik nigdy nie przekracza `size` i nigdy nie spada poniżej 1 dla niepustej grupy.
    """

    if size <= 0:
        return 0
    if quorum < 1:
        raise ValueError("quorum musi być >= 1")
    q = quorum
    if big_quorum is not None and size >= big_threshold:
        if big_quorum < quorum:
            raise ValueError("big_quorum nie może być niższe niż quorum (to floor)")
        q = big_quorum
    return min(size, q)


@dataclass(frozen=True, slots=True)
class GroupSelection:
    """Plan hoveru dla jednej grupy ikon (lub slotu bez grupy)."""

    icon_group: int | None
    members: tuple[int, ...]            # wszystkie sloty grupy (posortowane)
    representatives: tuple[int, ...]    # które zhoverować/odczytać
    deferred: tuple[int, ...]           # pominięte (dziedziczą consensus_unit)

    @property
    def has_consensus_material(self) -> bool:
        """Czy zhoverowano dość slotów, by konsensus miał mniejszość (≥2)."""

        return len(self.representatives) >= CONSENSUS_MIN


def select_representatives(
    groups: Mapping[int | None, Sequence[int]],
    *,
    quorum: int = DEFAULT_QUORUM,
    big_threshold: int = DEFAULT_BIG_THRESHOLD,
    big_quorum: int | None = None,
) -> dict[int | None, GroupSelection]:
    """Wybierz reprezentantów dla każdej grupy ikon. Czyste.

    `groups`: `icon_group -> lista slotów`. Klucz `None` = sloty bez `icon_group`
    — każdy taki slot jest własnym reprezentantem (BRAK dedupu bez klucza grupy;
    bezpieczny domyślny wybór). Reprezentanci to pierwsze `reps_for_size(...)`
    slotów po posortowaniu (deterministycznie, niskie id pierwsze).
    """

    out: dict[int | None, GroupSelection] = {}
    for icon_group, raw in groups.items():
        members = tuple(sorted(raw))
        if icon_group is None:
            # Bez klucza grupy nie dedupujemy: każdy slot to własny reprezentant.
            out[None] = GroupSelection(None, members, members, ())
            continue
        n = reps_for_size(
            len(members), quorum=quorum,
            big_threshold=big_threshold, big_quorum=big_quorum,
        )
        out[icon_group] = GroupSelection(
            icon_group, members, members[:n], members[n:],
        )
    return out


# --------------------------------------------------------------------------- #
# Bramka POST-hover: ile odczytów dymka FAKTYCZNIE się udało (≥2 = przejdź)     #
# --------------------------------------------------------------------------- #
# `reps_for_size` mówi ile slotów ZAPLANOWAĆ; ta bramka pilnuje ile się UDAŁO.
# Plan ≠ wynik: hover bywa pudłem (focus, tooltip nie wyskoczył). Z quorum=1 jeden
# nieudany hover = grupa z 0-1 odczytami → cichy błąd/luka na cały stos. User (23.06):
# „muszą być przynajmniej 2 odczyty dymków, żeby scan przeszedł dalej".

# Minimalna liczba UDANYCH odczytów, by konsensus miał mniejszość (= CONSENSUS_MIN).
MIN_READS_FLOOR = CONSENSUS_MIN  # 2


class ReadVerdict(str, Enum):
    """Czy grupa ma dość udanych odczytów dymka, by „przejść dalej"."""

    READY = "ready"              # ≥ floor udanych → konsensus możliwy, przepuść
    SINGLETON = "singleton"      # grupa rozmiaru 1 → 1 odczyt to maksimum (brak dedupu), przepuść
    INSUFFICIENT = "insufficient"  # grupa ≥2, ale < floor udanych → NIE przepuszczaj, re-hover

    def __str__(self) -> str:
        return self.value


def min_reads_required(group_size: int, *, floor: int = MIN_READS_FLOOR) -> int:
    """Ile UDANYCH odczytów dymka grupa musi mieć, by przejść dalej.

    `min(group_size, floor)` — grupa ≥floor potrzebuje floor; singleton (size 1) tylko 1
    (więcej fizycznie nie ma czego czytać). Floor domyślnie 2 = minimum na konsensus.
    """

    if group_size <= 0:
        return 0
    return min(group_size, floor)


def reads_verdict(
    successful_reads: int, group_size: int, *, floor: int = MIN_READS_FLOOR
) -> ReadVerdict:
    """Werdykt bramki: czy `successful_reads` w grupie rozmiaru `group_size` wystarcza.

    Czyste, bez gry. `successful_reads` = ile hoverów grupy ZWRÓCIŁO odczyt (nie ile
    zaplanowano). Singleton z 1 odczytem = SINGLETON (przepuść — to maksimum). Grupa ≥2
    z <floor udanych = INSUFFICIENT (re-hover więcej reprezentantów, nie propaguj na stos).
    """

    if group_size <= 1:
        return ReadVerdict.SINGLETON if successful_reads >= 1 else ReadVerdict.INSUFFICIENT
    return (
        ReadVerdict.READY
        if successful_reads >= min_reads_required(group_size, floor=floor)
        else ReadVerdict.INSUFFICIENT
    )


def needs_more_reads(
    successful_reads: int, group_size: int, *, floor: int = MIN_READS_FLOOR
) -> bool:
    """Czy grupa wymaga DOHOVEROWANIA zanim zaakceptujesz jej konsensus (≥2 dla stosu)."""

    return reads_verdict(successful_reads, group_size, floor=floor) is ReadVerdict.INSUFFICIENT


# --------------------------------------------------------------------------- #
# Pomiar oszczędności na realnych manifestach (read-only, zero gry)            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class SavingsRow:
    """Wiersz tabeli oszczędności dla jednej polityki wyboru reprezentantów."""

    policy: str
    hovers: int             # ile slotów zhoverowano łącznie wg polityki
    baseline: int           # ile slotów hoverowano bez Phase B (wszystkie zhoverowane)
    saved_pct: float        # (baseline - hovers) / baseline
    groups_no_consensus: int  # grupy ≥2 z <2 reprezentantami (konsensus niemożliwy)
    exposed_slots: int      # sloty dziedziczące z grupy bez materiału na konsensus


def group_sizes_from_manifest(manifest: Mapping) -> list[int]:
    """Rozmiary grup ikon w jednym manifeście (tylko sloty FAKTYCZNIE zhoverowane).

    Slot liczony, jeśli ma niepuste `images` (czyli capture go zhoverował). Slot bez
    `icon_group` to osobna grupa rozmiaru 1 (singleton — brak dedupu). Slot z błędem
    capture (brak obrazu) nie wchodzi do baseline (nie był hoverowany).
    """

    slots = manifest.get("slots") or {}
    by_group: dict[int, int] = {}
    singletons = 0
    for s in slots.values():
        if not s.get("images"):
            continue  # nie zhoverowany (failed capture) — poza baseline
        ig = s.get("icon_group")
        if ig is None:
            singletons += 1
        else:
            by_group[ig] = by_group.get(ig, 0) + 1
    return list(by_group.values()) + [1] * singletons


def savings_table(
    size_counts: Mapping[int, int],
    policies: Sequence[tuple[str, dict]],
) -> list[SavingsRow]:
    """Policz oszczędność hoverów dla każdej polityki.

    `size_counts`: `rozmiar_grupy -> liczba_takich_grup` (zagregowane po manifestach).
    `policies`: lista `(nazwa, kwargs_dla_reps_for_size)`.
    `exposed_slots` = sloty pominięte w grupach, które nie osiągnęły materiału na
    konsensus (<2 reprezentantów) — dziedziczą z niepotwierdzonego odczytu (ryzyko
    cichego błędu ×1000 propagowanego na całą grupę).
    """

    baseline = sum(size * n for size, n in size_counts.items())
    rows: list[SavingsRow] = []
    for name, kw in policies:
        hovers = no_consensus = exposed = 0
        for size, n in size_counts.items():
            r = reps_for_size(size, **kw)
            hovers += r * n
            if size >= CONSENSUS_MIN and r < CONSENSUS_MIN:
                no_consensus += n
                exposed += (size - r) * n
        saved = (baseline - hovers) / baseline if baseline else 0.0
        rows.append(SavingsRow(name, hovers, baseline, saved, no_consensus, exposed))
    return rows


_POLICIES: list[tuple[str, dict]] = [
    ("plan: 1, 2 dla >=8", {"quorum": 1, "big_threshold": 8, "big_quorum": 2}),
    ("floor=2",            {"quorum": 2}),
    ("floor=3 (kontrakt)", {"quorum": 3}),
    ("floor=3, 4 dla >=8", {"quorum": 3, "big_threshold": 8, "big_quorum": 4}),
]


def _scan_size_counts(scans_dir: Path) -> tuple[Counter, int]:
    """Zagreguj rozmiary grup po wszystkich manifestach. Zwraca (Counter, #manifestów)."""

    counts: Counter = Counter()
    seen = 0
    for path in sorted(scans_dir.glob("*/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        sizes = group_sizes_from_manifest(data)
        if not sizes:
            continue
        seen += 1
        counts.update(sizes)
    return counts, seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scanner.analysis.representatives",
        description="Pomiar oszczędności Phase B (reprezentanci stosu) na manifestach.",
    )
    parser.add_argument("--scans", default="scans", help="katalog skanów (read-only)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    scans = Path(args.scans)
    if not scans.exists():
        print(f"Brak katalogu {scans!r}.")
        return 1
    counts, seen = _scan_size_counts(scans)
    total_slots = sum(size * n for size, n in counts.items())
    total_groups = sum(counts.values())
    multi = sum(n for size, n in counts.items() if size >= CONSENSUS_MIN)
    print(f"=== Phase B — pomiar na {seen} manifestach ===")
    print(f"Slotów zhoverowanych (baseline): {total_slots}")
    print(f"Grup ikon: {total_groups} (w tym {multi} stosów >=2)")
    print(f"Redundancja slot/grupa: {total_slots / total_groups:.2f}x" if total_groups else "")
    print()
    rows = savings_table(counts, _POLICIES)
    print(f"{'polityka':22} {'hovers':>7} {'oszczędn.':>10} "
          f"{'grup bez konsensu':>18} {'slotów odsłonięt.':>18}")
    for r in rows:
        print(f"{r.policy:22} {r.hovers:>7} {r.saved_pct:>9.1%} "
              f"{r.groups_no_consensus:>18} {r.exposed_slots:>18}")
    print("\n'odsłonięte sloty' = dziedziczą z grupy bez materiału na konsensus "
          "(ryzyko cichego błędu ×1000 na całą grupę).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
