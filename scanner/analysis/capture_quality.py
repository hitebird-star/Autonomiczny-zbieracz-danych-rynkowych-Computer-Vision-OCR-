"""Offline pomiar jakości capture (działka Claude, czysty I/O — bez gry, bez AI).

Liczy, ile ZAJĘTYCH slotów faktycznie dało klatki dymka. To metryka, której
brakowało: pokazuje czy najazd kursora trafia dymek i — co ważniejsze — czy
pudła mają wzorzec PRZESTRZENNY (ta sama komórka pudłuje zawsze = geometria)
czy są LOSOWE (ta sama komórka raz trafia, raz nie = zawodny input).

Hit  = obserwacja slotu ma ≥1 zapisaną klatkę (``images``).
Miss = zajęty slot z 0 klatek.

Capture quality jest niezależne od wyniku analizy: liczy się sam fakt złapania
obrazu, nawet jeśli VLM/walidator potem go odrzuci.

Uruchomienie:
    python -m scanner.analysis.capture_quality [scans_dir] [--scan ID]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scanner.models import ShopScan
from scanner.storage import ScanRepository


@dataclass(frozen=True, slots=True)
class ScanCaptureStats:
    """Statystyka capture jednego skanu."""

    scan_id: str
    seller: str
    occupied: int
    hit_cells: tuple[tuple[int, int], ...]   # (column, row)
    miss_cells: tuple[tuple[int, int], ...]

    @property
    def hits(self) -> int:
        return len(self.hit_cells)

    @property
    def misses(self) -> int:
        return len(self.miss_cells)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.occupied if self.occupied else 0.0


def scan_stats(scan: ShopScan) -> ScanCaptureStats:
    """Policz hit/miss per komórka dla jednego skanu."""

    hits: list[tuple[int, int]] = []
    misses: list[tuple[int, int]] = []
    for observation in scan.slots.values():
        cell = (observation.column, observation.row)
        if observation.images:
            hits.append(cell)
        else:
            misses.append(cell)
    return ScanCaptureStats(
        scan_id=scan.scan_id,
        seller=scan.seller,
        occupied=len(scan.slots),
        hit_cells=tuple(sorted(hits)),
        miss_cells=tuple(sorted(misses)),
    )


def iter_stats(root: str | Path = "scans") -> list[ScanCaptureStats]:
    repository = ScanRepository(root)
    return [scan_stats(scan) for scan in repository.iter_scans() if scan.slots]


def cell_reliability(
    stats: list[ScanCaptureStats],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Zwróć {(column, row): (hits, appearances)} po wszystkich skanach."""

    table: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for entry in stats:
        for cell in entry.hit_cells:
            table[cell][0] += 1
            table[cell][1] += 1
        for cell in entry.miss_cells:
            table[cell][1] += 1
    return {cell: (hits, total) for cell, (hits, total) in table.items()}


def flipping_cells(
    reliability: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    """Komórki, które bywały i trafione, i pudłowane (0 < hits < appearances).

    Każda taka komórka to dowód NIEDETERMINIZMU: w jednym przebiegu dymek się
    pojawił, w innym nie — przy identycznej geometrii. To odróżnia zawodny
    input od błędu kalibracji (komórka pudłująca zawsze).
    """

    return sorted(
        cell
        for cell, (hits, total) in reliability.items()
        if total >= 2 and 0 < hits < total
    )


def _grid_map(reliability: dict[tuple[int, int], tuple[int, int]]) -> str:
    """Mapa 10×10: '#' zawsze trafia, 'x' zawsze pudłuje, '~' miesza, '.' brak."""

    rows = []
    for row in range(10):
        cells = []
        for column in range(10):
            stat = reliability.get((column, row))
            if stat is None:
                cells.append(".")
                continue
            hits, total = stat
            if hits == total:
                cells.append("#")
            elif hits == 0:
                cells.append("x")
            else:
                cells.append("~")
        rows.append(" ".join(cells))
    legend = "  legenda: # zawsze hit | x zawsze miss | ~ raz tak raz nie | . pusto"
    header = "      " + " ".join(f"{c}" for c in range(10))
    body = "\n".join(f"  r{row}  {line}" for row, line in enumerate(rows))
    return f"{header}\n{body}\n{legend}"


def format_report(stats: list[ScanCaptureStats]) -> str:
    if not stats:
        return "Brak skanów z zajętymi slotami w podanym katalogu."

    lines: list[str] = []
    lines.append("=== JAKOŚĆ CAPTURE (offline) ===\n")
    lines.append(
        f"{'scan_id':<24} {'occ':>4} {'hit':>4} {'miss':>5} {'hit%':>6}  seller"
    )
    total_occ = total_hit = 0
    rates: list[float] = []
    for entry in sorted(stats, key=lambda s: s.scan_id):
        total_occ += entry.occupied
        total_hit += entry.hits
        rates.append(entry.hit_rate)
        lines.append(
            f"{entry.scan_id:<24} {entry.occupied:>4} {entry.hits:>4} "
            f"{entry.misses:>5} {entry.hit_rate * 100:>5.0f}%  {entry.seller}"
        )

    overall = total_hit / total_occ if total_occ else 0.0
    lines.append("")
    lines.append(
        f"RAZEM: sloty={total_occ} hit={total_hit} miss={total_occ - total_hit} "
        f"hit_rate={overall * 100:.0f}%"
    )
    if rates:
        lines.append(
            f"ROZRZUT hit% między przebiegami: min={min(rates) * 100:.0f}% "
            f"max={max(rates) * 100:.0f}%  (duży rozrzut = zawodny input, nie geometria)"
        )

    reliability = cell_reliability(stats)
    flipping = flipping_cells(reliability)
    chronic = sorted(
        cell
        for cell, (hits, total) in reliability.items()
        if total >= 2 and hits == 0
    )
    lines.append("")
    lines.append(_grid_map(reliability))
    lines.append("")
    lines.append(
        f"KOMÓRKI MIESZAJĄCE (~, dowód niedeterminizmu): {len(flipping)}"
    )
    if flipping:
        lines.append("  " + ", ".join(f"({c},{r})" for c, r in flipping))
    lines.append(
        f"KOMÓRKI CHRONICZNIE PUDŁUJĄCE (x w >=2 przebiegach, podejrzenie geometrii): "
        f"{len(chronic)}"
    )
    if chronic:
        lines.append("  " + ", ".join(f"({c},{r})" for c, r in chronic))

    lines.append("")
    if flipping and not chronic:
        lines.append(
            "WERDYKT: pudła są LOSOWE (komórki mieszają, brak chronicznych) "
            "-> problem to zawodny input/hover, nie kalibracja siatki."
        )
    elif chronic and not flipping:
        lines.append(
            "WERDYKT: pudła są PRZESTRZENNE (te same komórki zawsze) "
            "-> podejrzenie błędu geometrii/kalibracji."
        )
    elif flipping and chronic:
        lines.append(
            "WERDYKT: mieszane — większość losowa (input), ale sprawdź komórki 'x'."
        )
    else:
        lines.append("WERDYKT: zbyt mało powtórzeń tych samych komórek do rozstrzygnięcia.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scanner.analysis.capture_quality",
        description="Offline pomiar jakości capture (hit-rate dymków + mapa hit/miss).",
    )
    parser.add_argument("scans", nargs="?", default="scans", help="katalog ze skanami")
    parser.add_argument("--scan", help="ogranicz raport do jednego scan_id")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    stats = iter_stats(args.scans)
    if args.scan:
        stats = [entry for entry in stats if entry.scan_id == args.scan]
    print(format_report(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
