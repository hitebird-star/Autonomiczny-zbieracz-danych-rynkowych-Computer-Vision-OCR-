"""Etap 2 Mapy Rynku: mapowanie w tle (Claude, offline, zero gry).

Obserwator READ-ONLY katalogu `scans/`: co interwał skanuje manifesty i ingestuje
nowe/zmienione do `ShopRegistry`. Mapa rośnie NA ŻYWO podczas skanowania, bez
dotykania capture, OCR ani gry — czyta tylko pliki, które potok i tak już zapisał.

To jest „skrypt mapujący w tle": odpalasz go obok skanera, a on dorzuca każdy
świeży manifest do trwałego rejestru sklepów (dedup po fingerprincie). Restart
bezpieczny — rejestr na dysku, watcher dożynkowuje resztę przy starcie.

Bezpieczeństwo: tylko `Path.glob` + `json.load` na gotowych plikach. Manifest w
trakcie zapisu (obcięty JSON) jest pomijany, nie wywraca pętli. Zero importów gry.

Uruchomienie (obok skanera):
    python -m scanner.analysis.map_watcher --scans scans --partition glevia_market
    Ctrl+C  -> ostatni zapis rejestru + podsumowanie
Jednorazowy backfill (zassij istniejące, bez pętli):
    python -m scanner.analysis.map_watcher --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scanner.analysis.shop_registry import ShopRegistry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Wynik jednego przejścia po `scans/`."""

    scanned: int       # manifestów obejrzanych w tym przejściu
    changed: int       # nowych lub zmodyfikowanych od ostatniego przejścia
    ingested: int      # miały fingerprint => upsert do rejestru
    shops_before: int
    shops_after: int

    @property
    def new_shops(self) -> int:
        return self.shops_after - self.shops_before


class MapWatcher:
    """Pilnuje `scans/`, ingestuje świeże manifesty do rejestru. Read-only wobec gry.

    Śledzi `(scan_id -> mtime)`: re-ingestuje plik dopiero gdy się zmienił (manifest
    dojrzewa — fingerprint/seller pojawiają się późno w cyklu skanu). `ingest` jest
    idempotentny (upsert+dedup), więc ponowne zassanie jest zawsze bezpieczne.
    """

    def __init__(self, registry: ShopRegistry, scans_dir: str | Path):
        self.registry = registry
        self.scans_dir = Path(scans_dir)
        self._seen: dict[str, float] = {}

    def _manifest_paths(self) -> list[Path]:
        if not self.scans_dir.exists():
            return []
        return sorted(self.scans_dir.glob("*/manifest.json"))

    def sync_once(self, *, save: bool = True) -> SyncResult:
        """Jedno przejście: ingest nowych/zmienionych manifestów. Zapisuje gdy coś doszło."""

        scanned = changed = ingested = 0
        before = len(self.registry)
        for path in self._manifest_paths():
            scanned += 1
            scan_id = path.parent.name
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if self._seen.get(scan_id) == mtime:
                continue  # bez zmian od ostatniego razu — pomiń (optymalizacja)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue  # manifest w trakcie zapisu / obcięty — pominie się i wróci
            self._seen[scan_id] = mtime
            changed += 1
            if self.registry.ingest(data) is not None:
                ingested += 1
        after = len(self.registry)
        if save and changed:
            self.registry.save()
        return SyncResult(scanned, changed, ingested, before, after)

    def watch(
        self,
        *,
        interval: float = 5.0,
        max_passes: int | None = None,
        on_pass=None,
    ) -> int:
        """Pętla: co `interval` sekund `sync_once`. Ctrl+C kończy z ostatnim zapisem.

        `max_passes` (do testów) ucina pętlę po N przejściach. `on_pass(result)` to
        opcjonalny callback (np. dla testu/żywego podglądu) wołany po każdym przejściu.
        """

        passes = 0
        try:
            while True:
                result = self.sync_once()
                passes += 1
                if on_pass is not None:
                    on_pass(result)
                else:
                    self._print_status(result)
                if max_passes is not None and passes >= max_passes:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n=== STOP — ostatni zapis rejestru ===")
            self.registry.save()
        return len(self.registry)

    def _print_status(self, result: SyncResult) -> None:
        located = len(self.registry.located())
        line = (
            f"\r[{_now_iso()[11:19]}] sklepow {len(self.registry):4d} "
            f"(+{result.new_shops}) | zlokal. {located:4d} | "
            f"zmian {result.changed:3d}/{result.scanned:4d} | "
            f"-> {self.registry.path}    "
        )
        print(line, end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scanner.analysis.map_watcher",
        description="Mapowanie w tle: ingest manifestow scans/ do rejestru (read-only).",
    )
    parser.add_argument("--scans", default="scans", help="katalog skanow do obserwacji")
    parser.add_argument("--out", default="market_map", help="korzeń partycji rejestru")
    parser.add_argument("--partition", default="glevia_market",
                        help="partycja rejestru (domyślnie zmierzony rynek glevia_market)")
    parser.add_argument("--interval", type=float, default=5.0, help="sekundy między przejściami")
    parser.add_argument("--once", action="store_true",
                        help="jednorazowy backfill (bez pętli)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    registry = ShopRegistry.open(args.out, partition=args.partition)
    watcher = MapWatcher(registry, args.scans)
    print(f"=== MAP WATCHER (read-only) — scans {args.scans!r} -> {registry.path} ===")
    print(f"Rejestr startowy: {len(registry)} sklepów.")

    if args.once:
        result = watcher.sync_once()
        print(f"Backfill: obejrzano {result.scanned}, zmian {result.changed}, "
              f"+{result.new_shops} sklepów -> razem {result.shops_after} "
              f"({len(registry.located())} ze współrzędną).")
        return 0

    print("Pętla startuje (Ctrl+C kończy). Skaner może pracować równolegle.\n")
    watcher.watch(interval=args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
