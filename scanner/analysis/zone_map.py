"""Etap 3 Mapy Rynku: strefy + pokrycie + nasycenie (Claude, offline, zero gry).

Strefowanie ZASTĘPUJE ślepy wężyk i z definicji likwiduje `route_exhausted`
(stała trasa kończyła się po ~20-30 sklepach, bo `--max-shops` to tylko sufit).
Bot nie ma budżetu kroków — chodzi po strefie, AŻ przestaje znajdować nowe sklepy.

Strefa = DONE gdy OBA sygnały:
  1. POKRYCIE — odwiedzone wszystkie pod-komórki boxu strefy (grube binowanie x,y),
  2. NASYCENIE — ostatnie K otwarć w strefie dało 0 nowych fingerprintów.
Sama „odwiedzona" nie wystarcza (można przejść środkiem, ominąć rogi).

Ten moduł to CZYSTA logika danych + geometrii: kafelkowanie koperty, binowanie
pokrycia, licznik nasycenia, wybór „najbliższa strefa ≠ DONE", trwałość zones.json.
Planer ruchu live (Etap 5b, DeepSeek) WOŁA te struktury — sam tu nie wchodzi.

Parametryzacja (nie przesądza decyzji DeepSeeka): siatka `cols×rows` i próg `K`
to argumenty; domyślne to rekomendacja z planu (3×3, K=8), live-tunable.

ODŁOŻONE z Etapu 3 (wymaga danych ze Stage 4): kalibracja „jednostek świata na
0.6s kroku" i dead-reckoning wypełniający miss OCR — potrzebuje PAR (sekwencja
OCR ↔ log ruchów), których dziś nie ma (game_position=None). Wejdzie, gdy DeepSeek
zacznie stemplować współrzędną. Strefy/pokrycie/nasycenie działają już teraz.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ZONES_FILENAME = "zones.json"
# Domyślne z MARKET_MAP_PLAN (rekomendacja startowa, live-tunable przez DeepSeeka).
DEFAULT_GRID = (3, 3)            # kolumny × wiersze stref na kopercie
DEFAULT_SUBGRID = (3, 3)        # pod-komórki pokrycia w obrębie strefy
DEFAULT_SATURATION_K = 8        # otwarć bez nowego fingerprintu => nasycona


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class Zone:
    """Geometria jednej strefy (komórki stałego rozmiaru na kopercie)."""

    zone_id: str
    col: int
    row: int
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (świat)

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def contains(self, x: float, y: float) -> bool:
        """Punkt w boxie (low inclusive, high inclusive — do zapytań/podglądu).

        Do JEDNOZNACZNEGO przypisania punktu używaj `ZoneMap.zone_for` (indeks
        kafla), nie tego — bo sąsiednie boxy dzielą krawędź.
        """

        x0, y0, x1, y1 = self.box
        return x0 <= x <= x1 and y0 <= y <= y1


@dataclass(slots=True)
class ZoneState:
    """Zmienny stan strefy: pokrycie pod-komórek + licznik nasycenia + latch DONE."""

    covered_cells: set[tuple[int, int]] = field(default_factory=set)
    saturation_counter: int = 0
    opens: int = 0
    last_new_shop_ts: str | None = None
    done: bool = False  # latch — raz DONE zostaje DONE (oznaczone trwale)


def build_zones(
    envelope: tuple[int, int, int, int],
    grid: tuple[int, int] = DEFAULT_GRID,
) -> list[Zone]:
    """Pokafelkuj kopertę `(x0,y0,x1,y1)` na `cols×rows` stref równego rozmiaru.

    zone_id = `Z1..Zn` w kolejności wierszami (Z1 lewy-górny). Granice kafli
    liczone z float i zaokrąglane, więc pokrywają całą kopertę bez dziur.
    """

    x0, y0, x1, y1 = envelope
    if x1 < x0 or y1 < y0:
        raise ValueError("koperta: x1>=x0 i y1>=y0 wymagane")
    cols, rows = grid
    if cols < 1 or rows < 1:
        raise ValueError("siatka musi być >= 1x1")

    cw = (x1 - x0) / cols
    ch = (y1 - y0) / rows
    zones: list[Zone] = []
    n = 0
    for r in range(rows):
        for c in range(cols):
            n += 1
            bx0 = round(x0 + c * cw)
            bx1 = round(x0 + (c + 1) * cw)
            by0 = round(y0 + r * ch)
            by1 = round(y0 + (r + 1) * ch)
            zones.append(Zone(zone_id=f"Z{n}", col=c, row=r, box=(bx0, by0, bx1, by1)))
    return zones


class ZoneMap:
    """Strefy koperty + ich stan. Pokrycie, nasycenie, wybór następnej strefy.

    Jeden ZoneMap = jedna partycja (mapa+kanał), współdzieli folder z rejestrem.
    """

    def __init__(
        self,
        envelope: tuple[int, int, int, int],
        *,
        grid: tuple[int, int] = DEFAULT_GRID,
        subgrid: tuple[int, int] = DEFAULT_SUBGRID,
        saturation_k: int = DEFAULT_SATURATION_K,
        directory: str | Path | None = None,
    ):
        self.envelope = tuple(envelope)  # type: ignore[assignment]
        self.grid = grid
        self.subgrid = subgrid
        self.saturation_k = saturation_k
        self.directory = Path(directory) if directory is not None else None
        self.zones: list[Zone] = build_zones(self.envelope, grid)
        self._state: dict[str, ZoneState] = {z.zone_id: ZoneState() for z in self.zones}
        self._by_id: dict[str, Zone] = {z.zone_id: z for z in self.zones}

    # --- lokalizacja punktu ----------------------------------------------

    def zone_for(self, x: float, y: float) -> Zone | None:
        """Strefa zawierająca punkt (indeks kafla, jednoznacznie). None poza kopertą."""

        x0, y0, x1, y1 = self.envelope
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        cols, rows = self.grid
        cw = (x1 - x0) / cols
        ch = (y1 - y0) / rows
        c = min(int((x - x0) / cw), cols - 1) if cw else 0
        r = min(int((y - y0) / ch), rows - 1) if ch else 0
        return self._by_id.get(f"Z{r * cols + c + 1}")

    def _subcell(self, zone: Zone, x: float, y: float) -> tuple[int, int]:
        x0, y0, x1, y1 = zone.box
        sc, sr = self.subgrid
        sw = (x1 - x0) / sc if x1 > x0 else 1
        sh = (y1 - y0) / sr if y1 > y0 else 1
        ci = min(int((x - x0) / sw), sc - 1) if sw else 0
        ri = min(int((y - y0) / sh), sr - 1) if sh else 0
        return (max(ci, 0), max(ri, 0))

    # --- rejestrowanie obserwacji ----------------------------------------

    def record_position(self, x: float, y: float) -> bool:
        """Zaznacz pod-komórkę pokrycia dla strefy zawierającej punkt.

        Zwraca True jeśli to NOWO pokryta pod-komórka (pokrycie wzrosło).
        """

        zone = self.zone_for(x, y)
        if zone is None:
            return False
        cell = self._subcell(zone, x, y)
        state = self._state[zone.zone_id]
        if cell in state.covered_cells:
            return False
        state.covered_cells.add(cell)
        return True

    def record_open(
        self,
        zone_id: str,
        *,
        is_new_fingerprint: bool,
        ts: str | None = None,
    ) -> ZoneState:
        """Zarejestruj otwarcie sklepu w strefie. Nowy fingerprint zeruje licznik
        nasycenia (i znaczy `last_new_shop_ts`); powtórka go inkrementuje."""

        state = self._state[zone_id]
        state.opens += 1
        if is_new_fingerprint:
            state.saturation_counter = 0
            state.last_new_shop_ts = ts or _now_iso()
        else:
            state.saturation_counter += 1
        return state

    # --- sygnały DONE -----------------------------------------------------

    def coverage_complete(self, zone_id: str) -> bool:
        """Czy odwiedzono wszystkie pod-komórki strefy (sygnał 1)."""

        sc, sr = self.subgrid
        return len(self._state[zone_id].covered_cells) >= sc * sr

    def saturated(self, zone_id: str) -> bool:
        """Czy ostatnie K otwarć bez nowego fingerprintu (sygnał 2)."""

        return self._state[zone_id].saturation_counter >= self.saturation_k

    def is_done(self, zone_id: str) -> bool:
        """Strefa DONE gdy OBA sygnały (lub już zalatchowana jako DONE)."""

        state = self._state[zone_id]
        if state.done:
            return True
        if self.coverage_complete(zone_id) and self.saturated(zone_id):
            state.done = True  # latch — oznacz trwale
        return state.done

    def state_of(self, zone_id: str) -> str:
        """PENDING (nietknięta) | ACTIVE (w toku) | DONE."""

        if self.is_done(zone_id):
            return "DONE"
        state = self._state[zone_id]
        if state.covered_cells or state.opens:
            return "ACTIVE"
        return "PENDING"

    def get_state(self, zone_id: str) -> ZoneState:
        return self._state[zone_id]

    # --- wybór następnej strefy ------------------------------------------

    def next_zone(self, x: float, y: float) -> Zone | None:
        """Najbliższa (centroid, Euklides²) strefa ≠ DONE od pozycji. None gdy
        wszystkie DONE (cała koperta wysycona — koniec mapowania)."""

        best: Zone | None = None
        best_key: tuple[float, int] | None = None
        for zone in self.zones:
            if self.is_done(zone.zone_id):
                continue
            cx, cy = zone.centroid
            dist = (cx - x) ** 2 + (cy - y) ** 2
            order = int(zone.zone_id[1:])
            key = (dist, order)  # remis => niższy numer strefy (determinizm)
            if best_key is None or key < best_key:
                best, best_key = zone, key
        return best

    def remaining(self) -> list[Zone]:
        """Strefy jeszcze nie-DONE (do zrobienia)."""

        return [z for z in self.zones if not self.is_done(z.zone_id)]

    def progress(self) -> dict[str, int]:
        """Podsumowanie: ile DONE / ACTIVE / PENDING."""

        counts = {"DONE": 0, "ACTIVE": 0, "PENDING": 0}
        for zone in self.zones:
            counts[self.state_of(zone.zone_id)] += 1
        return counts

    # --- trwałość ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope": list(self.envelope),
            "grid": list(self.grid),
            "subgrid": list(self.subgrid),
            "saturation_k": self.saturation_k,
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "col": z.col,
                    "row": z.row,
                    "box": list(z.box),
                    "state": self.state_of(z.zone_id),
                    "coverage_cells": sorted(list(c) for c in self._state[z.zone_id].covered_cells),
                    "saturation_counter": self._state[z.zone_id].saturation_counter,
                    "opens": self._state[z.zone_id].opens,
                    "last_new_shop_ts": self._state[z.zone_id].last_new_shop_ts,
                    "done": self._state[z.zone_id].done,
                }
                for z in self.zones
            ],
            "updated_at": _now_iso(),
        }

    def save(self) -> Path:
        if self.directory is None:
            raise ValueError("brak directory — ustaw przy konstrukcji albo użyj save_to")
        return self.save_to(self.directory)

    def save_to(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ZONES_FILENAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)  # atomowa podmiana
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "ZoneMap":
        directory = Path(directory)
        data = json.loads((directory / ZONES_FILENAME).read_text(encoding="utf-8"))
        zmap = cls(
            tuple(data["envelope"]),
            grid=tuple(data["grid"]),
            subgrid=tuple(data["subgrid"]),
            saturation_k=int(data["saturation_k"]),
            directory=directory,
        )
        for zd in data.get("zones", []):
            state = zmap._state[zd["zone_id"]]
            state.covered_cells = {tuple(c) for c in zd.get("coverage_cells", [])}
            state.saturation_counter = int(zd.get("saturation_counter", 0))
            state.opens = int(zd.get("opens", 0))
            state.last_new_shop_ts = zd.get("last_new_shop_ts")
            state.done = bool(zd.get("done", False))
        return zmap
