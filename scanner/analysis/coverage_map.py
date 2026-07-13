"""Mapa pokrycia farmy (Claude, offline-pure): które komórki odwiedzone, dokąd dalej.

**Po co (Filar 2).** Filar B (180° = negacja wektora) wyciąga z rogu, ALE retrasuje TĘ SAMĄ
przekątną tam-i-z-powrotem → re-pasuje te same sklepy, zero offsetu prostopadłego, brak pokrycia
2D ([[mapping-rebuild-odometry]] Filar 2, `APPROACH_A_SPEC.md`). Tu jest brakująca KSIĘGOWOŚĆ
pokrycia: siatka komórek nad kopertą, znacz komórkę odwiedzoną z pozycji odometrii, wskaż
NAJBLIŻSZĄ niepokrytą jako cel — DeepSeek do niej jedzie (sterowanie: rotuj-zmierz-koryguj,
`heading_from_ocr`, BEZ kalibracji obrotu — patrz spec). Gdy wszystkie pokryte → świadomy koniec.

Czysta logika — zero gry/OCR/obrazów, testowalna offline. Pozycja w jednostkach świata (z
odometrii), NIE w pikselach ekranu. Koperta w konwencji Claude `(x_min, x_max, y_min, y_max)`
(== `nav_guards.within_envelope`), NIE w divergentnej DeepSeeka. Por. `nav_guards.py` (granice),
`odometry.py` (pozycja+heading_from_ocr), `dense_stamp.py` (stempel na komórkę).
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# Bok komórki w jednostkach świata = ŚLAD PIERŚCIENIA KLIKALNEGO (jeden postój). Pomiar biegu
# 230719: w kadrze ~260 sklepów, klikalnych w pierścieniu 90–240px ~17, cap=18 (G1). Komórka ma
# odpowiadać temu pierścieniowi: sąsiednie komórki = sąsiednie postoje kafelkujące pole bez dziur.
# cell_size = (promień pierścienia w jednostkach świata) — KALIBRACJA G5 (DeepSeek mierzy świat/piksel).
# Saturacja (G3, until="done") trzyma bota w komórce aż „pierścień daje same znane", nie po 1 skanie.
DEFAULT_CELL_SIZE = 30.0

Point = tuple[float, float]
Cell = tuple[int, int]
Polygon = list[Point]
Envelope = tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max) — konwencja Claude


def _point_in_polygon(pt: Point, poly: Polygon) -> bool:
    """Ray casting (kopia z farm_boundary — coverage_map zostaje samodzielny, zero importów)."""

    if len(poly) < 3:
        return False
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _dist_point_to_segment(p: Point, a: Point, b: Point) -> float:
    """Najmniejsza odległość punktu od odcinka a–b (do dystansu od krawędzi wielokąta)."""

    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dist_to_polygon_edge(pt: Point, poly: Polygon) -> float:
    """Odległość punktu od NAJBLIŻSZEJ krawędzi wielokąta (obwód, nie wnętrze)."""

    n = len(poly)
    return min(_dist_point_to_segment(pt, poly[i], poly[(i + 1) % n]) for i in range(n))


@dataclass
class CoverageMap:
    """Siatka pokrycia nad kopertą farmy. Stan = zbiór odwiedzonych komórek.

    `mark(pos)` znaczy komórkę zawierającą pozycję (opcjonalnie sąsiedztwo w promieniu).
    `next_target(pos)` wskazuje środek najbliższej niepokrytej komórki (dokąd jechać).
    """

    envelope: Envelope
    cell_size: float = DEFAULT_CELL_SIZE
    boundary: Polygon | None = None   # „linia akceptowalności": komórki poza wielokątem WYKLUCZONE
    boundary_margin: float = 0.0      # >0: pas `margin` od krawędzi też NIE-cel (router=guard ruchu)
    _covered: set[Cell] = field(default_factory=set)
    _scans: dict[Cell, int] = field(default_factory=dict)   # NOWE sklepy zeskanowane / komórkę
    _dups: dict[Cell, int] = field(default_factory=dict)    # duplikaty trafione / komórkę (re-skan, suma)
    _dup_streak: dict[Cell, int] = field(default_factory=dict)  # G3: KOLEJNE duplikaty od ostatniego nowego (sygnał „pierścień daje same znane")
    _blocks: dict[Cell, dict[str, int]] = field(default_factory=dict)  # C2: zdarzenia blokad / komórkę (heatmapa porażek)
    _no_go: set[Cell] = field(default_factory=set)          # C3: komórki wykluczone z danych (nie tylko z granicy)
    _skip_target: set[Cell] = field(default_factory=set)    # nieosiągalne w TYM biegu: NIE-cel, ale PRZEJEZDNE w BFS (zero fragmentacji)

    def __post_init__(self) -> None:
        x_min, x_max, y_min, y_max = self.envelope
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("koperta musi mieć dodatni rozmiar (x_min<x_max, y_min<y_max)")
        if self.cell_size <= 0:
            raise ValueError("cell_size musi być > 0")
        self._ncols = max(1, math.ceil((x_max - x_min) / self.cell_size))
        self._nrows = max(1, math.ceil((y_max - y_min) / self.cell_size))
        # Komórki poza granicą = nie-farma → nigdy nie są celem (linia akceptowalności
        # KSZTAŁTUJE pokrycie, nie tylko prostokąt). Zabezpieczenie: gdyby wielokąt wykluczył
        # WSZYSTKO (zła/pusta granica), ignoruj go — pełna siatka, nie zerowa farma.
        self._excluded: set[Cell] = set()
        if self.boundary is not None and len(self.boundary) >= 3:
            outside = {c for c in self.all_cells()
                       if not _point_in_polygon(self.cell_center(c), self.boundary)}
            excluded = set(outside)
            # Pas przygraniczny (margin) = NIE-cel: guard ruchu i tak nie wpuści tu bota
            # (should_turn_at_boundary), a sklepy z krawędzi łapie pierścień ze standu o rząd
            # niżej. Wykluczamy komórkę przygraniczną TYLKO gdy ma GŁĘBSZEGO sąsiada (jej teren
            # pokryty z tego standu) — inaczej zostaje celem (zero osierocenia, zero zapadliska).
            if self.boundary_margin > 0:
                interior = {c for c in self.all_cells() if c not in outside}
                near_edge = {c for c in interior
                             if _dist_to_polygon_edge(self.cell_center(c), self.boundary)
                             < self.boundary_margin}
                deep = interior - near_edge
                droppable = {c for c in near_edge
                             if any(n in deep for n in self._neighbors(c))}
                excluded |= droppable
            if len(excluded) < self._ncols * self._nrows:
                self._excluded = excluded

    # --- geometria komórek --------------------------------------------------- #

    def cell_of(self, pos: Point) -> Cell:
        """Komórka zawierająca `pos` (klamrowana do siatki, by pozycja spoza koperty nie wyszła poza)."""

        x_min, _, y_min, _ = self.envelope
        cx = int((pos[0] - x_min) // self.cell_size)
        cy = int((pos[1] - y_min) // self.cell_size)
        cx = min(max(cx, 0), self._ncols - 1)
        cy = min(max(cy, 0), self._nrows - 1)
        return (cx, cy)

    def cell_center(self, cell: Cell) -> Point:
        """Środek komórki w jednostkach świata (klamrowany do wnętrza koperty)."""

        x_min, x_max, y_min, y_max = self.envelope
        x = x_min + (cell[0] + 0.5) * self.cell_size
        y = y_min + (cell[1] + 0.5) * self.cell_size
        return (min(x, x_max), min(y, y_max))

    def all_cells(self) -> list[Cell]:
        return [(cx, cy) for cy in range(self._nrows) for cx in range(self._ncols)]

    @property
    def total_cells(self) -> int:
        return self._ncols * self._nrows

    # --- stan pokrycia ------------------------------------------------------- #

    def mark(self, pos: Point, *, radius_cells: int = 0) -> None:
        """Znacz komórkę `pos` jako odwiedzoną. `radius_cells>0` znaczy też sąsiedztwo
        (skan widzi okolicę — ale konserwatywnie domyślnie 0 = tylko komórka pozycji)."""

        cx, cy = self.cell_of(pos)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self._ncols and 0 <= ny < self._nrows:
                    self._covered.add((nx, ny))

    def is_covered(self, pos: Point) -> bool:
        """Czy komórka zawierająca POZYCJĘ świata jest pokryta. (Komórkę wprost → `is_cell_covered`.)"""

        return self.cell_of(pos) in self._covered

    def is_cell_covered(self, cell: Cell) -> bool:
        return cell in self._covered

    def _skip(self) -> set[Cell]:
        """Komórki NIE-cele: poza granicą (`_excluded`), wyuczone puste (`_no_go`, C3),
        albo nieosiągalne w tym biegu (`_skip_target`). `is_blocked` (BFS) ich NIE używa —
        `_skip_target` jest NIE-celem, ale PRZEJEZDNYM, więc nie rozcina farmy."""

        return self._excluded | self._no_go | self._skip_target

    def uncovered_cells(self) -> list[Cell]:
        cov = self._covered
        skip = self._skip()
        return [c for c in self.all_cells() if c not in cov and c not in skip]

    def is_excluded(self, cell: Cell) -> bool:
        """Czy komórka NIE jest celem (poza granicą lub no_go)."""

        return cell in self._excluded or cell in self._no_go

    @property
    def covered_count(self) -> int:
        return len(self._covered)

    @property
    def farm_cells(self) -> int:
        """Komórki NALEŻĄCE do farmy (siatka minus wykluczone granicą i no_go). Mianownik pokrycia."""

        return self.total_cells - len(self._skip())

    def coverage_fraction(self) -> float:
        # Licznik = pokryte komórki farmy; mianownik = komórki farmy (nie cała siatka).
        denom = self.farm_cells
        if denom <= 0:
            return 0.0
        covered_farm = len(self._covered - self._skip())
        return covered_farm / denom

    def is_complete(self) -> bool:
        return not self.uncovered_cells()

    # --- anty-reskan: liczniki skanów/duplikatów per komórka (Faza 2) -------- #
    # „Te same sklepy" = powrót do komórki, której sklepy już zeskanowano (tożsamość
    # znana dopiero PO otwarciu → klaster ~4/coord). Sygnał wyczerpania = seria
    # DUPLIKATÓW w komórce (same znane fingerprinty). Czyste liczniki; decyzję
    # „klikać czy iść dalej" podejmuje `should_skip_click`. Por. [[shop-identity-two-level]].

    def record_scan(self, pos: Point, *, duplicate: bool) -> None:
        """Zarejestruj WYNIK otwarcia sklepu w komórce pozycji: nowy albo duplikat.

        Skanowanie też pokrywa komórkę (dodaje do `_covered`). `duplicate=True` =
        trafiono już-znany fingerprint (re-skan). NOWY sklep ZERUJE serię duplikatów
        (`_dup_streak`) — komórka wciąż produktywna; duplikat serię zwiększa (G3).
        """

        cell = self.cell_of(pos)
        if duplicate:
            self._dups[cell] = self._dups.get(cell, 0) + 1
            self._dup_streak[cell] = self._dup_streak.get(cell, 0) + 1
        else:
            self._scans[cell] = self._scans.get(cell, 0) + 1
            self._dup_streak[cell] = 0   # nowy sklep = komórka nie jest jeszcze przebrana
        self._covered.add(cell)

    def record_known_fresh(self, pos: Point) -> None:
        """Sklep POMINIĘTY bo znany-i-świeży (durable dedup cross-bieg, `duplicate_known_fresh`).

        Liczy się do saturacji TAK SAMO jak duplikat w-biegu (G2) — inaczej komórka pełna
        znanych sklepów nigdy nie osiągnie `is_done` i bot zapętla się ją „kończąc". DeepSeek:
        wołaj w gałęzi known_fresh zamiast pomijać księgowanie. Por. [[shop-identity-two-level]].
        """

        self.record_scan(pos, duplicate=True)

    # --- C2/C3: logi blokad NA MAPIE → zabezpieczenia borderowe (COVERAGE_MAP_CORE §5) ---- #
    # Każda blokada stempluje komórkę (heatmapa porażek). `border_adjustments` wyprowadza z niej
    # kandydatów `no_go` (rejony, gdzie bot wciąż zawodzi i NIC nie znajduje = puste/nieosiągalne).
    # Granica/`no_go` douczają się z realnych biegów. DeepSeek (D2): record_block w punktach blokad;
    # (D5): po biegu `mark_no_go(c) for c in border_adjustments()`.

    BLOCK_KINDS = ("failed_open", "stall", "edge_hit", "goto_fail", "boundary_turn", "duplicate")

    def record_block(self, pos: Point, kind: str) -> None:
        """Zarejestruj zdarzenie blokady w komórce pozycji (NIE pokrywa komórki — to porażka)."""

        cell = self.cell_of(pos)
        self._blocks.setdefault(cell, {})
        self._blocks[cell][kind] = self._blocks[cell].get(kind, 0) + 1

    def blocks_in_cell(self, cell: Cell, kind: str | None = None) -> int:
        """Liczba blokad w komórce — danego rodzaju (`kind`) albo łącznie (`kind=None`)."""

        kinds = self._blocks.get(cell)
        if not kinds:
            return 0
        return kinds.get(kind, 0) if kind is not None else sum(kinds.values())

    def mark_no_go(self, cell: Cell) -> None:
        """Trwale wyklucz komórkę z celów (rejon pusty/nieosiągalny — z danych, nie z obrysu)."""

        self._no_go.add(cell)

    def mark_unreachable(self, cell: Cell) -> None:
        """Komórka nieosiągalna w TYM biegu: przestaje być celem (is_done), ale ZOSTAJE
        przejezdna w BFS (`is_blocked` jej nie widzi) → router jej nie powtarza, a graf
        się nie rozpada. Anti „pętla w i s": cel za granicą oznaczamy tu, nie przez `no_go`.
        NIE persystuje — reset co bieg (osiągalność zależy od pozycji startu)."""

        self._skip_target.add(cell)

    def clear_no_go(self) -> int:
        """Usuń ręcznie wyuczone blokady, zachowując pokrycie i historię skanów.

        To operacja naprawcza dla map zapisanych przez dawną regułę ``goto_fail``.
        Nie usuwa ``_excluded`` wyliczanych z granicy farmy.
        """

        count = len(self._no_go)
        self._no_go.clear()
        return count

    def is_no_go(self, cell: Cell) -> bool:
        return cell in self._no_go

    def is_blocked(self, cell: Cell) -> bool:
        """Czy komórka jest NIEPRZECHODNIA jako cel/krok: poza granicą LUB wyuczona no_go (tekstura/pustka)."""

        return cell in self._excluded or cell in self._no_go

    def blocked_ahead(self, pos: Point, target: Point, *, lookahead: float | None = None) -> bool:
        """G6: czy krok `pos→target` wbiega w teksturę/no_go/POZA kopertę → „NIE PCHAJ".

        Patrzy o `lookahead` (domyślnie `cell_size` = ślad pierścienia = pas tolerancji) w kierunku
        celu. True → zatrzymaj się w pasie tolerancji, doskanuj co widać, `record_block(pos,"stall")`,
        i wybierz następny klaster (`next_target` pominie zablokowaną komórkę). Zapobiega wejściu
        w nieprzechodni obiekt i wyjściu poza farmę. (Por. `heading_nav.aim_point` — recovery gdy JUŻ poza.)
        """

        step = self.cell_size if lookahead is None else lookahead
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return False
        ax, ay = pos[0] + dx / d * step, pos[1] + dy / d * step
        x_min, x_max, y_min, y_max = self.envelope
        if not (x_min <= ax <= x_max and y_min <= ay <= y_max):
            return True                      # krok wyszedłby poza kopertę
        return self.is_blocked(self.cell_of((ax, ay)))

    # Sygnały blokady traktowane jako TEKSTURA/pustka (→ kandydat no_go): bot wielokrotnie zawodzi
    # i nic nie znajduje. `goto_fail` NIE jest sygnałem tekstury: oznacza wyłącznie, że planner
    # nie znalazł drogi do odległego celu. W przeciwnym razie błąd nawigacji amputowałby farmę.
    _NO_GO_FAIL_KINDS = ("failed_open", "stall")

    def border_adjustments(self, *, fail_floor: int = 3) -> list[Cell]:
        """Z heatmapy blokad: komórki, gdzie bot WIELOKROTNIE zawodzi i NIC nie znajduje → no_go.

        Kryterium (czyste, bez sklepów/hull): suma `failed_open + stall ≥ fail_floor`
        ORAZ `scans_in_cell == 0` (zero NOWYCH sklepów) → rejon pusty/nieosiągalny/tekstura (G6),
        nie warto tam wracać. Komórki już wykluczone/no_go pomijane. `edge_hit`/`boundary_turn` NIE
        dają no_go (zdrowe zawracanie na granicy). Zwraca KANDYDATÓW; zastosowanie = `mark_no_go` (D5).
        """

        out: list[Cell] = []
        for cell, kinds in self._blocks.items():
            if cell in self._excluded or cell in self._no_go:
                continue
            fails = sum(kinds.get(k, 0) for k in self._NO_GO_FAIL_KINDS)
            if fails >= fail_floor and self._scans.get(cell, 0) == 0:
                out.append(cell)
        return sorted(out)

    def scans_in_cell(self, cell: Cell) -> int:
        return self._scans.get(cell, 0)

    def dups_in_cell(self, cell: Cell) -> int:
        return self._dups.get(cell, 0)

    def dup_streak_in_cell(self, cell: Cell) -> int:
        """KOLEJNE duplikaty od ostatniego nowego sklepu (G3: sygnał „pierścień daje same znane")."""

        return self._dup_streak.get(cell, 0)

    def cell_exhausted(self, cell: Cell, *, dup_floor: int = 2, expected: int | None = None) -> bool:
        """Czy komórka jest przebrana (nie ma sensu dalej tu klikać).

        Wyczerpana, gdy SERIA kolejnych duplikatów (bez nowego sklepu) ≥ `dup_floor` —
        „pierścień daje same znane" (G3) — LUB jeśli `expected` podane, zeskanowano
        ≥ `expected` NOWYCH. SERIA (nie suma!): w gęstej komórce (~17 sklepów) sporadyczny
        duplikat wśród nowych NIE znaczy przebrania; dopiero `dup_floor` znanych z rzędu znaczy.
        """

        if expected is not None and self._scans.get(cell, 0) >= expected:
            return True
        return self._dup_streak.get(cell, 0) >= dup_floor

    def should_skip_click(
        self, char_pos: Point, *, dup_floor: int = 2, expected: int | None = None
    ) -> bool:
        """Czy POMINĄĆ klikanie tu i iść do niepokrytej (komórka postaci pokryta+wyczerpana).

        Bariera anty-reskan PRZED klikiem: nie marnuj otwarcia na sklep, który niemal
        na pewno jest duplikatem. Gdy True → `next_target` i krok, zamiast klikać.
        """

        cell = self.cell_of(char_pos)
        return self.is_cell_covered(cell) and self.cell_exhausted(
            cell, dup_floor=dup_floor, expected=expected
        )

    # --- DONE = saturacja (C6: anti-przedwczesne-przejście, COVERAGE_MAP_CORE §6b) --- #
    # Problem: „pokryta" = dotknięta RAZ → bot opuszcza sektor po 1 skanie, zostawia resztę
    # nieskanowaną. Rozwiązanie: komórka jest GOTOWA dopiero gdy WYSYCONA (pokryta + wyczerpana:
    # `dup_floor` duplikatów albo `expected` nowych). `next_target(until="done")` wraca do
    # pokrytych-ale-niewysyconych, aż przestaną dawać nowe sklepy. Mniejszy `cell_size` (~10–12u)
    # dodatkowo zbliża „1 przystanek ≈ komórka". Dwa lewary razem = brak przeskoku.

    def is_done(self, cell: Cell, *, dup_floor: int = 2, expected: int | None = None) -> bool:
        """Komórka GOTOWA = pokryta I wyczerpana (≠ tylko dotknięta). Nie-cel (poza granicą / no_go) też = done."""

        if cell in self._excluded or cell in self._no_go or cell in self._skip_target:
            return True
        return self.is_cell_covered(cell) and self.cell_exhausted(
            cell, dup_floor=dup_floor, expected=expected
        )

    def pending_cells(self, *, dup_floor: int = 2, expected: int | None = None) -> list[Cell]:
        """Komórki farmy jeszcze NIEgotowe (niepokryte LUB pokryte-ale-niewysycone)."""

        return [c for c in self.all_cells()
                if not self.is_done(c, dup_floor=dup_floor, expected=expected)]

    def all_done(self, *, dup_floor: int = 2, expected: int | None = None) -> bool:
        """Czy CAŁA farma wysycona (koniec biegu = `coverage_done`, nie `route_exhausted`)."""

        return not self.pending_cells(dup_floor=dup_floor, expected=expected)

    # --- wybór celu ---------------------------------------------------------- #

    def next_target(
        self, pos: Point, *, order: str = "nearest",
        until: str = "covered", dup_floor: int = 2, expected: int | None = None,
        prefer_uncovered: bool = False, heading: Point | None = None,
    ) -> Point | None:
        """Środek następnej komórki-celu (świat). `None` = nic do roboty → koniec.

        `until="covered"` (domyślnie, kompat.): celuje w NIEpokryte (komórka opuszczana po 1 dotknięciu).
        `until="done"` (C6): celuje w NIEwysycone — bot wraca do pokrytej-ale-niewyczerpanej komórki,
        aż przestaje dawać nowe sklepy (anti-przedwczesne-przejście). DeepSeek: użyj `until="done"`.
        `order="nearest"`: najbliższa (greedy); `order="boustrophedon"`: wężyk po komórkach.

        `prefer_uncovered=True` (z `until="done"`): celuj NAJPIERW w PUSTE (niepokryte) komórki, a
        do pokrytych-ale-niewysyconych wracaj DOPIERO gdy pustych brak — „skanujemy przód, nie wokół":
        bot idzie w świeży teren zamiast re-drenować to, co już dotknięte (saturacja jako domknięcie, nie cel).
        `heading` (wektor jazdy): przy `order="nearest"` przednie komórki MAJĄ PIERWSZEŃSTWO przed
        tylnymi, potem najbliższa — „od najbliższych do najdalszych" w kierunku marszu, nie wstecz.
        """

        candidates = self._target_candidates(
            until=until,
            dup_floor=dup_floor,
            expected=expected,
            prefer_uncovered=prefer_uncovered,
        )
        if not candidates:
            return None
        if order == "boustrophedon":
            target = self._boustrophedon_first(candidates)
        else:
            target = min(candidates, key=lambda c: self._target_key(pos, c, heading))
        return self.cell_center(target)

    def path_to_next_target(
        self,
        pos: Point,
        *,
        order: str = "nearest",
        until: str = "covered",
        dup_floor: int = 2,
        expected: int | None = None,
        prefer_uncovered: bool = False,
        heading: Point | None = None,
    ) -> tuple[Cell, ...] | None:
        """Najkrótsza 4-spójna droga do osiągalnego celu pokrycia.

        ``next_target`` wybierał dotąd geograficznie najbliższy cel, nawet gdy
        oddzielała go granica lub ``no_go``. Tu BFS najpierw znajduje składową,
        w której stoi postać, a dopiero potem wybiera cel według tej samej
        polityki (nearest / boustrophedon). Zwrócona droga zawiera komórkę startu
        i celu; ``None`` znaczy brak osiągalnego celu w bieżącej składowej.
        """

        candidates = self._target_candidates(
            until=until,
            dup_floor=dup_floor,
            expected=expected,
            prefer_uncovered=prefer_uncovered,
        )
        if not candidates:
            return None

        start = self.cell_of(pos)
        if self.is_blocked(start):
            start = self.nearest_reentry_cell(pos)
            if start is None:
                return None
        parents: dict[Cell, Cell | None] = {start: None}
        queue: deque[Cell] = deque([start])
        while queue:
            cell = queue.popleft()
            for neighbor in self._neighbors(cell):
                if neighbor in parents or self.is_blocked(neighbor):
                    continue
                parents[neighbor] = cell
                queue.append(neighbor)

        reachable = [cell for cell in candidates if cell in parents]
        if not reachable:
            return None
        if order == "boustrophedon":
            target = self._boustrophedon_first(reachable)
        else:
            target = min(reachable, key=lambda c: self._target_key(pos, c, heading))

        path: list[Cell] = []
        current: Cell | None = target
        while current is not None:
            path.append(current)
            current = parents[current]
        path.reverse()
        return tuple(path)

    def nearest_reentry_cell(self, pos: Point) -> Cell | None:
        """Najbliższa przejezdna komórka farmy, gdy pozycja wypadła w X/no_go.

        To nie jest cel coverage, tylko punkt powrotu do grafu. Bez tego BFS
        startował z komórki wykluczonej i mógł zgłosić lokalne wyczerpanie,
        mimo że globalnie zostały niepokryte cele.
        """

        cells = [c for c in self.all_cells() if not self.is_blocked(c)]
        if not cells:
            return None
        return min(cells, key=lambda c: self._dist2(pos, self.cell_center(c)))

    def reentry_path(self, pos: Point) -> tuple[Cell, ...] | None:
        """Ścieżka jednoelementowa do najbliższego wejścia w graf coverage.

        Wywołujący może użyć tego, gdy `path_to_next_target` zwróci None, ale
        `all_done()` jest False. Zwracamy komórkę, do której trzeba fizycznie
        wrócić, zamiast kończyć run.
        """

        cell = self.cell_of(pos)
        if not self.is_blocked(cell):
            return (cell,)
        reentry = self.nearest_reentry_cell(pos)
        if reentry is None:
            return None
        return (reentry,)

    def next_reachable_hop(self, pos: Point, **kwargs) -> Point | None:
        """Pierwszy bezpieczny krok na ścieżce do następnego celu pokrycia."""

        path = self.path_to_next_target(pos, **kwargs)
        if path is None:
            return None
        # Gdy postać już stoi w celu, zwracamy jego środek; wywołujący może
        # przeskanować bieżący postój bez wykonywania zbędnego ruchu.
        return self.cell_center(path[1] if len(path) > 1 else path[0])

    def _target_candidates(
        self,
        *,
        until: str,
        dup_floor: int,
        expected: int | None,
        prefer_uncovered: bool,
    ) -> list[Cell]:
        if until == "done":
            candidates = self.pending_cells(dup_floor=dup_floor, expected=expected)
            if prefer_uncovered:
                uncov = [c for c in candidates if c not in self._covered]
                if uncov:
                    candidates = uncov
            return candidates
        return self.uncovered_cells()

    def _neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        x, y = cell
        neighbors = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        return tuple(
            candidate for candidate in neighbors
            if 0 <= candidate[0] < self._ncols and 0 <= candidate[1] < self._nrows
        )

    def _target_key(self, pos: Point, cell: Cell, heading: Point | None):
        """Klucz sortowania celu: (przód-pierwszy, dystans², cy, cx). Heading None → sam dystans
        (identyczne z poprzednim zachowaniem — stała 0 nie zmienia argmin)."""

        cc = self.cell_center(cell)
        d2 = self._dist2(pos, cc)
        ahead = 0
        if heading is not None:
            dot = (cc[0] - pos[0]) * heading[0] + (cc[1] - pos[1]) * heading[1]
            ahead = 0 if dot > 0 else 1        # przednie komórki (dot>0) przed tylnymi
        return (ahead, d2, cell[1], cell[0])

    def coverage_path(self) -> list[Cell]:
        """G4: PEŁNY plan zamiatania farmy — wszystkie komórki-cele w kolejności wężyka (lawn-mower).

        Rzędy rosnąco; rząd parzysty L→P, nieparzysty P→L (minimalny przejazd między postojami).
        Wyklucza poza-granicą i `no_go`. Deterministyczny obchód „postój po postoju": DeepSeek
        iteruje, jedzie do każdej komórki, skanuje aż saturacja (`is_done`), pomija już-gotowe.
        Krok między sąsiednimi komórkami = `cell_size` ≈ ślad pierścienia klikalnego (G5).
        Reaktywny odpowiednik (omija dynamicznie no_go/saturację) = `next_target(until="done")`.
        """

        skip = self._skip()
        path: list[Cell] = []
        for row in range(self._nrows):
            cols = range(self._ncols) if row % 2 == 0 else range(self._ncols - 1, -1, -1)
            path.extend((col, row) for col in cols if (col, row) not in skip)
        return path

    def remaining(self) -> int:
        return self.total_cells - self.covered_count

    # --- wewnętrzne ---------------------------------------------------------- #

    def _boustrophedon_first(self, uncovered: list[Cell]) -> Cell:
        # Najniższy rząd z niepokrytą; w rzędzie parzystym → najmniejsze cx, w nieparzystym → największe.
        by_row = min(c[1] for c in uncovered)
        row = [c for c in uncovered if c[1] == by_row]
        return min(row, key=lambda c: c[0]) if by_row % 2 == 0 else max(row, key=lambda c: c[0])

    @staticmethod
    def _dist2(a: Point, b: Point) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    # --- persystencja (C1: mapa żyje między biegami) ------------------------- #
    # `coverage.json` w partycji mapy. Granica NIE jest tu serializowana — żyje w
    # `farm_map.json` i jest podawana przy load (boundary=) → `_excluded` przeliczane.
    # Format list [cx,cy(,n)] zamiast dict-z-kluczem-krotką (JSON nie ma krotek-kluczy).

    def to_dict(self) -> dict:
        return {
            "envelope": list(self.envelope),
            "cell_size": self.cell_size,
            "covered": [list(c) for c in sorted(self._covered)],
            "scans": [[cx, cy, n] for (cx, cy), n in sorted(self._scans.items())],
            "dups": [[cx, cy, n] for (cx, cy), n in sorted(self._dups.items())],
            "dup_streak": [[cx, cy, n] for (cx, cy), n in sorted(self._dup_streak.items())],
            "blocks": [[cx, cy, kinds] for (cx, cy), kinds in sorted(self._blocks.items())],
            "no_go": [list(c) for c in sorted(self._no_go)],
        }

    @classmethod
    def from_dict(cls, d: dict, *, boundary: Polygon | None = None,
                  boundary_margin: float = 0.0) -> "CoverageMap":
        m = cls(
            tuple(d["envelope"]),                       # type: ignore[arg-type]
            cell_size=float(d.get("cell_size", DEFAULT_CELL_SIZE)),
            boundary=boundary,
            boundary_margin=boundary_margin,
        )
        m._covered = {(int(cx), int(cy)) for cx, cy in d.get("covered", [])}
        m._scans = {(int(cx), int(cy)): int(n) for cx, cy, n in d.get("scans", [])}
        m._dups = {(int(cx), int(cy)): int(n) for cx, cy, n in d.get("dups", [])}
        if "dup_streak" in d:
            m._dup_streak = {(int(cx), int(cy)): int(n) for cx, cy, n in d["dup_streak"]}
        else:
            m._dup_streak = dict(m._dups)   # migracja starej mapy: przyjmij sumę dups jako serię
        m._blocks = {(int(cx), int(cy)): {str(k): int(v) for k, v in kinds.items()}
                     for cx, cy, kinds in d.get("blocks", [])}
        m._no_go = {(int(cx), int(cy)) for cx, cy in d.get("no_go", [])}
        return m

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, boundary: Polygon | None = None,
             boundary_margin: float = 0.0) -> "CoverageMap | None":
        """Wczytaj stan z `coverage.json`. `None` gdy brak pliku / uszkodzony (bieg startuje świeżo).

        `boundary` (z `farm_map.json`) podane → `_excluded` przeliczane na bieżącą granicę,
        więc granica i pokrycie zawsze spójne nawet gdy obrys się zmienił między biegami.
        """

        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls.from_dict(d, boundary=boundary, boundary_margin=boundary_margin)
