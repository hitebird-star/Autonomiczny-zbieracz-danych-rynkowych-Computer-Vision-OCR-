"""Granica farmy = „Linia akceptowalności" (Claude, offline-pure, zero zależności).

Koncept usera (24.06): prostokątna koperta jest ZA DUŻA. Realna granica = gdzie sklepy
NAPRAWDĘ są. Bot przy granicy: jeśli widzi sklep — skanuje, ale NIE biegnie dalej w pustkę;
skręca (±45°) i szuka wzdłuż granicy; gdy wszystko zeskanowane — zawraca.

Ten moduł = pure-logika granicy:
- `convex_hull(shops)` — wylicz ciasną granicę z pozycji sklepów (od razu, bez live'a),
- `point_in_polygon` / `dist_to_boundary` / `near_boundary` — gdzie jest bot względem granicy,
- `would_exit` / `should_turn_at_boundary` — decyzja „skręć zamiast biec dalej",
- `FarmBoundary` + load/save `farm_map.json` — trwała, edytowalna granica.

Wszystko w układzie świata (X,Y) z OCR — wspólna ramka z `coverage_map`/`shop_registry`/`map_view`.
Precyzyjne obrysowanie (spacer po obwodzie + OCR 20 fps uśredniony) to KARMA z live (pas DeepSeek);
ten moduł dopasowuje wielokąt do punktów i go egzekwuje. Por. `coverage_map.py`, `map_view.py`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

Point = tuple[float, float]
Polygon = list[Point]

BOUNDARY_FILENAME = "farm_map.json"


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


@dataclass(frozen=True)
class PointEstimate:
    """Jeden punkt obrysu z N odczytów OCR: mediana + jakość (rozrzut)."""

    point: Point
    samples: int      # ile prawidłowych odczytów uśredniono
    spread: float     # max odległość próbki od mediany (mały = pewny punkt)


def robust_point(
    readings: list[Point | None], *, reject_beyond: float | None = None
) -> PointEstimate | None:
    """Uśrednij serię odczytów OCR (np. 20 klatek w 1s) w JEDEN dokładny punkt.

    Mediana per-oś (odporna na pojedyncze misready OCR — stąd, nie średnia). `reject_beyond`
    odrzuca próbki dalej niż próg od mediany (drugie przejście) — czyści grube błędy.
    `None` gdy brak prawidłowych odczytów. `spread` mówi, czy punkt jest pewny (mały rozrzut).
    """

    pts = [(float(p[0]), float(p[1])) for p in readings if p is not None]
    if not pts:
        return None
    mx, my = _median([p[0] for p in pts]), _median([p[1] for p in pts])
    if reject_beyond is not None and len(pts) >= 3:
        kept = [p for p in pts if math.hypot(p[0] - mx, p[1] - my) <= reject_beyond]
        if kept:
            pts = kept
            mx, my = _median([p[0] for p in pts]), _median([p[1] for p in pts])
    spread = max(math.hypot(p[0] - mx, p[1] - my) for p in pts)
    return PointEstimate((mx, my), len(pts), spread)


def convex_hull(points: list[Point]) -> Polygon:
    """Otoczka wypukła (Andrew's monotone chain) — ciasna granica wokół punktów. CCW.

    Zwraca [] dla <3 unikalnych punktów. To pierwsza, data-driven „linia akceptowalności":
    ciaśniejsza niż prostokąt, bo opisuje realne pozycje sklepów.
    """

    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: Polygon = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: Polygon = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    """Czy punkt jest WEWNĄTRZ wielokąta (ray casting). Brzeg liczony nieściśle."""

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


def _dist_point_segment(p: Point, a: Point, b: Point) -> float:
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_to_boundary(pt: Point, poly: Polygon) -> float:
    """Najmniejsza odległość punktu do KRAWĘDZI wielokąta (niezależnie od strony)."""

    if len(poly) < 2:
        return float("inf")
    return min(_dist_point_segment(pt, poly[i], poly[(i + 1) % len(poly)])
               for i in range(len(poly)))


def near_boundary(pt: Point, poly: Polygon, *, margin: float = 7.0) -> bool:
    """Czy bot jest BLISKO granicy (w pasie `margin` od krawędzi) lub już poza nią.

    To strefa, w której obowiązuje reguła „skręć, nie biegnij dalej".
    """

    return (not point_in_polygon(pt, poly)) or dist_to_boundary(pt, poly) <= margin


def signed_distance_to_polygon(pt: Point, poly: Polygon) -> float:
    """Odległość z kierunkiem: dodatnia wewnątrz, ujemna poza wielokątem."""

    distance = dist_to_boundary(pt, poly)
    return distance if point_in_polygon(pt, poly) else -distance


def would_exit(pos: Point, next_pos: Point, poly: Polygon) -> bool:
    """Czy następny krok WYJDZIE poza granicę (jestem w środku, krok ląduje na zewnątrz)."""

    return point_in_polygon(pos, poly) and not point_in_polygon(next_pos, poly)


def should_turn_at_boundary(
    pos: Point, next_pos: Point, poly: Polygon, *, margin: float = 7.0
) -> bool:
    """Decyzja „skręć (±45°) zamiast iść prosto": następny krok wyjdzie LUB jesteśmy w pasie granicy."""

    d_now = signed_distance_to_polygon(pos, poly)
    d_next = signed_distance_to_polygon(next_pos, poly)
    if d_next <= 0.0:
        return True
    if d_now < margin and d_next <= d_now:
        return True
    return False


def _centroid(poly: Polygon) -> Point:
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def expand_to_include(poly: Polygon, pt: Point, *, margin: float = 0.0) -> Polygon:
    """Najmniejsza rozbudowa pierścienia tak, by `pt` znalazł się WEWNĄTRZ granicy.

    Wstawia nowy wierzchołek na NAJBLIŻSZEJ krawędzi (lokalnie, zachowuje resztę kształtu —
    NIE convex hull, żeby nie tracić wklęsłości „linii akceptowalności"). `margin` > 0 wypycha
    wierzchołek o tyle na zewnątrz (od centroidu), żeby sklep był pewnie wewnątrz, nie NA brzegu
    (bot ma dojść do sprzedawcy, więc granica ciut za nim). Zwraca NOWY pierścień (nie mutuje).
    """

    if point_in_polygon(pt, poly):
        return list(poly)
    p = (float(pt[0]), float(pt[1]))
    if len(poly) < 3:
        return list(poly) + [p]
    if margin > 0:
        cx, cy = _centroid(poly)
        d = math.hypot(p[0] - cx, p[1] - cy)
        if d > 1e-9:
            p = (p[0] + (p[0] - cx) / d * margin, p[1] + (p[1] - cy) / d * margin)
    n = len(poly)
    best_i = min(range(n),
                 key=lambda i: _dist_point_segment(p, poly[i], poly[(i + 1) % n]))
    return list(poly[:best_i + 1]) + [p] + list(poly[best_i + 1:])


@dataclass
class FarmBoundary:
    """Trwała granica farmy (wielokąt w X,Y) + metadane. Zapis/odczyt `farm_map.json`."""

    polygon: Polygon
    source: str = "hull_of_shops"   # "hull_of_shops" | "perimeter_walk" | "hand"

    @classmethod
    def from_shops(cls, shops: list[Point]) -> "FarmBoundary":
        return cls(convex_hull(shops), source="hull_of_shops")

    @classmethod
    def from_perimeter(
        cls,
        points: list[Point],
        *,
        min_gap: float = 2.0,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> "FarmBoundary":
        """Zbuduj granicę z UPORZĄDKOWANYCH punktów obrysu (spacer po obwodzie + OCR).

        Punkty są już w kolejności wzdłuż obwodu → tworzą wielokąt wprost (NIE otoczka).
        Zlewa kolejne punkty bliższe niż `min_gap` (20 fps daje gęste, prawie identyczne) i
        usuwa domknięcie-duplikat. Por. spec obrysu (pas DeepSeek: karma punktów).

        `bounds=(x_min, x_max, y_min, y_max)` = sanity-koperta: punkty poza nią są ODRZUCANE.
        To łapie systematyczny misread OCR (cała seria 20 klatek czyta tak samo, np. cyfra-
        wstawka 413→4137 albo 680→60), którego `robust_point`/`spread` NIE wyłapie (spread≈0,
        bo wszystkie klatki zgodne). Bez `bounds` zachowanie jak dawniej. Por. [[stage4-coord-separator]].
        """

        if bounds is not None:
            x0, x1, y0, y1 = bounds
            points = [p for p in points if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
        ring: Polygon = []
        for p in points:
            q = (float(p[0]), float(p[1]))
            if not ring or math.hypot(q[0] - ring[-1][0], q[1] - ring[-1][1]) >= min_gap:
                ring.append(q)
        if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0],
                                         ring[0][1] - ring[-1][1]) < min_gap:
            ring.pop()
        return cls(ring, source="perimeter_walk")

    def contains(self, pt: Point) -> bool:
        return point_in_polygon(pt, self.polygon)

    def grown_to_include(
        self,
        pt: Point,
        *,
        bounds: tuple[float, float, float, float] | None = None,
        max_jump: float | None = None,
        margin: float = 3.0,
    ) -> "tuple[FarmBoundary, bool]":
        """Auto-rozszerzenie granicy, gdy bot ZOBACZY sklep poza nią (samokorekta podczas skanu).

        Zwraca `(granica, czy_urosła)`. Granica rośnie TYLKO gdy `pt` przejdzie OBA strażniki
        (inaczej zwraca siebie + False — nie zaśmieca granicy misreadem OCR):
        - `bounds=(x_min,x_max,y_min,y_max)` — koperta sanity: sklep „na" X=4137 to misread,
          NIE realny sklep za granicą → ignoruj (ta sama ochrona co `from_perimeter(bounds=)`).
        - `max_jump` — realny nowy sklep jest TUŻ za granicą (kilka u); odczyt o `dist_to_boundary`
          > `max_jump` to skok nierealny (misread/teleport) → ignoruj.

        Gdy `pt` jest już wewnątrz → bez zmian (False). Caller zapisuje `farm_map.json` tylko gdy
        True. `margin` daje oddech, by sklep był wewnątrz, nie NA brzegu. Por. [[stage4-coord-separator]].
        """

        p = (float(pt[0]), float(pt[1]))
        if self.contains(p):
            return self, False
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
                return self, False
        if max_jump is not None and dist_to_boundary(p, self.polygon) > max_jump:
            return self, False
        return FarmBoundary(expand_to_include(self.polygon, p, margin=margin),
                            source=self.source), True

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]; ys = [p[1] for p in self.polygon]
        return (min(xs), max(xs), min(ys), max(ys)) if self.polygon else (0, 0, 0, 0)

    def area(self) -> float:
        """Pole wielokąta (shoelace) — do porównania „o ile ciaśniej niż prostokąt"."""

        n = len(self.polygon)
        if n < 3:
            return 0.0
        s = sum(self.polygon[i][0] * self.polygon[(i + 1) % n][1]
                - self.polygon[(i + 1) % n][0] * self.polygon[i][1] for i in range(n))
        return abs(s) / 2.0

    def to_dict(self) -> dict:
        return {"polygon": [[round(x, 1), round(y, 1)] for x, y in self.polygon],
                "source": self.source}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FarmBoundary | None":
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        poly = [(float(x), float(y)) for x, y in d.get("polygon", [])]
        return cls(poly, source=str(d.get("source", "hand")))
