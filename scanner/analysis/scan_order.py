"""Kolejność skanowania w kadrze: PRZÓD nie wokół, od najbliższych do najdalszych (Claude, pure).

**Po co.** Skan pierścienia 360° wokół postaci re-łapie sklepy ZA plecami (już zeskanowane gdy
postać szła do przodu) → „skanujemy te same sklepy". Rozwiązanie geometryczne: zostaw tylko
kandydatów w PRZEDNIM stożku względem kierunku jazdy (heading z odometrii) i klikaj od
NAJBLIŻSZEGO do najdalszego osiągalnego. Tył (przebyty) odpada Z GEOMETRII — nie z dedupu po
fakcie, więc nie marnujemy otwarcia na pewny duplikat ([[shop-identity-two-level]]).

Czysta geometria: zero gry/OCR/ekranu-vs-świat. `center`, `heading`, `points` w JEDNYCH
jednostkach (DeepSeek podaje screen-heading + screen-bloby ALBO świat — moduł agnostyczny).
Heading None/zerowy (nie znamy jeszcze kierunku, przed 1. 'w') → brak filtra kąta, sam dystans
(degraduje łagodnie). Komplementarny do `coverage_map.next_target` (cel KOMÓRKOWY: PUSTE pierwsze).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]


def _unit(v: Point | None) -> Point | None:
    if v is None:
        return None
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return None
    return (v[0] / n, v[1] / n)


def forward_indices(
    center: Point,
    heading: Point | None,
    points: Sequence[Point],
    *,
    half_angle_deg: float = 90.0,
    min_r: float = 0.0,
    max_r: float | None = None,
) -> list[int]:
    """Indeksy `points` w PRZEDNIM stożku, posortowane NAJBLIŻEJ→NAJDALEJ (osiągalne).

    - `heading` = kierunek jazdy (wektor; None/zero → bez filtra kąta, sama odległość).
    - `half_angle_deg` = połowa rozwarcia stożka. 90 = przednia PÓŁPŁASZCZYZNA (180°, tnie cały
      tył); mniejsze = węższy stożek (mniej re-skanu, ale ryzyko pominięcia boków); 180 = pełne koło.
    - `min_r`/`max_r` = pas OSIĄGALNOŚCI (pierścień klikalny; max_r=None → bez górnego limitu).
    Zwraca INDEKSY (nie punkty), by DeepSeek przeporządkował swoją listę blobów/klikalnych
    bez gubienia powiązania blob↔punkt. Punkty wprost → `forward_points`.
    """

    h = _unit(heading)
    cos_lim = math.cos(math.radians(half_angle_deg))
    scored: list[tuple[float, int]] = []
    for i, p in enumerate(points):
        dx, dy = p[0] - center[0], p[1] - center[1]
        r = math.hypot(dx, dy)
        if r < min_r or (max_r is not None and r > max_r):
            continue
        if h is not None and r > 1e-9:
            cos_a = (dx * h[0] + dy * h[1]) / r   # cos kąta heading↔kierunek-do-punktu
            if cos_a + 1e-9 < cos_lim:            # epsilon: granica stożka (np. bok na 90°) INKLUZYWNA
                continue                          # poza stożkiem (z tyłu / poza rozwarciem)
        scored.append((r, i))
    scored.sort()
    return [i for _, i in scored]


def forward_points(
    center: Point,
    heading: Point | None,
    points: Sequence[Point],
    *,
    half_angle_deg: float = 90.0,
    min_r: float = 0.0,
    max_r: float | None = None,
) -> list[Point]:
    """Jak `forward_indices`, ale zwraca same PUNKTY w kolejności skanowania (najbliżej→najdalej)."""

    idx = forward_indices(
        center, heading, points,
        half_angle_deg=half_angle_deg, min_r=min_r, max_r=max_r,
    )
    return [tuple(points[i]) for i in idx]
