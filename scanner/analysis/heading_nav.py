"""C4: decyzja obrotu do celu z żywego heading (Claude, offline-pure, ZERO kolizji nazw).

Domyka F2 („goto dowozi 1/8"): dziś `drive_toward_target` obraca ŚLEPO 'd' + używa STAŁEGO
wektora 'w' — po obrocie dead-reckoning kłamie, postać nie dojeżdża. Tu jest czysta geometria
„czy patrzę na cel / w którą stronę się obrócić", którą DeepSeek wpina w pętlę rotate-measure-correct.

Świadomie OSOBNY moduł (nie `nav_guards.py`): nav_guards rozjechał się między gałęziami
(claude: `within_envelope(pos,env)` vs live: `within_envelope(x,y,env)`) — żeby C4 był adoptowalny
bez wchodzenia w ten rozjazd. Heading bierzemy z `heading_from_ocr` (wektor (dx,dy) z dwóch OCR).

Konwencja kąta: ramka (X,Y) z OCR. `signed_angle>0` = skręt CCW w tej ramce. Mapowanie CCW→klawisz
(a/d) zależy od gry → param `d_is_ccw` (KALIBRACJA live) albo wariant bez-konwencji (rotate-measure).
"""

from __future__ import annotations

import math

Vec = tuple[float, float]
Point = tuple[float, float]


def direction_to(pos: Point, target: Point) -> Vec:
    """Wektor jednostkowy z `pos` do `target`. (0,0) gdy pokrywają się (już na celu)."""

    dx, dy = target[0] - pos[0], target[1] - pos[1]
    d = math.hypot(dx, dy)
    return (0.0, 0.0) if d < 1e-9 else (dx / d, dy / d)


def signed_angle(a: Vec, b: Vec) -> float:
    """Kąt skrętu od wektora `a` do `b` w radianach, znormalizowany do [-π, π].

    Znak = strona obrotu (w ramce X,Y: + = CCW). 0 gdy któryś wektor zerowy.
    """

    if (a[0] == 0 and a[1] == 0) or (b[0] == 0 and b[1] == 0):
        return 0.0
    ang = math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])
    return (ang + math.pi) % (2 * math.pi) - math.pi


def facing_error_deg(heading: Vec, pos: Point, target: Point) -> float:
    """Bezwzględny błąd kąta (stopnie) między `heading` a kierunkiem `pos→target`. 0 = patrzę na cel."""

    tgt = direction_to(pos, target)
    if tgt == (0.0, 0.0):
        return 0.0
    return abs(math.degrees(signed_angle(heading, tgt)))


def should_drive(heading: Vec, pos: Point, target: Point, *, tol_deg: float = 25.0) -> bool:
    """Czy patrzę na cel wystarczająco, by jechać 'w' (błąd kąta ≤ `tol_deg`).

    Bez headingu (np. brak pary OCR) → True: jedź i zmierz heading z ruchu (rotate-measure-correct).
    """

    if heading == (0.0, 0.0):
        return True
    return facing_error_deg(heading, pos, target) <= tol_deg


def turn_to_face(
    heading: Vec, pos: Point, target: Point, *, tol_deg: float = 25.0, d_is_ccw: bool = False
) -> str:
    """Co nacisnąć: `'w'` (patrzę na cel → jedź), `'a'`/`'d'` (obróć w stronę celu).

    `d_is_ccw` = czy klawisz 'd' obraca CCW w ramce (X,Y) — KALIBRACJA live (param, domyślnie
    'd'=CW). Gdy heading nieznany → 'w' (jedź, zmierz). To wariant „mam konwencję"; wariant
    bez konwencji = `should_drive` + nudge-zmierz-popraw (DeepSeek wybiera klawisz po efekcie).
    """

    if should_drive(heading, pos, target, tol_deg=tol_deg):
        return "w"
    ang = signed_angle(heading, direction_to(pos, target))  # >0 = trzeba CCW
    turn_ccw = ang > 0
    if turn_ccw:
        return "d" if d_is_ccw else "a"
    return "a" if d_is_ccw else "d"


Envelope = tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)


def nearest_in_envelope(pos: Point, envelope: Envelope) -> Point:
    """Najbliższy punkt WEWNĄTRZ koperty (clamp). Gdy `pos` w środku → zwraca `pos`."""

    x0, x1, y0, y1 = envelope
    return (min(max(pos[0], x0), x1), min(max(pos[1], y0), y1))


def is_outside(pos: Point, envelope: Envelope) -> bool:
    x0, x1, y0, y1 = envelope
    return not (x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1)


def direction_to_envelope(pos: Point, envelope: Envelope) -> Vec:
    """Wektor jednostkowy „z powrotem do farmy" (do najbliższego punktu koperty). (0,0) gdy w środku.

    Klucz do recovery „wyszedłem poza obszar": cel = `nearest_in_envelope(pos)`, a `turn_to_face`
    obróci bota TWARZĄ do farmy. Bez tego goto celuje w komórkę, ale ślepy 'w' pcha dalej OUT.
    """

    return direction_to(pos, nearest_in_envelope(pos, envelope))


def aim_point(pos: Point, target: Point, envelope: Envelope, *, inset: float = 15.0) -> Point:
    """Dokąd realnie celować: gdy bot POZA farmą → punkt powrotny w głąb (NIE oryginalny cel),
    
    inaczej → oryginalny `target`. Jedna linia decyzji dla `_drive_toward_target`: priorytet to
    najpierw wrócić w obszar, potem normalne docieranie. Por. `recovery_target`, `is_outside`.
    """
    
    if is_outside(pos, envelope):
        return recovery_target(pos, envelope, inset=inset)
    return target


def recovery_target(pos: Point, envelope: Envelope, *, inset: float = 15.0) -> Point:
    """Cel powrotny GŁĘBIEJ niż krawędź (o `inset` do środka), żeby bot nie utknął NA brzegu.

    Jedzie do punktu wsuniętego do wnętrza od najbliższej ściany — pewniej wraca w obszar niż
    celując dokładnie w krawędź (gdzie OCR znika i znów wypada).
    """

    x0, x1, y0, y1 = envelope
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    nx, ny = nearest_in_envelope(pos, envelope)
    # wsuń w stronę środka o `inset`
    nx += inset if cx > nx else (-inset if cx < nx else 0.0)
    ny += inset if cy > ny else (-inset if cy < ny else 0.0)
    return (min(max(nx, x0), x1), min(max(ny, y0), y1))


def better_turn_key(err_before: float, err_after: float, tried_key: str) -> str:
    """Rotate-measure-correct BEZ kalibracji konwencji: po próbnym nudge'u `tried_key`,

    jeśli błąd kąta zmalał → trzymaj `tried_key`; jeśli wzrósł → druga strona. Zwraca klawisz
    do dalszego obrotu. To bezpieczny wariant gdy `d_is_ccw` niepewne (sam się douczy).
    """

    if err_after < err_before:
        return tried_key
    return "a" if tried_key == "d" else "d"
