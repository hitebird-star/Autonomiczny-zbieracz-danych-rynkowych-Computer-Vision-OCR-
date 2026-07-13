"""Warstwa zabezpieczen nawigacji (A1–C3, E, F) – czyste funkcje decyzyjne.

Zero importow z gry. Testowalne offline. Uzywane przez densyfikacje (Faza 1)
i navigate_to (Faza 2). Guardy D1–D4 (sensing) sa w pasie live.

Decyzja = czysta funkcja: bierze liczby/historie, zwraca bool/enum.
DeepSeek wplata to w pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Typy
# ---------------------------------------------------------------------------

class FailReason(StrEnum):
    """Powody FAIL – uzywane i przez densyfikacje, i przez navigate_to."""
    OUT_OF_BOUNDS = "out_of_bounds"
    TARGET_OOB = "target_out_of_bounds"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    POSITION_STUCK = "position_stuck"
    STALE_MOVEMENT = "stale_movement"
    OSCILLATING = "oscillating"
    DISTANCE_REGRESSING = "distance_regressing"
    SHOP_VANISHED = "shop_vanished"
    SHOP_STALE = "shop_stale"
    SHOP_NO_LOCATION = "shop_no_location"
    TIMEOUT = "timeout"
    OCR_FROZEN = "ocr_frozen"


@dataclass(frozen=True, slots=True)
class NavigateResult:
    """Wynik nawigacji: sukces albo powod FAIL."""
    success: bool
    reason: str = ""           # "reached" | FailReason
    steps_taken: int = 0
    final_position: tuple[int, int] | None = None
    shop_found: bool = False


# A convenience success
REACHED = NavigateResult(True, reason="reached")


# ---------------------------------------------------------------------------
# A1 – Koperta swiata
# ---------------------------------------------------------------------------

def within_envelope(
    x: float, y: float,
    envelope: tuple[int, int, int, int],
) -> bool:
    """Czy punkt (x,y) w kopercie (x0,y0,x1,y1)? Inclusive."""
    x0, y0, x1, y1 = envelope
    return x0 <= x <= x1 and y0 <= y <= y1


def next_step_within_envelope(
    x: float, y: float,
    dx: float, dy: float,
    envelope: tuple[int, int, int, int],
) -> bool:
    """Czy nastepny krok (x+dx, y+dy) bedzie w kopercie?"""
    return within_envelope(x + dx, y + dy, envelope)


# ---------------------------------------------------------------------------
# A2 – Cel poza koperta
# ---------------------------------------------------------------------------

def target_valid(
    tx: float, ty: float,
    envelope: tuple[int, int, int, int],
) -> bool:
    """Czy cel nawigacji lezy w kopercie?"""
    return within_envelope(tx, ty, envelope)


# ---------------------------------------------------------------------------
# A3 – Prog dotarcia („jestem na miejscu")
# ---------------------------------------------------------------------------

def reached_target(
    current: tuple[float, float],
    target: tuple[float, float],
    eps: float = 3.0,
) -> bool:
    """Czy dystans do celu <= eps?"""
    cx, cy = current
    tx, ty = target
    return ((tx - cx) ** 2 + (ty - cy) ** 2) <= eps**2


# ---------------------------------------------------------------------------
# B1 – OCR frozen (ten sam odczyt mimo krokow)
# ---------------------------------------------------------------------------

def is_ocr_frozen(
    readings: list[tuple[float, float]],
    max_same: int = 5,
    eps: float = 2.0,
) -> bool:
    """Czy ostatnie `max_same` odczytow OCR to ta sama pozycja (±eps)?"""
    if len(readings) < max_same:
        return False
    recent = readings[-max_same:]
    x0, y0 = recent[0]
    return all(
        abs(x - x0) <= eps and abs(y - y0) <= eps
        for x, y in recent[1:]
    )


# ---------------------------------------------------------------------------
# B4 – Stuck (pozycja nie rusza sie mimo krokow)
# ---------------------------------------------------------------------------

def is_stuck(
    recent_positions: list[tuple[float, float]],
    max_same: int = 5,
    eps: float = 2.0,
) -> bool:
    """Czy ostatnie `max_same` pozycji (OCR lub DR) stoi w miejscu?"""
    return is_ocr_frozen(recent_positions, max_same, eps)


# ---------------------------------------------------------------------------
# C1 – Dystans rosnie zamiast malec
# ---------------------------------------------------------------------------

def distance_regressing(
    distances: list[float],
    regress_threshold: int = 3,
) -> bool:
    """Czy dystans rosnie `regress_threshold` razy z rzedu?"""
    if len(distances) < regress_threshold + 1:
        return False
    # sprawdz ostatnie N+1: kazda kolejna > poprzednia?
    recent = distances[-(regress_threshold + 1):]
    return all(recent[i] > recent[i - 1] for i in range(1, len(recent)))


# ---------------------------------------------------------------------------
# C2 – Oscylacja (zygzak w miejscu)
# ---------------------------------------------------------------------------

def is_oscillating(
    positions: list[tuple[float, float]],
    window: int = 5,
    max_amplitude: float = 10.0,
) -> bool:
    """Czy ostatnie `window` pozycji oscyluje w promieniu `max_amplitude`?"""
    if len(positions) < window:
        return False
    recent = positions[-window:]
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    return spread_x <= max_amplitude and spread_y <= max_amplitude


# ---------------------------------------------------------------------------
# C3 – Budzet krokow
# ---------------------------------------------------------------------------

def step_budget(
    target: tuple[float, float],
    current: tuple[float, float],
    units_per_step: float,
    margin: float = 2.0,
) -> int:
    """Maksymalna liczba krokow na dojscie do celu (z marginesem ×2)."""
    tx, ty = target
    cx, cy = current
    dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
    return max(1, round(dist / max(units_per_step, 0.1) * margin))


def budget_exceeded(steps_taken: int, budget: int) -> bool:
    return steps_taken > budget


# ---------------------------------------------------------------------------
# B5 – Stale movement (APPROACH_A_SPEC: stuck-w-teksturze, bez OCR)
# ---------------------------------------------------------------------------

def fix_stale(steps_since_last_fix: int, max_steps: int = 7) -> bool:
    """Czy przekroczono limit nieproduktywnych kroków od ostatniego fixu?

    Lapsuje stucka w teksturze gdzie is_stuck zawodzi (brak OCR, dead-reckon
    dryfuje mimo braku realnego ruchu). DeepSeek: po fix_stale=True →
    recovery: cofnij s, obróć a/d ~90°, wznow W, odswiez heading z OCR.
    """
    return steps_since_last_fix > max_steps


# ---------------------------------------------------------------------------
# Recovery plan (APPROACH_A_SPEC: co robic po guard triggerze)
# ---------------------------------------------------------------------------

def recovery_plan(attempt: int, back_steps: int = 3) -> list[tuple[str, float, float]]:
    """Zwraca sekwencje ruchow: cofnij s×N → obrot a/d → wznow w.

    attempt parzyste = obrot 'a', nieparzyste = 'd' (naprzemiennie).
    Kazdy element to (key, duration, settle).
    DeepSeek wykonuje po kolei przez movement.execute().
    """
    turn_key = "d" if attempt % 2 == 1 else "a"
    plan: list[tuple[str, float, float]] = []
    # Cofnij s × N
    for _ in range(back_steps):
        plan.append(("s", 0.6, 0.0))
    # Obrót ~90° (4× 0.2s = 0.8s)
    for _ in range(4):
        plan.append((turn_key, 0.2, 0.0))
    plan.append((turn_key, 0.2, 0.5))  # ostatni z settle
    # Wznów marsz – nic (caller woła normalny krok W)
    return plan


# ---------------------------------------------------------------------------
# Heading z OCR (Filar B: odswiezenie kierunku po skrecie)
# ---------------------------------------------------------------------------

def heading_from_ocr(
    p0: tuple[float, float],
    p1: tuple[float, float],
    magnitude: float = 3.2,
) -> tuple[float, float]:
    """Oblicza wektor kierunku W z dwoch kotwic OCR.

    p0, p1: pozycje (x,y) z dwoch kolejnych odczytow OCR po kroku W.
    magnitude: dlugosc wektora (z kalibracji Claude'a: ~3.2 u/krok).

    Zwraca (dx, dy) – wektor jednostkowy * magnitude.
    DeepSeek: Odometer.set_vector("w", heading_from_ocr(p0, p1)).
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = (dx**2 + dy**2) ** 0.5
    if dist < 0.1:
        return (0.0, -magnitude)  # fallback: domyslny W w gore
    return (dx / dist * magnitude, dy / dist * magnitude)
