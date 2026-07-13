"""Rejestracja klatek dla Atlasa: globalna translacja ekranu przez korelację fazową.

Alternatywa dla dopasowania tracków sklepów (`match_shop_deltas`), które w regularnej
kracie identycznych straganów jest ZDEGENEROWANE — aliasing. Dowód z 2 ukończonych runów:
game-delty w/s są przeciwne (+x / −x), ale screen-tracki OBU lecą „w dół" (cos +0.96),
bo wysokie nogi straganów tworzą pionową periodyczność i matcher kolapsuje każde dopasowanie
do przeskoku o oczko kraty, niezależnie od prawdziwego ruchu.

Klucz: statyczne tło (ziemia + stragany) przesuwa się SZTYWNO z kamerą jednym wektorem, a
teksturowa ziemia jest APERIODYCZNA — łamie periodyczność kraty i daje jednoznaczny pik
korelacji. Ten moduł jest OFFLINE-CZYSTY: przyjmuje dwa obrazy (numpy/PIL) i zwraca (dx, dy)
+ pewność. Codex podaje klatki before/after z żywego okna; matematyka zostaje tu.

Konwencja: `dx>0` = treść w `after` przesunęła się w PRAWO względem `before`, `dy>0` = w DÓŁ
(tj. `after ≈ np.roll(before, (dy, dx), axis=(0,1))`).
"""

from __future__ import annotations

import numpy as np

# Domyślne marginesy UI do WYCIĘCIA przed korelacją (ułamki wymiaru klatki klienta).
# Statyczne UI (HUD/minimapa top-right, dolny pasek skilli) NIE przesuwa się z kamerą →
# daje konkurencyjny pik zero-shift, który zabija mały ruch świata. Layout Glevia2 1920×1080:
# HUD skupiony top-right, pasek na dole. Kadr zostawia lewe ~74% szer. i 5–93% wys.
DEFAULT_UI_MARGINS = (0.02, 0.05, 0.26, 0.07)  # (left, top, right, bottom)


def _to_gray(image) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    elif a.ndim != 2:
        raise ValueError(f"oczekiwano obrazu 2D/3D, dostałem shape={a.shape}")
    return a


def _parabolic(prev: float, mid: float, nxt: float) -> float:
    """Subpikselowe przesunięcie piku z 3 sąsiednich próbek (interpolacja paraboliczna)."""
    denom = prev - 2.0 * mid + nxt
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (prev - nxt) / denom


def estimate_screen_shift(
    before,
    after,
    *,
    max_shift_px: int | None = None,
    window: bool = True,
) -> tuple[float, float, float]:
    """Globalna translacja `after` względem `before` w pikselach: (dx, dy, confidence).

    Korelacja fazowa (znormalizowany cross-power → IFFT → pik). Okno Hanninga tłumi
    artefakty brzegowe. `max_shift_px` ogranicza szukanie piku do |przesunięcia| ≤ tej
    wartości (odrzuca aliasy o wiele oczek kraty dalej). `confidence` = wysokość piku
    korelacji (≈1 dla czystej translacji, niżej gdy scena wieloznaczna/zaszumiona).
    """
    a = _to_gray(before)
    b = _to_gray(after)
    if a.shape != b.shape:
        raise ValueError(f"klatki mają różne wymiary: {a.shape} vs {b.shape}")
    h, w = a.shape
    if h < 2 or w < 2:
        raise ValueError("klatki za małe do korelacji")

    if window:
        win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        a = a * win
        b = b * win

    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-12
    corr = np.fft.irfft2(cross, s=(h, w))

    if max_shift_px is not None:
        m = int(max_shift_px)
        allowed = np.full((h, w), False)
        m = max(0, min(m, h // 2, w // 2))
        allowed[: m + 1, : m + 1] = True
        allowed[: m + 1, w - m :] = True
        allowed[h - m :, : m + 1] = True
        allowed[h - m :, w - m :] = True
        search = np.where(allowed, corr, -np.inf)
    else:
        search = corr

    py, px = np.unravel_index(int(np.argmax(search)), search.shape)
    peak = float(corr[py, px])

    # subpiksel wokół piku (z zawijaniem indeksów)
    dy_sub = _parabolic(
        float(corr[(py - 1) % h, px]), peak, float(corr[(py + 1) % h, px])
    )
    dx_sub = _parabolic(
        float(corr[py, (px - 1) % w]), peak, float(corr[py, (px + 1) % w])
    )

    # przelicz indeks piku na przesunięcie ze znakiem (wrap wokół połowy wymiaru).
    # cross = FA·conj(FB) daje pik przy −shift, więc negujemy, by (dx,dy) zgadzało się
    # z konwencją `after ≈ np.roll(before, (dy, dx))`.
    dy = (py - h if py > h // 2 else py) + dy_sub
    dx = (px - w if px > w // 2 else px) + dx_sub

    confidence = float(max(0.0, min(1.0, peak)))
    return -float(dx), -float(dy), confidence


def crop_world_viewport(
    image,
    *,
    margins: tuple[float, float, float, float] = DEFAULT_UI_MARGINS,
) -> np.ndarray:
    """Wytnij pas świata gry, odcinając statyczne UI (marginesy jako ułamki wymiaru)."""
    a = np.asarray(image)
    h, w = a.shape[0], a.shape[1]
    left, top, right, bottom = margins
    x0 = int(round(w * left))
    x1 = int(round(w * (1.0 - right)))
    y0 = int(round(h * top))
    y1 = int(round(h * (1.0 - bottom)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("marginesy UI wycinają zbyt dużo — zostaje za mały kadr")
    return a[y0:y1, x0:x1]


def screen_shift_between_frames(
    before,
    after,
    *,
    max_shift_px: int = 80,
    margins: tuple[float, float, float, float] = DEFAULT_UI_MARGINS,
    window: bool = True,
) -> tuple[float, float, float]:
    """Klocek dla live-kalibratora: obraz before/after klienta → (dx, dy, confidence).

    Sam kadruje świat (odcina UI wg `margins`), potem korelacja fazowa. Codex woła to
    z `feed.grab_client_image()` i wkłada wynik w `MoveObservation(delta_screen=[(dx,dy)],
    confidence=conf)` — bez `match_shop_deltas`. Próguj `conf`: niski = niepewny ruch.
    """
    return estimate_screen_shift(
        crop_world_viewport(before, margins=margins),
        crop_world_viewport(after, margins=margins),
        max_shift_px=max_shift_px,
        window=window,
    )
