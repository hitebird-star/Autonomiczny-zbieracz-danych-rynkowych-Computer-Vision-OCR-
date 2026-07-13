"""Kalibracja siatki sklepu — czysty rdzeń (Claude, pure).

`calibrate` w CLI = odometria (units_per_step). Kalibracja OKNA/siatki sklepa
(shop_origin + cell + grid_dx/dy) była ręczna w configu; po przesunięciu okna
trzeba ją odświeżyć. Ten moduł daje narzędziu (komenda DeepSeeka, patrz
`WINDOW_CALIBRATION_AND_REGRESSION_FIX.md`) dwie czyste funkcje:

- `render_grid_overlay` — nakłada siatkę + środki klików na zrzut regionu siatki,
  żeby człowiek zobaczył, czy linie trafiają w ramki gry (zero gry/IO tutaj).
- `apply_calibration` — zwraca NOWY dict configu z podmienionymi polami kalibracji
  (czysta transformacja; zapis pliku robi warstwa live).

Zasada [[shop-reloop-livelock]]/[[mapping-rebuild-odometry]]: zła kalibracja = klik
obok komórki → pustka → brak popupa → ESC → sklep. Dlatego kalibracja MUSI być
weryfikowalna wizualnie, nie zgadywana.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

_RED = (255, 0, 0)
_GREEN = (0, 255, 0)


@dataclass(frozen=True, slots=True)
class WindowCalibration:
    origin: tuple[int, int]
    cell: int
    grid_dx: int
    grid_dy: int


def calibration_from_points(
    origin: tuple[int, int],
    slot00_center: tuple[int, int],
    slot10_center: tuple[int, int],
    *,
    min_cell: int = 8,
) -> WindowCalibration:
    """Policz kalibrację okna z trzech punktów złapanych kursorem.

    `origin` = lewy-górny róg okna sklepu.
    `slot00_center` = środek slotu (0,0).
    `slot10_center` = środek slotu (1,0).

    To jest czysty odpowiednik starego `shop_scanner.py calibrate`, ale bez
    zapisu całego starego schematu configu.
    """

    ox, oy = int(origin[0]), int(origin[1])
    s00x, s00y = int(slot00_center[0]), int(slot00_center[1])
    s10x, _ = int(slot10_center[0]), int(slot10_center[1])
    cell = abs(s10x - s00x)
    if cell < int(min_cell):
        raise ValueError(
            f"odległość slotów jest za mała ({cell}px); złap ponownie środki slotów"
        )
    return WindowCalibration(
        origin=(ox, oy),
        cell=cell,
        grid_dx=int(s00x - ox - cell // 2),
        grid_dy=int(s00y - oy - cell // 2),
    )


def render_grid_overlay(
    grid_image: Image.Image,
    cols: int,
    rows: int,
    cell: int,
    *,
    line_color: tuple[int, int, int] = _RED,
    dot_color: tuple[int, int, int] = _GREEN,
) -> Image.Image:
    """Zrzut regionu siatki (cols*cell × rows*cell) → kopia z nałożoną siatką.

    Czerwone linie = granice komórek wg configu. Zielone kropki = środki slotów
    (gdzie poleci klik/hover). Jeśli linie NIE pokrywają się z ramkami gry albo
    kropki nie siedzą na ikonach — `cell`/origin są źle skalibrowane.
    """

    if cols <= 0 or rows <= 0 or cell <= 0:
        raise ValueError("cols, rows, cell muszą być > 0")
    canvas = grid_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = cols * cell, rows * cell
    for r in range(rows + 1):
        draw.line([(0, r * cell), (w, r * cell)], fill=line_color, width=1)
    for c in range(cols + 1):
        draw.line([(c * cell, 0), (c * cell, h)], fill=line_color, width=1)
    for r in range(rows):
        for c in range(cols):
            cx, cy = c * cell + cell // 2, r * cell + cell // 2
            draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=dot_color)
    return canvas


def apply_calibration(
    config: dict[str, Any],
    *,
    origin: tuple[int, int] | None = None,
    cell: int | None = None,
    grid_dx: int | None = None,
    grid_dy: int | None = None,
) -> dict[str, Any]:
    """Zwróć NOWY dict configu z podmienioną kalibracją (nie mutuje wejścia).

    Podmienia tylko podane pola: `shop_origin` (top-level) oraz
    `grid.{cell,grid_dx,grid_dy}`. Reszta (cols/rows/occ_*/qty_*) nietknięta —
    to konwencja [[adoption-utf16-powershell]]: minimalna, jawna zmiana.
    """

    out = deepcopy(config)
    grid = out.setdefault("grid", {})
    if origin is not None:
        out["shop_origin"] = [int(origin[0]), int(origin[1])]
    if cell is not None:
        grid["cell"] = int(cell)
    if grid_dx is not None:
        grid["grid_dx"] = int(grid_dx)
    if grid_dy is not None:
        grid["grid_dy"] = int(grid_dy)
    return out
