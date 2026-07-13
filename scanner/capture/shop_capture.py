"""Szybki zrzut całego sklepu i mapa zajętych slotów."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from scanner.config import GridGeometry
from scanner.runtime import Clock, InputBackend, ScreenBackend, SystemClock


@dataclass(frozen=True, slots=True)
class OccupiedSlot:
    slot: int
    row: int
    column: int
    residual: float


class ShopCapturer:
    def __init__(
        self,
        screen: ScreenBackend,
        input_backend: InputBackend,
        geometry: GridGeometry,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.screen = screen
        self.input = input_backend
        self.geometry = geometry
        self.clock = clock or SystemClock()

    def capture_shop(self) -> Image.Image:
        rest_x = self.geometry.origin[0] + self.geometry.shop_box[2] // 2
        rest_y = self.geometry.origin[1] - 25
        self.input.move_to(rest_x, rest_y, duration=0.05)
        self.clock.sleep(0.12)
        return self.screen.grab(self.geometry.shop_box)

    def capture_grid(self) -> Image.Image:
        return self.screen.grab(self.geometry.box)

    def occupied_slots(self, grid_image: Image.Image) -> list[OccupiedSlot]:
        rgb = np.asarray(grid_image.convert("RGB"))
        geometry = self.geometry
        expected = (
            geometry.rows * geometry.cell,
            geometry.columns * geometry.cell,
            3,
        )
        if rgb.shape != expected:
            raise ValueError(f"nieprawidłowy rozmiar siatki: {rgb.shape}, oczekiwano {expected}")

        inset = max(2, geometry.cell // 7)
        cells: list[np.ndarray] = []
        positions: list[tuple[int, int]] = []
        for row in range(geometry.rows):
            for column in range(geometry.columns):
                y0 = row * geometry.cell + inset
                y1 = (row + 1) * geometry.cell - inset
                x0 = column * geometry.cell + inset
                x1 = (column + 1) * geometry.cell - inset
                cells.append(rgb[y0:y1, x0:x1])
                positions.append((column, row))

        stack = np.stack(cells)
        gray = stack.mean(axis=3)
        means = gray.mean(axis=(1, 2))
        bright_share = (gray > 75).mean(axis=(1, 2))
        empty_ids = np.where((means < 40.0) & (bright_share < 0.16))[0]

        # Pełny sklep nie ma ciemnego pustego slotu. Wtedy wszystkie 100 pól
        # są zajęte; nie wolno tworzyć wzorca z najczęstszej ikony.
        if not len(empty_ids):
            return [
                OccupiedSlot(row * geometry.columns + column, row, column, 255.0)
                for column, row in positions
            ]

        template = np.median(stack[empty_ids], axis=0)
        residuals = np.mean(
            np.abs(stack.astype(np.float32) - template.astype(np.float32)),
            axis=(1, 2, 3),
        )
        result = []
        for (column, row), residual, mean, bright in zip(
            positions, residuals, means, bright_share
        ):
            is_empty_dark = mean < 40.0 and bright < 0.16
            if not is_empty_dark and residual >= geometry.occupancy_residual:
                result.append(
                    OccupiedSlot(
                        row * geometry.columns + column,
                        row,
                        column,
                        float(residual),
                    )
                )
        return result
