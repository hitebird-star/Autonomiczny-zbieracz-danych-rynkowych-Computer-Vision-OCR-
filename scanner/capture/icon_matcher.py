"""Grupowanie zajętych slotów po wyglądzie środka ikony."""

from __future__ import annotations

import numpy as np
from PIL import Image

from scanner.capture.shop_capture import OccupiedSlot
from scanner.config import GridGeometry


ICON_PAD = 6
ICON_MATCH_THRESHOLD = 6.0


def icon_signature(
    grid_image: Image.Image,
    slot: OccupiedSlot,
    geometry: GridGeometry,
    *,
    pad: int = ICON_PAD,
) -> np.ndarray:
    """Wytnij środkowe 20×20 px ikony, bez wspólnej ramki slotu."""

    x = slot.column * geometry.cell + pad
    y = slot.row * geometry.cell + pad
    size = geometry.cell - 2 * pad
    return np.asarray(
        grid_image.crop((x, y, x + size, y + size)).convert("RGB"),
        dtype=np.int16,
    )


def icon_distance(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    return float(np.abs(first - second).mean())


def group_slots_by_icon(
    grid_image: Image.Image,
    slots: list[OccupiedSlot],
    geometry: GridGeometry,
    *,
    threshold: float = ICON_MATCH_THRESHOLD,
) -> list[list[OccupiedSlot]]:
    """Pogrupuj sloty; kolejność grup i elementów pozostaje deterministyczna."""

    groups: list[list[OccupiedSlot]] = []
    representatives: list[np.ndarray] = []
    for slot in slots:
        signature = icon_signature(grid_image, slot, geometry)
        best_index = -1
        best_distance = threshold
        for index, representative in enumerate(representatives):
            distance = icon_distance(signature, representative)
            if distance <= best_distance:
                best_index = index
                best_distance = distance
        if best_index >= 0:
            groups[best_index].append(slot)
        else:
            groups.append([slot])
            representatives.append(signature)
    return groups
