"""Konfiguracja nowego potoku bez zależności od monolitu legacy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GridGeometry:
    origin: tuple[int, int] = (0, 0)
    offset: tuple[int, int] = (4, 39)
    cell: int = 32
    columns: int = 10
    rows: int = 10
    occupancy_residual: float = 3.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (
            self.origin[0] + self.offset[0],
            self.origin[1] + self.offset[1],
            self.columns * self.cell,
            self.rows * self.cell,
        )

    @property
    def shop_box(self) -> tuple[int, int, int, int]:
        return (
            self.origin[0],
            self.origin[1],
            self.offset[0] + self.columns * self.cell + 12,
            self.offset[1] + self.rows * self.cell + 42,
        )

    def slot_box(self, column: int, row: int) -> tuple[int, int, int, int]:
        return (
            self.origin[0] + self.offset[0] + column * self.cell,
            self.origin[1] + self.offset[1] + row * self.cell,
            self.cell,
            self.cell,
        )

    def slot_center(self, column: int, row: int) -> tuple[int, int]:
        x, y, width, height = self.slot_box(column, row)
        return x + width // 2, y + height // 2


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    hover_delay: float = 0.12
    move_duration: float = 0.05
    frames_per_slot: int = 1
    frame_interval: float = 0.035
    tooltip_timeout: float = 0.9
    tooltip_poll_interval: float = 0.05
    tooltip_diff_threshold: int = 18
    tooltip_min_area: int = 4_000
    hover_attempts: int = 3
    first_pass_hover_attempts: int = 2
    hover_retry_delay: float = 0.08
    cursor_tolerance: int = 4
    tooltip_width: int = 620
    tooltip_height: int = 620
    cursor_rest_offset: tuple[int, int] = (160, -30)


@dataclass(frozen=True, slots=True)
class DetectorSettings:
    r_min: int = 80
    r_max: int = 210
    rg_min: int = 25
    gb_min: int = 5
    rb_min: int = 45
    area_min: int = 12
    area_max: int = 200
    width_min: int = 5
    width_max: int = 24
    height_min: int = 4
    height_max: int = 22
    merge_distance: int = 22
    click_offset_y: int = 6
    min_radius: int = 90
    max_radius: int = 240
    max_results: int = 12
    skip_boxes: tuple[tuple[int, int, int, int], ...] = ()
    hybrid_enabled: bool = True
    hybrid_model_path: str = "scanner/detection/shop_target_svm.npz"
    hybrid_metadata_path: str = "scanner/detection/shop_target_svm.json"


@dataclass(frozen=True, slots=True)
class ScannerSettings:
    window_title: str = "Glevia2"
    grid: GridGeometry = field(default_factory=GridGeometry)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    close_key: str = "esc"
    open_timeout: float = 4.0
    retry_open_timeout: float = 1.0
    source: str = "scanner-v10"


def _tuple(value: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)) and len(value) == len(fallback):
        return tuple(int(part) for part in value)
    return fallback


def load_settings(path: str | Path = "scanner_config.json") -> ScannerSettings:
    """Wczytaj istniejącą kalibrację bez importowania ``shop_scanner.py``."""

    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    grid_data = data.get("grid") or {}
    timing = data.get("timing") or {}
    capture_data = data.get("capture_v10") or {}
    color = data.get("color") or {}

    grid = GridGeometry(
        origin=_tuple(data.get("shop_origin"), (0, 0)),
        offset=(
            int(grid_data.get("grid_dx", 4)),
            int(grid_data.get("grid_dy", 39)),
        ),
        # Okno sklepu Glevii ma stałą siatkę 10×10 po 32 px. Kalibracja
        # wykonywana kursorem potrafi wyliczyć 30/31 przez niedokładne trafienie
        # w środek slotu, co rozjeżdża całą mapę zajętości.
        cell=32,
        columns=int(grid_data.get("cols", 10)),
        rows=int(grid_data.get("rows", 10)),
        occupancy_residual=float(grid_data.get("occ_residual", 3.0)),
    )
    capture = CaptureSettings(
        # Timingi legacy obejmują OCR wykonywany przy każdym slocie. Capture v10
        # tylko zapisuje klatki, więc ma własną, znacznie szybszą konfigurację.
        hover_delay=float(capture_data.get("hover_delay", 0.14)),
        move_duration=float(capture_data.get("move_duration", 0.05)),
        frames_per_slot=int(capture_data.get("frames_per_slot", 1)),
        frame_interval=float(capture_data.get("frame_interval", 0.035)),
        tooltip_timeout=float(capture_data.get("tooltip_timeout", 0.9)),
        tooltip_poll_interval=float(
            capture_data.get("tooltip_poll_interval", 0.05)
        ),
        tooltip_diff_threshold=int(
            capture_data.get("tooltip_diff_threshold", 18)
        ),
        tooltip_min_area=int(capture_data.get("tooltip_min_area", 4_000)),
        hover_attempts=max(1, int(capture_data.get("hover_attempts", 3))),
        first_pass_hover_attempts=max(
            1,
            int(capture_data.get("first_pass_hover_attempts", 2)),
        ),
        hover_retry_delay=float(capture_data.get("hover_retry_delay", 0.08)),
        cursor_tolerance=max(
            1, int(capture_data.get("cursor_tolerance", 4))
        ),
        tooltip_width=int(capture_data.get("tooltip_width", 620)),
        tooltip_height=int(capture_data.get("tooltip_height", 620)),
    )
    detector = DetectorSettings(
        r_min=int(color.get("r_min", 80)),
        r_max=int(color.get("r_max", 210)),
        rg_min=int(color.get("rg_min", 25)),
        gb_min=int(color.get("gb_min", 5)),
        rb_min=int(color.get("rb_min", 45)),
        area_min=int(color.get("area_min", 12)),
        area_max=int(color.get("area_max", 200)),
        width_min=int(color.get("w_min", 5)),
        width_max=int(color.get("w_max", 24)),
        height_min=int(color.get("h_min", 4)),
        height_max=int(color.get("h_max", 22)),
        merge_distance=int(color.get("merge_dist", 22)),
        click_offset_y=int(color.get("click_dy", 6)),
        min_radius=int(color.get("min_radius", 90)),
        max_radius=int(color.get("max_radius", 240)),
        max_results=int(color.get("max_results", 12)),
        skip_boxes=tuple(tuple(int(v) for v in box) for box in color.get("skip_boxes", [])),
        hybrid_enabled=bool(color.get("hybrid_enabled", True)),
        hybrid_model_path=str(
            color.get(
                "hybrid_model_path",
                "scanner/detection/shop_target_svm.npz",
            )
        ),
        hybrid_metadata_path=str(
            color.get(
                "hybrid_metadata_path",
                "scanner/detection/shop_target_svm.json",
            )
        ),
    )
    return ScannerSettings(
        window_title=str(data.get("window_title", "Glevia2")),
        grid=grid,
        capture=capture,
        detector=detector,
        close_key=str(data.get("close_key", "esc")),
        open_timeout=float(timing.get("open_timeout", 4.0)),
        retry_open_timeout=max(
            0.1, float(timing.get("retry_open_timeout", 1.0))
        ),
        source=str(data.get("source", "scanner-v10")),
    )
