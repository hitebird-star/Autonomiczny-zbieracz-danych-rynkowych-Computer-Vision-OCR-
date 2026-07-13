"""Detekcja straganów po kolorze — port sprawdzonej logiki legacy."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from scanner.config import DetectorSettings
from scanner.detection.target_verifier import ShopTargetVerifier


@dataclass(frozen=True, slots=True)
class ShopCandidate:
    screen_position: tuple[int, int]
    local_position: tuple[int, int]
    area: int
    distance: float
    hybrid_score: float | None = None
    likely_false: bool = False


class ShopDetector:
    def __init__(
        self,
        settings: DetectorSettings,
        *,
        verifier: ShopTargetVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.verifier = verifier
        if verifier is None and settings.hybrid_enabled:
            self.verifier = ShopTargetVerifier.load(
                settings.hybrid_model_path,
                settings.hybrid_metadata_path,
            )

    def detect(
        self,
        image: Image.Image,
        *,
        screen_offset: tuple[int, int] = (0, 0),
    ) -> list[ShopCandidate]:
        mask = self.mask_array(image)
        cfg = self.settings
        candidates = self._candidates_from_mask(
            image,
            mask,
            screen_offset=screen_offset,
            relaxed=False,
        )
        # W gęstszych/ukośnych rejonach farmy wierzch straganu bywa szerszy
        # niż legacy 24x22. Wtedy maska widzi sklepy, ale filtr kształtu zostawia
        # 0-2 kandydatów mimo pełnego ekranu straganów. Fallback poluzowuje tylko
        # geometrię komponentu i odpala się dopiero przy niedoborze celów.
        target_min = min(3, cfg.max_results) if cfg.max_results > 0 else 3
        if len(candidates) < target_min:
            relaxed = self._candidates_from_mask(
                image,
                mask,
                screen_offset=screen_offset,
                relaxed=True,
            )
            merge_distance_squared = cfg.merge_distance**2
            for candidate in relaxed:
                if all(
                    (
                        candidate.local_position[0] - existing.local_position[0]
                    ) ** 2
                    + (
                        candidate.local_position[1] - existing.local_position[1]
                    ) ** 2
                    > merge_distance_squared
                    for existing in candidates
                ):
                    candidates.append(candidate)
        # Zachowujemy dokladnie ten sam zbior najblizszych kandydatow co
        # detektor legacy, gdy legacy widzi dość celów. Fallback dopisuje
        # kandydatów tylko w trybie "za mało celów".
        candidates.sort(key=lambda candidate: candidate.distance)
        if cfg.max_results > 0:
            candidates = candidates[: cfg.max_results]
        if self.verifier is not None:
            candidates = [
                ShopCandidate(
                    candidate.screen_position,
                    candidate.local_position,
                    candidate.area,
                    candidate.distance,
                    assessment.score,
                    assessment.likely_false,
                )
                for candidate in candidates
                for assessment in [
                    self.verifier.assess(image, candidate.local_position)
                ]
            ]
        # K1: model tylko odsuwa podejrzane cele. Nie usuwa ich, wiec slaby
        # klasyfikator nie moze trwale zgubic prawdziwego sklepu.
        candidates.sort(
            key=lambda candidate: (
                candidate.likely_false,
                candidate.distance,
            )
        )
        return candidates

    def _candidates_from_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        *,
        screen_offset: tuple[int, int],
        relaxed: bool,
    ) -> list[ShopCandidate]:
        cfg = self.settings
        count, _, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        raw: list[tuple[int, int, int]] = []
        for index in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[index])
            center_x, center_y = (int(v) for v in centers[index])
            area_max = max(cfg.area_max, 500) if relaxed else cfg.area_max
            width_max = max(cfg.width_max, 45) if relaxed else cfg.width_max
            height_max = max(cfg.height_max, 35) if relaxed else cfg.height_max
            ratio_max = 4 if relaxed else 3
            if not cfg.area_min <= area <= area_max:
                continue
            if not (
                cfg.width_min <= width <= width_max
                and cfg.height_min <= height <= height_max
            ):
                continue
            if (
                width / max(1, height) > ratio_max
                or height / max(1, width) > ratio_max
            ):
                continue
            if self._skipped(center_x, center_y):
                continue
            raw.append((center_x, center_y, area))

        raw.sort(key=lambda item: -item[2])
        merged: list[tuple[int, int, int]] = []
        distance_squared = cfg.merge_distance**2
        for x, y, area in raw:
            if all(
                (x - previous_x) ** 2 + (y - previous_y) ** 2 > distance_squared
                for previous_x, previous_y, _ in merged
            ):
                merged.append((x, y, area))

        center_x, center_y = image.width / 2, image.height / 2
        candidates = []
        for x, y, area in merged:
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            # Postać i broń są stale blisko środka ekranu i potrafią spełnić
            # tę samą regułę brązowego koloru co stragan.
            if cfg.min_radius > 0 and distance < cfg.min_radius:
                continue
            if cfg.max_radius > 0 and distance > cfg.max_radius:
                continue
            local = (x, y + cfg.click_offset_y)
            candidates.append(
                ShopCandidate(
                    (local[0] + screen_offset[0], local[1] + screen_offset[1]),
                    local,
                    area,
                    distance,
                )
            )
        candidates.sort(key=lambda candidate: candidate.distance)
        return candidates

    def mask_array(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        cfg = self.settings
        mask = (
            (red > cfg.r_min)
            & (red < cfg.r_max)
            & (red - green > cfg.rg_min)
            & (green - blue > cfg.gb_min)
            & (red - blue > cfg.rb_min)
        ).astype(np.uint8)
        return cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

    def mask_image(self, image: Image.Image) -> Image.Image:
        return Image.fromarray(self.mask_array(image) * 255, mode="L")

    def _skipped(self, x: int, y: int) -> bool:
        return any(
            x0 <= x <= x1 and y0 <= y <= y1
            for x0, y0, x1, y1 in self.settings.skip_boxes
        )
