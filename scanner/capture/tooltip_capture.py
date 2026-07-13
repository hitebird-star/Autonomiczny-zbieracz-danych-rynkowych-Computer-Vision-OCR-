"""Przechwytywanie surowych klatek dymków — bez OCR i bez Ollamy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image

from scanner.capture.shop_capture import OccupiedSlot
from scanner.config import CaptureSettings, GridGeometry
from scanner.runtime import (
    Clock,
    FocusBackend,
    InputBackend,
    ScreenBackend,
    SystemClock,
)


@dataclass(frozen=True, slots=True)
class TooltipFrames:
    slot: OccupiedSlot
    frames: tuple[Image.Image, ...]
    matched_reference: bool = False
    baseline: Image.Image | None = None     # surowa ramka sprzed hovera (diagnostyka)
    last_candidate: Image.Image | None = None  # ostatnia ramka hovera (diagnostyka)


class TooltipCapturer:
    def __init__(
        self,
        screen: ScreenBackend,
        input_backend: InputBackend,
        geometry: GridGeometry,
        settings: CaptureSettings,
        *,
        clock: Clock | None = None,
        bounds: tuple[int, int, int, int] | None = None,
        focus: FocusBackend | None = None,
        focus_timeout: float = 30.0,
        marker_recognizer: Callable[[Image.Image], list[dict]] | None = None,
    ) -> None:
        self.screen = screen
        self.input = input_backend
        self.geometry = geometry
        self.settings = settings
        self.clock = clock or SystemClock()
        self.bounds = bounds
        self.focus = focus
        self.focus_timeout = focus_timeout
        self._marker_recognizer = marker_recognizer

    def _ensure_focus(self) -> None:
        if self.focus is None or self.focus.is_foreground():
            return
        print(
            "\n  PAUZA: Glevia2 straciła fokus. "
            "Kliknij okno gry — nie skanuję kolejnych slotów."
        )
        if not self.focus.wait_until_foreground(self.focus_timeout):
            raise RuntimeError("game_focus_lost")
        # Gra potrzebuje krótkiej chwili po odzyskaniu fokusu, zanim pokaże dymek.
        self.clock.sleep(0.15)

    def _frame_box(self, center: tuple[int, int]) -> tuple[int, int, int, int]:
        width = self.settings.tooltip_width
        height = self.settings.tooltip_height
        x = center[0] - width // 2
        y = center[1] - height // 2
        if self.bounds is not None:
            left, top, bound_width, bound_height = self.bounds
            width = min(width, bound_width)
            height = min(height, bound_height)
            x = max(left, min(x, left + bound_width - width))
            y = max(top, min(y, top + bound_height - height))
        else:
            x = max(0, x)
            y = max(0, y)
        return int(x), int(y), int(width), int(height)

    def _rest_point(self) -> tuple[int, int]:
        # Pasek tytułowy sklepu nie wywołuje dymka przedmiotu.
        return (
            self.geometry.origin[0]
            + self.geometry.offset[0]
            + self.geometry.columns * self.geometry.cell // 2,
            self.geometry.origin[1] + 16,
        )

    def _validate_point(self, point: tuple[int, int], label: str) -> None:
        if self.bounds is None:
            if point[0] < 0 or point[1] < 0:
                raise RuntimeError(f"{label}_outside_screen:{point}")
            return
        left, top, width, height = self.bounds
        if not (
            left <= point[0] < left + width
            and top <= point[1] < top + height
        ):
            raise RuntimeError(
                f"{label}_outside_game:{point}; bounds={self.bounds}; "
                f"origin={self.geometry.origin}"
            )

    def _move_and_verify(
        self, point: tuple[int, int], *, duration: float
    ) -> bool:
        self._validate_point(point, "cursor_target")
        self.input.move_to(*point, duration=duration)
        # Glevia korzysta z DirectInput. GetCursorPos może nadal raportować
        # pozycję pulpitu, mimo że kursor gry trafił w slot. Działający skaner
        # legacy nie blokował ruchu na podstawie tej wartości.
        return True

    def _baseline(self, box: tuple[int, int, int, int]) -> Image.Image:
        """Zapisz świeże, współpołożone tło dla bieżącego slotu."""

        rest = self._rest_point()
        self._move_and_verify(
            rest, duration=max(self.settings.move_duration, 0.08)
        )
        self.clock.sleep(0.1)
        baseline = self.screen.grab(box)
        self.clock.sleep(0.04)
        return self.screen.grab(box) if baseline else baseline

    def _tooltip_bbox(
        self, baseline: Image.Image, current: Image.Image
    ) -> tuple[int, int, int, int] | None:
        before = np.asarray(baseline.convert("RGB"), dtype=np.int16)
        after = np.asarray(current.convert("RGB"), dtype=np.int16)
        if before.shape != after.shape:
            return None
        difference = np.max(np.abs(after - before), axis=2).astype(np.uint8)
        mask = (
            difference > self.settings.tooltip_diff_threshold
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)
        )
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        candidates: list[tuple[int, int, int, int, int]] = []
        minimum_area = min(
            self.settings.tooltip_min_area,
            max(100, int(current.width * current.height * 0.15)),
        )
        minimum_width = min(180, max(30, current.width // 2))
        minimum_height = min(80, max(20, current.height // 3))
        current_gray = cv2.cvtColor(
            np.asarray(current.convert("RGB")), cv2.COLOR_RGB2GRAY
        )
        for index in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[index])
            if area < minimum_area:
                continue
            if not (
                minimum_width <= width <= min(500, current.width)
                and minimum_height <= height <= min(420, current.height)
            ):
                continue
            if width * height <= 0 or area / (width * height) < 0.08:
                continue
            panel = current_gray[y : y + height, x : x + width]
            if panel.size == 0:
                continue
            dark_share = float((panel < 55).mean())
            bright_share = float((panel > 150).mean())
            # Tooltip Glevii jest ciemnym, prawie jednolitym panelem z wieloma
            # jasnymi liniami tekstu. Poruszające się postacie mają odwrotne
            # proporcje i nie mogą zostać uznane za dymek.
            if dark_share < 0.55 or bright_share < 0.015:
                continue
            candidates.append((x, y, width, height, area))
        if not candidates:
            return None
        x, y, width, height, _ = max(candidates, key=lambda item: item[4])
        padding = 10
        return (
            max(0, x - padding),
            max(0, y - padding),
            min(current.width, x + width + padding),
            min(current.height, y + height + padding),
        )

    @staticmethod
    def _normalize_marker(text: str) -> str:
        return re.sub(r"[^a-z]", "", (text or "").casefold())

    def _has_sales_marker(self, image: Image.Image) -> bool:
        recognizer = self._marker_recognizer
        if recognizer is None:
            try:
                import win_ocr

                if not getattr(win_ocr, "AVAILABLE", False):
                    return False
                recognizer = win_ocr.recognize
            except Exception:
                return False
        scaled = image.resize(
            (image.width * 2, image.height * 2), Image.Resampling.LANCZOS
        )
        try:
            lines = recognizer(scaled)
        except Exception:
            return False
        from scanner.analysis.sale_marker import has_sale_marker

        if has_sale_marker(str(line.get("text") or "") for line in lines):
            return True
        return False

    @staticmethod
    def _matches_reference(
        reference: Image.Image, candidate: Image.Image
    ) -> bool:
        if reference.size != candidate.size:
            return False
        first = np.asarray(reference.convert("RGB"))
        second = np.asarray(candidate.convert("RGB"))
        # Fast-path obowiązuje wyłącznie dla obrazu piksel-identycznego.
        # Zmiana choćby jednego piksela (np. cyfry ceny) uruchamia pełny odczyt.
        return bool(np.array_equal(first, second))

    def capture_stack_member(
        self, slot: OccupiedSlot, reference_frame: Image.Image
    ) -> TooltipFrames:
        return self._capture(slot, reference_frame=reference_frame)

    def capture_stack_member_fast(
        self, slot: OccupiedSlot, reference_frame: Image.Image
    ) -> TooltipFrames:
        return self._capture(
            slot,
            reference_frame=reference_frame,
            max_attempts=self.settings.first_pass_hover_attempts,
        )

    def capture(self, slot: OccupiedSlot) -> TooltipFrames:
        return self._capture(slot)

    def capture_fast(self, slot: OccupiedSlot) -> TooltipFrames:
        return self._capture(
            slot,
            max_attempts=self.settings.first_pass_hover_attempts,
        )

    def _capture(
        self,
        slot: OccupiedSlot,
        *,
        reference_frame: Image.Image | None = None,
        max_attempts: int | None = None,
    ) -> TooltipFrames:
        self._ensure_focus()
        center = self.geometry.slot_center(slot.column, slot.row)
        self._validate_point(center, "slot_center")
        box = self._frame_box(center)

        first: Image.Image | None = None
        last_candidate = None
        bbox: tuple[int, int, int, int] | None = None
        baseline: Image.Image | None = None
        attempts = min(
            self.settings.hover_attempts,
            max_attempts
            if max_attempts is not None
            else self.settings.hover_attempts,
        )
        attempts = max(1, attempts)
        for attempt in range(1, attempts + 1):
            self._ensure_focus()
            baseline = self._baseline(box)
            self._ensure_focus()
            if not self._move_and_verify(
                center, duration=max(self.settings.move_duration, 0.1)
            ):
                if attempt < attempts:
                    self.clock.sleep(self.settings.hover_retry_delay)
                    continue
                raise RuntimeError(
                    f"cursor_missed_slot: target={center}, "
                    f"actual={self.input.position()}"
                )

            # Metin2/Glevia często ignoruje samo ustawienie absolutnej pozycji.
            # Względny ruch Win32 wymusza zdarzenie hover w DirectInput.
            self.input.nudge(3 + attempt)
            self.clock.sleep(
                self.settings.hover_delay
                + (attempt - 1) * self.settings.hover_retry_delay
            )
            self._ensure_focus()

            deadline = self.clock.monotonic() + self.settings.tooltip_timeout
            while self.clock.monotonic() < deadline:
                self._ensure_focus()
                candidate = self.screen.grab(box)
                last_candidate = candidate
                candidate_bbox = self._tooltip_bbox(baseline, candidate)
                if candidate_bbox is None:
                    self.clock.sleep(self.settings.tooltip_poll_interval)
                    continue
                cropped = candidate.crop(candidate_bbox)
                if reference_frame is not None and self._matches_reference(
                    reference_frame, cropped
                ):
                    return TooltipFrames(slot, (cropped,), matched_reference=True)
                # OCR markera jest relatywnie drogi. Uruchamiamy go dopiero,
                # gdy różnica obrazu faktycznie przypomina panel dymka.
                if not self._has_sales_marker(cropped):
                    self.clock.sleep(self.settings.tooltip_poll_interval)
                    continue
                bbox = candidate_bbox
                first = cropped
                break

            if first is not None:
                break
            if attempt < attempts:
                self.clock.sleep(self.settings.tooltip_poll_interval)

        if first is None or bbox is None or baseline is None:
            return TooltipFrames(slot, (), baseline=baseline, last_candidate=last_candidate)

        frames = [first]
        for _ in range(1, self.settings.frames_per_slot):
            self.clock.sleep(self.settings.frame_interval)
            self._ensure_focus()
            current = self.screen.grab(box)
            current_bbox = self._tooltip_bbox(baseline, current)
            if current_bbox is not None and not self._has_sales_marker(
                current.crop(current_bbox)
            ):
                current_bbox = None
            if current_bbox is None:
                # Ponowne względne drgnięcie zamiast zapisywania sceny bez dymka.
                self.input.nudge(4)
                self.clock.sleep(self.settings.tooltip_poll_interval)
                current = self.screen.grab(box)
                current_bbox = self._tooltip_bbox(baseline, current)
                if current_bbox is not None and not self._has_sales_marker(
                    current.crop(current_bbox)
                ):
                    current_bbox = None
            if current_bbox is not None:
                frames.append(current.crop(current_bbox))
        return TooltipFrames(slot, tuple(frames))
