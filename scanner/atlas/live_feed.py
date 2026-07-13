"""Live bridge Atlasa: zrzut klienta + coord OCR + detekcja sklepów.

Moduł jest osobny od głównej pętli skanera. Niczego nie klika i nie zapisuje w
pipeline — tylko produkuje `FrameSnapshot` dla przyszłej mapy/UI.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from collections import Counter
import re

from scanner.analysis import coord_reader
from scanner.atlas.contracts import FrameSnapshot, ShopScreenObservation
from scanner.detection import ShopDetector
from scanner.runtime import GameWindow, ScreenBackend


@dataclass(frozen=True, slots=True)
class LiveFeedOptions:
    coord_bounds: tuple[int, int] | tuple[int, int, int, int] = coord_reader.DEFAULT_BOUNDS
    coord_max_jump: int = coord_reader.MAX_JUMP
    use_startup_fallback: bool = True


ATLAS_COORD_FALLBACK_ATTEMPTS = (
    *coord_reader.STARTUP_FALLBACK_ATTEMPTS,
    ((0.90, 0.155, 1.0, 0.195), 5, 190),
    ((0.90, 0.155, 1.0, 0.195), 5, "white_outline"),
    ((0.90, 0.155, 1.0, 0.195), 5, "white"),
    ((0.88, 0.150, 1.0, 0.200), 5, 190),
    ((0.88, 0.150, 1.0, 0.200), 5, "white_outline"),
    ((0.88, 0.150, 1.0, 0.200), 5, "white"),
)


class AtlasLiveFeed:
    """Produkuje snapshoty dla Atlasa z żywego okna Glevia2."""

    def __init__(
        self,
        *,
        screen: ScreenBackend,
        window: GameWindow,
        detector: ShopDetector,
        options: LiveFeedOptions | None = None,
    ) -> None:
        self.screen = screen
        self.window = window
        self.detector = detector
        self.options = options or LiveFeedOptions()
        self._last_position: tuple[int, int] | None = None
        self.last_client_image = None
        self.last_window_rect: tuple[int, int, int, int] | None = None
        self.last_snapshot: FrameSnapshot | None = None
        self.last_coord_trace: dict | None = None

    def capture_once(self) -> FrameSnapshot:
        rect, image = self.grab_client_image()
        self.last_client_image = image.copy()
        self.last_window_rect = (rect.x, rect.y, rect.width, rect.height)
        parsed, trace = _read_coord_for_atlas_with_trace(
            image,
            self.options,
            previous=self._last_position,
        )
        self.last_coord_trace = trace
        player_game = None
        if parsed is not None:
            candidate = (parsed.x, parsed.y)
            accepted, reason = _accept_reading_with_reason(
                self._last_position,
                candidate,
                bounds=self.options.coord_bounds,
                max_jump=self.options.coord_max_jump,
            )
            trace["final_accept"] = {
                "previous": self._last_position,
                "candidate": candidate,
                "accepted": accepted,
                "reason": reason,
            }
            if accepted:
                self._last_position = candidate
                player_game = (float(candidate[0]), float(candidate[1]))
        else:
            trace["final_accept"] = {
                "previous": self._last_position,
                "candidate": None,
                "accepted": False,
                "reason": "no_parsed_coord",
            }

        candidates = self.detector.detect(image, screen_offset=(rect.x, rect.y))
        observations = tuple(
            ShopScreenObservation(
                local_position=(
                    float(candidate.local_position[0]),
                    float(candidate.local_position[1]),
                ),
                screen_position=(
                    float(candidate.screen_position[0]),
                    float(candidate.screen_position[1]),
                ),
                area=int(candidate.area),
                distance=float(candidate.distance),
                hybrid_score=(
                    None
                    if candidate.hybrid_score is None
                    else float(candidate.hybrid_score)
                ),
                likely_false=bool(candidate.likely_false),
            )
            for candidate in candidates
        )
        snapshot = FrameSnapshot(
            timestamp=_utc_now_iso(),
            window_rect=(rect.x, rect.y, rect.width, rect.height),
            player_game=player_game,
            shops_screen=tuple(item.local_position for item in observations),
            shop_observations=observations,
        )
        self.last_snapshot = snapshot
        return snapshot

    def grab_client_image(self):
        rect = self.window.locate()
        return rect, self.screen.grab(rect.box)

    def snapshots(
        self,
        *,
        count: int | None = None,
        interval_s: float = 0.25,
    ) -> Iterator[FrameSnapshot]:
        emitted = 0
        while count is None or emitted < count:
            yield self.capture_once()
            emitted += 1
            if count is None or emitted < count:
                time.sleep(max(0.0, interval_s))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_coord_for_atlas_with_trace(image, options: LiveFeedOptions, *, previous=None):
    trace = {
        "bounds": options.coord_bounds,
        "max_jump": options.coord_max_jump,
        "previous": previous,
        "stages": [],
    }
    parsed = _read_coord_attempts_for_atlas(
        image,
        coord_reader.DEFAULT_ATTEMPTS,
        options.coord_bounds,
        trace=trace,
        stage="default",
    )
    if parsed is None and options.use_startup_fallback:
        parsed = _read_coord_attempts_for_atlas(
            image,
            ATLAS_COORD_FALLBACK_ATTEMPTS,
            options.coord_bounds,
            trace=trace,
            stage="atlas_fallback",
        )
    if parsed is None and options.use_startup_fallback:
        parsed = _salvage_coord_from_image(image, options.coord_bounds, trace=trace)
    trace["parsed"] = None if parsed is None else {
        "x": parsed.x,
        "y": parsed.y,
        "channel": parsed.channel,
        "map_name": parsed.map_name,
    }
    return parsed, trace


def _read_coord_for_atlas(image, options: LiveFeedOptions):
    parsed, _ = _read_coord_for_atlas_with_trace(image, options)
    return parsed


def _read_coord_attempts_for_atlas(image, attempts, bounds, *, trace: dict | None = None, stage: str = "attempt"):
    for roi, scale, binarize in attempts:
        text = coord_reader._ocr_text(image, roi, scale, binarize=binarize)
        parsed = coord_reader.parse_coord_text(text)
        entry = {
            "stage": stage,
            "roi": roi,
            "scale": scale,
            "binarize": binarize,
            "text": text,
            "parsed": None if parsed is None else {
                "x": parsed.x,
                "y": parsed.y,
                "channel": parsed.channel,
                "map_name": parsed.map_name,
            },
        }
        if parsed is None:
            entry["accepted"] = False
            entry["reason"] = "parse_failed"
            if trace is not None:
                trace["stages"].append(entry)
            continue
        accepted, reason = _accept_reading_with_reason(None, (parsed.x, parsed.y), bounds=bounds)
        entry["accepted"] = accepted
        entry["reason"] = reason
        if trace is not None:
            trace["stages"].append(entry)
        if accepted:
            return parsed
    return None


def _salvage_coord_from_image(image, bounds, *, trace: dict | None = None):
    texts: list[str] = []
    attempts = (*coord_reader.DEFAULT_ATTEMPTS, *ATLAS_COORD_FALLBACK_ATTEMPTS)
    for roi, scale, binarize in attempts:
        try:
            text = coord_reader._ocr_text(image, roi, scale, binarize=binarize)
            texts.append(text)
            if trace is not None:
                trace["stages"].append(
                    {
                        "stage": "salvage_text",
                        "roi": roi,
                        "scale": scale,
                        "binarize": binarize,
                        "text": text,
                    }
                )
        except Exception:
            continue
    parsed = _salvage_coord_from_texts(texts, bounds)
    if trace is not None:
        trace["salvage"] = None if parsed is None else {
            "x": parsed.x,
            "y": parsed.y,
            "channel": parsed.channel,
            "map_name": parsed.map_name,
        }
    return parsed


def _accept_reading_with_reason(
    previous: tuple[int, int] | None,
    current: tuple[int, int],
    *,
    bounds,
    max_jump: int = coord_reader.MAX_JUMP,
) -> tuple[bool, str]:
    if len(bounds) == 4:
        x_min, x_max, y_min, y_max = bounds
        if not (x_min <= current[0] <= x_max):
            return False, "x_out_of_bounds"
        if not (y_min <= current[1] <= y_max):
            return False, "y_out_of_bounds"
    else:
        if not coord_reader.in_bounds(current[0], bounds):
            return False, "x_out_of_bounds"
        if not coord_reader.in_bounds(current[1], bounds):
            return False, "y_out_of_bounds"
    if not coord_reader.plausible_jump(previous, current, max_jump):
        return False, "jump_too_large"
    return True, "accepted"


def _salvage_coord_from_texts(texts: list[str], bounds):
    x_bounds, y_bounds = _split_bounds(bounds)
    x_candidates: Counter[int] = Counter()
    y_candidates: Counter[int] = Counter()
    for text in texts:
        for number in _numeric_windows(_normalize_ocr_digits(text)):
            if x_bounds[0] <= number <= x_bounds[1]:
                x_candidates[number] += 1
            if y_bounds[0] <= number <= y_bounds[1]:
                y_candidates[number] += 1
    if not x_candidates or not y_candidates:
        return None
    x = _best_candidate(x_candidates, x_bounds)
    y = _best_candidate(y_candidates, y_bounds)
    return coord_reader.CoordParse(x=x, y=y, channel=None, map_name=None)


def _normalize_ocr_digits(text: str) -> str:
    translation = getattr(coord_reader, "_OCR_DIGIT_TRANSLATION", {})
    normalized = (text or "").translate(translation)
    # Windows OCR przy tym foncie często rozbija zero na "CI" albo "C1".
    normalized = normalized.replace("CI", "0").replace("C1", "0")
    normalized = normalized.replace("Cl", "0").replace("C|", "0")
    return normalized


def _numeric_windows(text: str) -> Iterator[int]:
    compact = re.sub(r"\s+", "", text or "")
    for match in re.finditer(r"\d{3,5}", compact):
        run = match.group(0)
        for length in (4, 3):
            if len(run) < length:
                continue
            for index in range(0, len(run) - length + 1):
                yield int(run[index : index + length])


def _split_bounds(bounds) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(bounds) == 4:
        x_min, x_max, y_min, y_max = bounds
        return (int(x_min), int(x_max)), (int(y_min), int(y_max))
    low, high = bounds
    return (int(low), int(high)), (int(low), int(high))


def _best_candidate(candidates: Counter[int], bounds: tuple[int, int]) -> int:
    center = (bounds[0] + bounds[1]) / 2.0
    return max(candidates, key=lambda value: (candidates[value], -abs(value - center)))
