"""Kontrolowany ruch kalibracyjny dla Atlasa.

Ten moduł dotyka żywej gry, więc jest w pasie Codexa. Produkuje wyłącznie listę
`MoveObservation`; dopasowanie macierzy projekcji pozostaje w offline-core.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from scanner.atlas.contracts import FrameSnapshot, MoveObservation, Point2
from scanner.atlas.registration import screen_shift_between_frames
from scanner.runtime import InputBackend


@dataclass(frozen=True, slots=True)
class CalibrationMovePlan:
    keys: tuple[str, ...] = ("w", "d", "s", "a")
    hold_s: float = 0.45
    settle_s: float = 0.35
    after_timeout_s: float = 2.0
    after_poll_s: float = 0.2
    after_stable_px: float = 6.0
    repeats: int = 2
    min_shop_tracks: int = 2
    max_match_distance_px: float = 120.0
    expected_delta_radius_px: float = 25.0
    registration_max_shift_px: int = 80
    registration_min_confidence: float = 0.08
    use_frame_registration: bool = True


class AtlasLiveCalibrator:
    """Steruje krótkimi krokami i zbiera obserwacje ruchu dla fotogrametrii."""

    def __init__(
        self,
        *,
        input_backend: InputBackend,
        snapshot_provider,
        image_provider=None,
        plan: CalibrationMovePlan | None = None,
        clock=None,
    ) -> None:
        self.input = input_backend
        self.snapshot_provider = snapshot_provider
        self.image_provider = image_provider
        self.plan = plan or CalibrationMovePlan()
        self.clock = clock or time
        self.last_failure_reason: str | None = None
        self._screen_delta_hint_by_key: dict[str, Point2] = {}
        self._screen_delta_samples_by_key: dict[str, list[Point2]] = {}

    def run(self) -> list[MoveObservation]:
        observations: list[MoveObservation] = []
        for _ in range(max(1, self.plan.repeats)):
            for key in self.plan.keys:
                observation = self.move_once(key)
                if observation is not None:
                    observations.append(observation)
        return observations

    def move_once(
        self,
        key: str,
        *,
        before: FrameSnapshot | None = None,
    ) -> MoveObservation | None:
        self.last_failure_reason = None
        before = before or self.snapshot_provider()
        before_image = self._latest_image_copy()
        started = _utc_now_iso()
        start_time = self.clock.monotonic()
        self.input.key_down(key)
        try:
            self.clock.sleep(max(0.0, self.plan.hold_s))
        finally:
            self.input.key_up(key)
        self.clock.sleep(max(0.0, self.plan.settle_s))
        duration = self.clock.monotonic() - start_time

        if before.player_game is None:
            self.last_failure_reason = "before_coord_missing"
            return None

        after, delta_screen, confidence_override = self._wait_for_stable_after(
            before,
            key=key,
            before_image=before_image,
        )
        if after is None:
            return None

        return self.observation_from_snapshots(
            key,
            before,
            after,
            started_at=started,
            duration_s=duration,
            delta_screen_override=delta_screen,
            confidence_override=confidence_override,
        )

    def observation_from_snapshots(
        self,
        key: str,
        before: FrameSnapshot,
        after: FrameSnapshot,
        *,
        started_at: str | None = None,
        duration_s: float = 0.0,
        delta_screen_override: list[Point2] | None = None,
        confidence_override: float | None = None,
    ) -> MoveObservation | None:
        self.last_failure_reason = None
        if before.player_game is None:
            self.last_failure_reason = "before_coord_missing"
            return None
        if after.player_game is None:
            self.last_failure_reason = "after_coord_missing"
            return None
        if delta_screen_override is None:
            delta_screen = match_shop_deltas(
                before.shops_screen,
                after.shops_screen,
                max_distance_px=self.plan.max_match_distance_px,
                expected_delta=self._screen_delta_hint_by_key.get(key),
                expected_radius_px=self.plan.expected_delta_radius_px,
            )
            if len(delta_screen) < self._required_tracks_for_after(key):
                self.last_failure_reason = (
                    f"after_tracks_missing:{len(delta_screen)}/{self._required_tracks_for_after(key)}"
                )
                return None
            confidence = min(1.0, len(delta_screen) / max(1, self.plan.min_shop_tracks * 2))
        else:
            delta_screen = list(delta_screen_override)
            if not delta_screen:
                self.last_failure_reason = "registration_missing"
                return None
            confidence = 1.0 if confidence_override is None else float(confidence_override)
        delta_game = _delta(before.player_game, after.player_game)
        if delta_screen_override is None:
            self._remember_screen_delta_hint(key, delta_screen)
        return MoveObservation(
            key=key,
            started_at=started_at or _utc_now_iso(),
            duration_s=duration_s,
            player_before=before.player_game,
            player_after=after.player_game,
            delta_game=delta_game,
            delta_screen=tuple(delta_screen),
            confidence=confidence,
        )

    def _wait_for_stable_after(
        self,
        before: FrameSnapshot,
        *,
        key: str,
        before_image=None,
    ) -> tuple[FrameSnapshot | None, list[Point2], float | None]:
        deadline = self.clock.monotonic() + max(0.0, self.plan.after_timeout_s)
        best_track_count = 0
        best_registration_confidence = 0.0
        saw_coord = False
        previous_good: FrameSnapshot | None = None
        while True:
            try:
                after: FrameSnapshot = self.snapshot_provider()
            except Exception:
                self.last_failure_reason = "after_snapshot_error"
                return None, [], None
            if after.player_game is not None:
                saw_coord = True
                after_image = self._latest_image_copy()
                if self.plan.use_frame_registration and before_image is not None and after_image is not None:
                    registered = self._screen_delta_from_registration(before_image, after_image)
                    if registered is not None:
                        delta_screen, confidence = registered
                        best_registration_confidence = max(best_registration_confidence, confidence)
                        return after, delta_screen, confidence
                else:
                    delta_screen = match_shop_deltas(
                        before.shops_screen,
                        after.shops_screen,
                        max_distance_px=self.plan.max_match_distance_px,
                        expected_delta=self._screen_delta_hint_by_key.get(key),
                        expected_radius_px=self.plan.expected_delta_radius_px,
                    )
                    best_track_count = max(best_track_count, len(delta_screen))
                    if len(delta_screen) >= self._required_tracks_for_after(key):
                        if previous_good is not None and _snapshots_are_stable(
                            previous_good,
                            after,
                            min_tracks=self._required_tracks_for_after(key),
                            max_motion_px=self.plan.after_stable_px,
                            max_match_distance_px=self.plan.max_match_distance_px,
                        ):
                            return after, delta_screen, None
                        previous_good = after
            if self.clock.monotonic() >= deadline:
                if not saw_coord:
                    self.last_failure_reason = "after_coord_missing"
                elif best_registration_confidence > 0.0:
                    self.last_failure_reason = (
                        f"registration_low_confidence:{best_registration_confidence:.3f}/"
                        f"{self.plan.registration_min_confidence:.3f}"
                    )
                elif previous_good is not None:
                    self.last_failure_reason = "after_scene_unstable"
                else:
                    self.last_failure_reason = (
                        f"after_tracks_missing:{best_track_count}/{self.plan.min_shop_tracks}"
                    )
                return None, [], None
            self.clock.sleep(max(0.05, self.plan.after_poll_s))

    def _latest_image_copy(self):
        if self.image_provider is None:
            return None
        image = self.image_provider()
        if image is None:
            return None
        if hasattr(image, "copy"):
            return image.copy()
        return np.array(image, copy=True)

    def _screen_delta_from_registration(self, before_image, after_image) -> tuple[list[Point2], float] | None:
        dx, dy, confidence = screen_shift_between_frames(
            before_image,
            after_image,
            max_shift_px=self.plan.registration_max_shift_px,
        )
        if confidence < self.plan.registration_min_confidence:
            return None
        return [(float(dx), float(dy))], float(confidence)

    def _required_tracks_for_after(self, key: str) -> int:
        if key in self._screen_delta_hint_by_key:
            return 1
        return self.plan.min_shop_tracks

    def _remember_screen_delta_hint(self, key: str, delta_screen: list[Point2]) -> None:
        """Aktualizuj hint tylko z ruchów o realnym konsensusie.

        Jednoelementowy match jest dopuszczalny jako awaryjny pomiar po istniejącym
        hinte, ale nie może przesuwać hintu. W regularnej siatce właśnie taki
        `last-wins` drift potrafi zatwierdzić alias jako nową prawdę.
        """

        if len(delta_screen) < self.plan.min_shop_tracks:
            return
        samples = self._screen_delta_samples_by_key.setdefault(key, [])
        samples.append(_median_delta(delta_screen))
        self._screen_delta_hint_by_key[key] = _median_delta(samples)


def _snapshots_are_stable(
    previous: FrameSnapshot,
    current: FrameSnapshot,
    *,
    min_tracks: int,
    max_motion_px: float,
    max_match_distance_px: float,
) -> bool:
    deltas = match_shop_deltas(
        previous.shops_screen,
        current.shops_screen,
        max_distance_px=max_match_distance_px,
    )
    if len(deltas) < min_tracks:
        return False
    motion = sorted((dx * dx + dy * dy) ** 0.5 for dx, dy in deltas)
    median = motion[len(motion) // 2]
    return median <= max_motion_px


def _registration_view(image) -> np.ndarray:
    """Crop świata gry do korelacji fazowej, bez stałego UI.

    Nie korelujemy pełnego klienta, bo minimapa/HUD/dolny pasek są nieruchome
    względem ekranu i dla małych ruchów tworzą konkurencyjny pik zero-shift.
    Zostawiamy centralny viewport świata, a postać/efekty w środku neutralizujemy
    medianą cropa.
    """

    if hasattr(image, "convert"):
        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    else:
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError(f"oczekiwano obrazu 2D/3D, dostalem shape={arr.shape}")
    h, w = arr.shape[:2]
    y0 = int(h * 0.20)
    y1 = int(h * 0.80)
    x0 = int(w * 0.08)
    x1 = int(w * 0.86)
    crop = np.array(arr[y0:y1, x0:x1], copy=True)
    if crop.size == 0:
        raise ValueError("crop rejestracji jest pusty")

    ch, cw = crop.shape[:2]
    cx = int(w * 0.50) - x0
    cy = int(h * 0.53) - y0
    mask_w = max(24, int(w * 0.13))
    mask_h = max(32, int(h * 0.18))
    mx0 = max(0, cx - mask_w // 2)
    mx1 = min(cw, cx + mask_w // 2)
    my0 = max(0, cy - mask_h // 2)
    my1 = min(ch, cy + mask_h // 2)
    if mx0 < mx1 and my0 < my1:
        fill = np.median(crop.reshape(-1, crop.shape[-1]), axis=0)
        crop[my0:my1, mx0:mx1] = fill
    return crop


def match_shop_deltas(
    before: tuple[Point2, ...],
    after: tuple[Point2, ...],
    *,
    max_distance_px: float = 120.0,
    consensus_radius_px: float = 18.0,
    expected_delta: Point2 | None = None,
    expected_radius_px: float = 25.0,
) -> list[Point2]:
    """Dopasuj statyczne sklepy przez dominujący wektor przesunięcia.

    W bardzo gęstym rynku nearest-neighbor często paruje nie ten sam sklep.
    Statyczne obiekty powinny jednak po ruchu kamery przesunąć się prawie jednym
    wspólnym wektorem, więc wybieramy największy spójny klaster delt.
    """

    max_dist2 = max_distance_px * max_distance_px
    radius2 = consensus_radius_px * consensus_radius_px
    candidates: list[tuple[int, int, float, float]] = []
    for before_index, (bx, by) in enumerate(before):
        for after_index, (ax, ay) in enumerate(after):
            dx = float(ax - bx)
            dy = float(ay - by)
            if dx * dx + dy * dy <= max_dist2:
                candidates.append((before_index, after_index, dx, dy))
    if not candidates:
        return []

    best: list[tuple[int, int, float, float]] = []
    best_spread = float("inf")
    best_expected_distance = float("inf")
    for _, _, anchor_dx, anchor_dy in candidates:
        near = [
            candidate
            for candidate in candidates
            if (candidate[2] - anchor_dx) ** 2 + (candidate[3] - anchor_dy) ** 2
            <= radius2
        ]
        near.sort(
            key=lambda item: (item[2] - anchor_dx) ** 2 + (item[3] - anchor_dy) ** 2
        )
        used_before: set[int] = set()
        used_after: set[int] = set()
        selected: list[tuple[int, int, float, float]] = []
        for item in near:
            before_index, after_index, _, _ = item
            if before_index in used_before or after_index in used_after:
                continue
            used_before.add(before_index)
            used_after.add(after_index)
            selected.append(item)
        if not selected:
            continue
        mean_dx = sum(item[2] for item in selected) / len(selected)
        mean_dy = sum(item[3] for item in selected) / len(selected)
        spread = sum(
            ((item[2] - mean_dx) ** 2 + (item[3] - mean_dy) ** 2) ** 0.5
            for item in selected
        ) / len(selected)
        expected_distance = 0.0
        expected_ok = True
        if expected_delta is not None:
            expected_distance = (
                (mean_dx - expected_delta[0]) ** 2
                + (mean_dy - expected_delta[1]) ** 2
            ) ** 0.5
            expected_ok = expected_distance <= expected_radius_px
        if not expected_ok:
            continue
        if _cluster_is_better(
            count=len(selected),
            spread=spread,
            expected_distance=expected_distance,
            best_count=len(best),
            best_spread=best_spread,
            best_expected_distance=best_expected_distance,
            has_expected=expected_delta is not None,
        ):
            best = selected
            best_spread = spread
            best_expected_distance = expected_distance

    return [(dx, dy) for _, _, dx, dy in best]


def _cluster_is_better(
    *,
    count: int,
    spread: float,
    expected_distance: float,
    best_count: int,
    best_spread: float,
    best_expected_distance: float,
    has_expected: bool,
) -> bool:
    if best_count <= 0:
        return True
    if has_expected:
        if expected_distance < best_expected_distance - 1e-9:
            return True
        if abs(expected_distance - best_expected_distance) <= 1e-9:
            return count > best_count or (count == best_count and spread < best_spread)
        return False
    return count > best_count or (count == best_count and spread < best_spread)


def _median_delta(deltas: list[Point2]) -> Point2:
    xs = sorted(delta[0] for delta in deltas)
    ys = sorted(delta[1] for delta in deltas)
    middle = len(deltas) // 2
    if len(deltas) % 2:
        return float(xs[middle]), float(ys[middle])
    return (
        float((xs[middle - 1] + xs[middle]) / 2.0),
        float((ys[middle - 1] + ys[middle]) / 2.0),
    )


def _delta(before: Point2 | None, after: Point2 | None) -> Point2:
    if before is None or after is None:
        return (0.0, 0.0)
    return (float(after[0] - before[0]), float(after[1] - before[1]))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
