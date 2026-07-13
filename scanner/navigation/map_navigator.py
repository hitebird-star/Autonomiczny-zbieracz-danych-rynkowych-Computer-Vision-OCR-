
"""Etap 5b Mapy Rynku: autonomiczna nawigacja zsynchronizowana z mapa.

Zastepuje sztywna trase wezyka dynamicznym planerem krokow.
Loop wola next_step() przed kazdym ruchem - navigator sam decyduje:
- czy isc w prawo/lewo (bounce na krawedzi strefy)
- czy zmienic pas
- czy przejsc do nastepnej strefy
- czy skonczyc (wszystkie strefy DONE)

Gdy pozycja nieznana -> fallback na wezyk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .route_planner import MovementStep

if TYPE_CHECKING:
    from scanner.analysis.shop_registry import ShopRegistry
    from scanner.analysis.zone_map import ZoneMap, Zone

# Szacunkowa kalibracja: 1 krok WASD (0.6s) = ile jednostek swiata.
# Do utwardzenia pomiarem gdy Stage 4 zacznie produkowac game_position.
UNITS_PER_STEP = 7.0
FALLBACK_LANES = 3
FALLBACK_STEPS_PER_LANE = 4

class MapSynchronizedNavigator:
    """Planer krokow zsynchronizowany z ZoneMap + ShopRegistry.

    Uzycie w run loop:
        nav = MapSynchronizedNavigator(zone_map, registry)
        loop.run(navigator=nav, max_shops=100)

    Hook po capturze (w command_auto):
        nav.stamp_position(scan.game_position)
        nav.record_shop(zone_id, is_new_fingerprint)
        nav.ingest_manifest(scan)
    """

    def __init__(
        self,
        zone_map: ZoneMap,
        shop_registry: ShopRegistry,
        *,
        left_key: str = "a", right_key: str = "d", lane_key: str = "s",
        step_duration: float = 0.6, settle: float = 0.9,
        units_per_step: float = UNITS_PER_STEP,
        fallback_lanes: int = FALLBACK_LANES,
        fallback_steps_per_lane: int = FALLBACK_STEPS_PER_LANE,
    ) -> None:
        self.zone_map = zone_map
        self.registry = shop_registry
        self.left_key = left_key
        self.right_key = right_key
        self.lane_key = lane_key
        self.step_duration = step_duration
        self.settle = settle
        self.units_per_step = units_per_step
        self.fallback_lanes = fallback_lanes
        self.fallback_steps_per_lane = fallback_steps_per_lane

        self._x: float | None = None
        self._y: float | None = None
        self._direction: str = "right"
        self._lane: int = 0
        self._step_in_lane: int = 0
        self._total_steps: int = 0
        self._current_zone_id: str | None = None
        self._transition_target: tuple[float, float] | None = None
        self._trans_idx: int = 0
        self._transition_total_x: int = 0
        self._finished: bool = False

        self._fb_lane: int = 0
        self._fb_step: int = 0
        self._fb_dir: str = "right"

    # --- Hooki dla pipeline ------------------------------------------------

    def stamp_position(self, pos: tuple[int, int] | None) -> None:
        if pos is None:
            return
        self._x, self._y = float(pos[0]), float(pos[1])
        self.zone_map.record_position(self._x, self._y)
        zone = self.zone_map.zone_for(self._x, self._y)
        self._current_zone_id = zone.zone_id if zone else None

    def record_shop(self, zone_id: str, is_new_fingerprint: bool) -> None:
        self.zone_map.record_open(zone_id, is_new_fingerprint=is_new_fingerprint)

    def ingest_manifest(self, manifest: dict) -> None:
        self.registry.ingest(manifest)

    # --- API dla run loop --------------------------------------------------

    @property
    def current_zone_id(self) -> str | None:
        return self._current_zone_id

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def is_finished(self) -> bool:
        return self._finished

    def next_step(self) -> MovementStep | None:
        if self._finished:
            return None
        # Bez pozycji -> wezyk
        if self._x is None or self._y is None:
            return self._fallback_step()
        zone = self.zone_map.zone_for(self._x, self._y)
        if zone is None:
            return self._fallback_step()
        self._current_zone_id = zone.zone_id
        # Przejscie miedzy strefami
        if self._transition_target is not None:
            return self._transition_step()
        # Strefa DONE -> inicjuj przejscie
        if self.zone_map.is_done(zone.zone_id):
            step = self._start_transition()
            if step is not None:
                return step
            if self._finished:
                return None
            return self._fallback_step()
        return self._zone_step(zone)

    def _zone_step(self, zone: Zone) -> MovementStep:
        x0, x1 = zone.box[0], zone.box[2]
        margin = self.units_per_step * 2
        if self._direction == "right" and self._x is not None and self._x + margin >= x1:
            self._direction = "left"
            self._lane += 1
            self._step_in_lane = 0
            key, kind = self.lane_key, "lane_change"
        elif self._direction == "left" and self._x is not None and self._x - margin <= x0:
            self._direction = "right"
            self._lane += 1
            self._step_in_lane = 0
            key, kind = self.lane_key, "lane_change"
        else:
            key = self.right_key if self._direction == "right" else self.left_key
            kind = "horizontal"
            self._step_in_lane += 1
        step = MovementStep(key, self.step_duration, self.settle, self._lane, self._step_in_lane, kind)
        self._dr(step)
        return step

    def _start_transition(self) -> MovementStep | None:
        next_z = self.zone_map.next_zone(self._x, self._y)  # type: ignore[arg-type]
        if next_z is None:
            self._finished = True
            return None
        self._transition_target = next_z.centroid
        self._trans_idx = 0
        dx = self._transition_target[0] - (self._x or 0)
        self._transition_total_x = max(1, round(abs(dx) / self.units_per_step))
        return self._transition_step()

    def _transition_step(self) -> MovementStep | None:
        if self._transition_target is None:
            return None
        tx, ty = self._transition_target
        if self._trans_idx < self._transition_total_x:
            dx = tx - (self._x or 0)
            key = self.right_key if dx > 0 else self.left_key
            kind = "navigate_x"
            self._trans_idx += 1
        else:
            dy = ty - (self._y or 0)
            if abs(dy) > self.units_per_step * 0.5:
                key, kind = self.lane_key, "navigate_y"
            else:
                self._transition_target = None
                self._lane += 1
                self._step_in_lane = 0
                self._direction = "right"
                if self._x is not None and self._y is not None:
                    z = self.zone_map.zone_for(self._x, self._y)
                    self._current_zone_id = z.zone_id if z else None
                return self.next_step()
        step = MovementStep(key, self.step_duration, self.settle, self._lane, self._trans_idx, kind)
        self._dr(step)
        return step

    def _fallback_step(self) -> MovementStep:
        if self._fb_step >= self.fallback_steps_per_lane:
            self._fb_lane += 1; self._fb_step = 0
            if self._fb_lane >= self.fallback_lanes:
                self._fb_lane = 0
                self._fb_dir = "left" if self._fb_dir == "right" else "right"
            key, kind = self.lane_key, "lane_change"
        else:
            key = self.right_key if self._fb_dir == "right" else self.left_key
            kind = "horizontal"
            self._fb_step += 1
        step = MovementStep(key, self.step_duration, self.settle, self._fb_lane, self._fb_step, kind)
        self._total_steps += 1
        return step

    def _dr(self, step: MovementStep) -> None:
        self._total_steps += 1
        if self._x is None:
            return
        if step.key == self.right_key: self._x += self.units_per_step
        elif step.key == self.left_key: self._x -= self.units_per_step
        elif step.key == self.lane_key and self._y is not None: self._y += self.units_per_step
