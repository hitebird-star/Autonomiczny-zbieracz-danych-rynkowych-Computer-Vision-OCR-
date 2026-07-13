"""Deterministyczna trasa wężykiem pokrywająca rynek."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MovementStep:
    key: str
    duration: float
    settle: float
    lane: int
    step: int
    kind: str


class SerpentineRoutePlanner:
    def __init__(
        self,
        *,
        left_key: str = "a",
        right_key: str = "d",
        lane_key: str = "s",
        step_duration: float = 0.6,
        steps_per_lane: int = 4,
        lanes: int = 3,
        settle: float = 0.9,
    ) -> None:
        if steps_per_lane < 1 or lanes < 1:
            raise ValueError("trasa musi mieć co najmniej jeden krok i pas")
        self.left_key = left_key
        self.right_key = right_key
        self.lane_key = lane_key
        self.step_duration = step_duration
        self.steps_per_lane = steps_per_lane
        self.lanes = lanes
        self.settle = settle

    def steps(self) -> tuple[MovementStep, ...]:
        route = []
        for lane in range(self.lanes):
            horizontal_key = self.right_key if lane % 2 == 0 else self.left_key
            for step in range(self.steps_per_lane):
                route.append(
                    MovementStep(
                        horizontal_key,
                        self.step_duration,
                        self.settle,
                        lane,
                        step,
                        "horizontal",
                    )
                )
            if lane + 1 < self.lanes:
                route.append(
                    MovementStep(
                        self.lane_key,
                        self.step_duration,
                        self.settle,
                        lane,
                        self.steps_per_lane,
                        "lane_change",
                    )
                )
        return tuple(route)
