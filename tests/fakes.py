from __future__ import annotations

from collections import deque

from PIL import Image


class FakeScreen:
    def __init__(self, images: list[Image.Image] | Image.Image) -> None:
        if isinstance(images, Image.Image):
            images = [images]
        self.images = deque(images)
        self.boxes: list[tuple[int, int, int, int]] = []

    def grab(self, box: tuple[int, int, int, int]) -> Image.Image:
        self.boxes.append(box)
        if len(self.images) > 1:
            return self.images.popleft()
        return self.images[0].copy()


class FakeInput:
    def __init__(self) -> None:
        self.actions: list[tuple] = []

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        self.actions.append(("move", x, y, duration))

    def nudge(self, pixels: int = 3) -> None:
        self.actions.append(("nudge", pixels))

    def position(self) -> tuple[int, int]:
        for action in reversed(self.actions):
            if action[0] == "move":
                return int(action[1]), int(action[2])
        return 0, 0

    def click(self, x: int | None = None, y: int | None = None) -> None:
        self.actions.append(("click", x, y))

    def key_down(self, key: str) -> None:
        self.actions.append(("down", key))

    def key_up(self, key: str) -> None:
        self.actions.append(("up", key))

    def press(self, key: str) -> None:
        self.actions.append(("press", key))


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
