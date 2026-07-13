"""Bezpieczne sterowanie ruchem postaci klawiszami."""

from __future__ import annotations

from scanner.runtime import Clock, InputBackend, SystemClock


class MovementController:
    def __init__(
        self, input_backend: InputBackend, *, clock: Clock | None = None
    ) -> None:
        self.input = input_backend
        self.clock = clock or SystemClock()

    def hold(self, key: str, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("czas ruchu nie może być ujemny")
        self.input.key_down(key)
        try:
            self.clock.sleep(seconds)
        finally:
            self.input.key_up(key)

    def execute(self, key: str, seconds: float, settle: float = 0.0) -> None:
        self.hold(key, seconds)
        if settle > 0:
            self.clock.sleep(settle)

    def stop(self, keys: tuple[str, ...] = ("w", "a", "s", "d")) -> None:
        for key in keys:
            self.input.key_up(key)
