"""Kliknięcie sklepu i niezależne od OCR potwierdzenie otwarcia okna."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from scanner.config import GridGeometry
from scanner.runtime import Clock, InputBackend, ScreenBackend, SystemClock


@dataclass(frozen=True, slots=True)
class InteractionResult:
    opened: bool
    attempts: int
    elapsed: float
    reason: str | None = None


class ShopWindowProbe:
    """Rozpoznaje siatkę sklepu po periodycznych krawędziach, bez OCR."""

    def __init__(
        self,
        screen: ScreenBackend,
        geometry: GridGeometry,
        *,
        minimum_grid_score: float = 15.0,
    ) -> None:
        self.screen = screen
        self.geometry = geometry
        self.minimum_grid_score = minimum_grid_score

    def score(self) -> float:
        gray = np.asarray(self.screen.grab(self.geometry.box).convert("L"), dtype=float)
        if gray.shape != (
            self.geometry.rows * self.geometry.cell,
            self.geometry.columns * self.geometry.cell,
        ):
            return 0.0

        # Najmocniejsza krawędź ramki nie zawsze wypada dokładnie na granicy
        # komórki. W pełnym sklepie ikony mają też większy gradient niż ramka.
        # Szukamy więc dominującej fazy gradientu powtarzanej co ``cell`` px.
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        cell = self.geometry.cell
        if gray.shape[1] <= cell or gray.shape[0] <= cell:
            return 0.0
        x_phase = np.asarray(
            [np.mean(gx[:, phase::cell]) for phase in range(cell)]
        )
        y_phase = np.asarray(
            [np.mean(gy[phase::cell, :]) for phase in range(cell)]
        )
        x_prominence = float(np.max(x_phase) - np.median(x_phase))
        y_prominence = float(np.max(y_phase) - np.median(y_phase))
        return max(0.0, (x_prominence + y_prominence) / 2.0)

    def is_open(self) -> bool:
        return self.score() >= self.minimum_grid_score


class ShopInteractor:
    def __init__(
        self,
        input_backend: InputBackend,
        probe: ShopWindowProbe,
        *,
        clock: Clock | None = None,
        poll_interval: float = 0.1,
        move_duration: float = 0.12,
        retry_timeout: float = 1.0,
        early_bailout_s: float = 2.0,
        early_bailout_threshold: float = 20.0,
        observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.input = input_backend
        self.probe = probe
        self.clock = clock or SystemClock()
        self.poll_interval = poll_interval
        self.move_duration = move_duration
        self.retry_timeout = retry_timeout
        self.early_bailout_s = early_bailout_s
        self.early_bailout_threshold = early_bailout_threshold
        self.observer = observer

    def _emit(self, name: str, **data: Any) -> None:
        if self.observer is not None:
            self.observer({"name": name, **data})

    def _cancel_navigation(self) -> None:
        """Przerwij click-to-move po kliknięciu fałszywego lub niedostępnego celu."""

        # Najpierw gwarantujemy brak zakleszczonego klawisza, potem bardzo
        # krótki ruch do tyłu nadpisuje trasę wyznaczoną kliknięciem myszy.
        for key in ("w", "a", "s", "d"):
            self.input.key_up(key)
        self.input.key_down("s")
        self.clock.sleep(0.06)
        self.input.key_up("s")
        self._emit("navigation_cancelled")

    def open(
        self,
        position: tuple[int, int],
        *,
        timeout: float = 4.0,
        attempts: int = 2,
    ) -> InteractionResult:
        started = self.clock.monotonic()
        initial_score = self.probe.score()
        if initial_score >= self.probe.minimum_grid_score:
            self._emit("already_open", score=round(initial_score, 2))
            return InteractionResult(True, 0, 0.0)

        for attempt in range(1, attempts + 1):
            # Glevia nie zawsze rejestruje absolutne ``click(x, y)``. Najpierw
            # wykonujemy sprawdzony, względny ruch kursora używany także przez
            # capture dymków, a dopiero potem klikamy w bieżącej pozycji.
            cursor_before = self.input.position()
            self.input.move_to(*position, duration=self.move_duration)
            self.input.nudge(2)
            cursor_after = self.input.position()
            self.input.click()
            self._emit(
                "click",
                attempt=attempt,
                target=list(position),
                cursor_before=list(cursor_before),
                cursor_after=list(cursor_after),
            )
            # Pierwsze kliknięcie może uruchomić podejście postaci do sklepu,
            # dlatego zachowuje pełny timeout. Jeżeli potrzebne jest drugie
            # kliknięcie, postać jest już obok celu i prawdziwe okno pojawia
            # się szybko. Krótki retry ogranicza podatek płotów/postaci bez
            # odbierania czasu potrzebnego na dojście.
            attempt_timeout = (
                timeout
                if attempt == 1
                else min(timeout, self.retry_timeout)
            )
            deadline = self.clock.monotonic() + attempt_timeout
            best_score = 0.0
            last_score = 0.0
            bailed_out = False
            try:
                while self.clock.monotonic() < deadline:
                    last_score = self.probe.score()
                    best_score = max(best_score, last_score)
                    if last_score >= self.probe.minimum_grid_score:
                        self._emit(
                            "opened",
                            attempt=attempt,
                            score=round(last_score, 2),
                            elapsed=round(self.clock.monotonic() - started, 3),
                        )
                        return InteractionResult(
                            True, attempt, self.clock.monotonic() - started
                        )
                    # Tani early-out: false-targety daja score ~5-6,
                    # realne sklepy ~39+. Prog 20 w przepasci.
                    # Dziala na kazdej probie (retry tez tnie).
                    if (
                        self.clock.monotonic() - started >= self.early_bailout_s
                        and best_score < self.early_bailout_threshold
                    ):
                        self._emit(
                            "early_bailout",
                            attempt=attempt,
                            best_score=round(best_score, 2),
                            elapsed=round(self.clock.monotonic() - started, 3),
                        )
                        bailed_out = True
                        break
                    self.clock.sleep(self.poll_interval)
            except BaseException:
                self._cancel_navigation()
                self._emit("navigation_cancelled_on_abort")
                raise
            if bailed_out:
                self._cancel_navigation()
                break
            self._emit(
                "attempt_timeout",
                attempt=attempt,
                timeout=round(attempt_timeout, 3),
                best_score=round(best_score, 2),
                last_score=round(last_score, 2),
            )
            self._cancel_navigation()
        self._emit(
            "open_failed",
            attempts=attempts,
            elapsed=round(self.clock.monotonic() - started, 3),
        )
        return InteractionResult(
            False,
            attempts,
            self.clock.monotonic() - started,
            "shop_window_not_detected",
        )

    def wait_closed(self, timeout: float = 1.5) -> bool:
        """Poczekaj, aż siatka sklepu faktycznie zniknie po wysłaniu Esc."""

        deadline = self.clock.monotonic() + timeout
        while self.clock.monotonic() < deadline:
            if not self.probe.is_open():
                return True
            self.clock.sleep(self.poll_interval)
        return not self.probe.is_open()
