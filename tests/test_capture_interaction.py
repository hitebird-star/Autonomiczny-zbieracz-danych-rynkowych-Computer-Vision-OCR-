from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from scanner.capture import ShopCapturer, TooltipCapturer
from scanner.capture.icon_matcher import group_slots_by_icon
from scanner.config import CaptureSettings, GridGeometry
from scanner.detection.interaction import ShopInteractor, ShopWindowProbe
from tests.fakes import FakeClock, FakeInput, FakeScreen


def grid_image(
    geometry: GridGeometry, occupied: set[tuple[int, int]]
) -> Image.Image:
    cell = geometry.cell
    array = np.full(
        (geometry.rows * cell, geometry.columns * cell, 3), 18, dtype=np.uint8
    )
    for row in range(geometry.rows):
        for column in range(geometry.columns):
            y0, x0 = row * cell, column * cell
            array[y0 : y0 + cell, x0] = 48
            array[y0 + 3 : y0 + cell - 3, x0 + 3 : x0 + cell - 3] = 18
            if (column, row) in occupied:
                array[y0 + 7 : y0 + cell - 7, x0 + 7 : x0 + cell - 7] = (
                    180,
                    70,
                    30,
                )
    return Image.fromarray(array)


def tooltip_sequence(width: int, height: int, count: int = 3) -> list[Image.Image]:
    baseline = Image.new("RGB", (width, height), (70, 90, 55))
    post = baseline.copy()
    array = np.asarray(post).copy()
    array[8 : height - 8, 10 : width - 10] = (15, 15, 15)
    array[18 : height - 18 : 8, 20 : width - 20] = (220, 220, 220)
    post = Image.fromarray(array)
    return [baseline, baseline] + [post] * count


def marker_recognizer(image: Image.Image) -> list[dict]:
    return [{"text": "[Cena sprzedaży]", "box": (10, 10, 100, 25)}]


class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = GridGeometry(
            origin=(100, 200),
            offset=(4, 39),
            cell=32,
            columns=3,
            rows=2,
            occupancy_residual=3.0,
        )

    def test_detects_only_occupied_cells(self) -> None:
        image = grid_image(self.geometry, {(1, 0), (2, 1)})
        capturer = ShopCapturer(FakeScreen(image), FakeInput(), self.geometry)

        slots = capturer.occupied_slots(image)

        self.assertEqual(
            [(slot.column, slot.row) for slot in slots], [(1, 0), (2, 1)]
        )

    def test_tooltip_capture_collects_multiple_raw_frames(self) -> None:
        image = grid_image(self.geometry, {(1, 0)})
        settings = CaptureSettings(
            hover_delay=0.2,
            move_duration=0.0,
            frames_per_slot=3,
            frame_interval=0.05,
            tooltip_width=80,
            tooltip_height=60,
        )
        screen = FakeScreen(tooltip_sequence(80, 60, 4))
        input_backend = FakeInput()
        clock = FakeClock()
        slot = ShopCapturer(screen, input_backend, self.geometry).occupied_slots(image)[0]

        result = TooltipCapturer(
            screen,
            input_backend,
            self.geometry,
            settings,
            clock=clock,
            marker_recognizer=marker_recognizer,
        ).capture(slot)

        self.assertEqual(len(result.frames), 3)
        self.assertGreaterEqual(clock.now, 0.3)
        # Najpierw punkt spoczynkowy (usuwa stary dymek), potem właściwy slot.
        self.assertEqual(input_backend.actions[0][:3], ("move", 152, 216))
        self.assertIn(("move", 152, 255, 0.1), input_backend.actions)
        self.assertIn(("nudge", 4), input_backend.actions)

    def test_each_slot_gets_a_fresh_baseline(self) -> None:
        settings = CaptureSettings(
            hover_delay=0.0,
            move_duration=0.0,
            frames_per_slot=1,
            tooltip_width=80,
            tooltip_height=60,
        )
        baseline = Image.new("RGB", (80, 60), (70, 90, 55))
        tooltip = tooltip_sequence(80, 60, 1)[-1]
        screen = FakeScreen(
            [baseline, baseline, tooltip, baseline, baseline, tooltip]
        )
        input_backend = FakeInput()
        capturer = TooltipCapturer(
            screen,
            input_backend,
            self.geometry,
            settings,
            clock=FakeClock(),
            bounds=(100, 200, 120, 100),
            marker_recognizer=marker_recognizer,
        )
        slots = ShopCapturer(
            FakeScreen(grid_image(self.geometry, {(1, 0), (2, 0)})),
            FakeInput(),
            self.geometry,
        ).occupied_slots(grid_image(self.geometry, {(1, 0), (2, 0)}))

        first = capturer.capture(slots[0])
        second = capturer.capture_stack_member(slots[1], first.frames[0])
        results = [first, second]

        self.assertEqual([len(result.frames) for result in results], [1, 1])
        self.assertFalse(first.matched_reference)
        self.assertTrue(second.matched_reference)
        self.assertEqual(len(screen.boxes), 6)
        rest_moves = [
            action
            for action in input_backend.actions
            if action[:3] == ("move", 152, 216)
        ]
        self.assertEqual(len(rest_moves), 2)

    def test_icon_groups_do_not_merge_visually_different_items(self) -> None:
        image = grid_image(self.geometry, {(0, 0), (1, 0), (2, 0)})
        pixels = np.asarray(image).copy()
        # Sloty 0 i 1 dostają tę samą ikonę, slot 2 wyraźnie inną.
        pixels[6:26, 6:26] = (150, 80, 20)
        pixels[6:26, 38:58] = (151, 80, 20)
        pixels[6:26, 70:90] = (20, 160, 180)
        grid = Image.fromarray(pixels)
        slots = [
            type("Slot", (), {"slot": index, "column": index, "row": 0})()
            for index in range(3)
        ]

        groups = group_slots_by_icon(grid, slots, self.geometry)

        self.assertEqual([[slot.slot for slot in group] for group in groups], [[0, 1], [2]])

    def test_reference_match_rejects_changed_price_pixels(self) -> None:
        first = Image.new("RGB", (180, 160), (15, 15, 15))
        changed = first.copy()
        pixels = np.asarray(changed).copy()
        pixels[120:130, 80:88] = (220, 220, 220)
        changed = Image.fromarray(pixels)

        self.assertFalse(TooltipCapturer._matches_reference(first, changed))

    def test_tooltip_frame_is_clamped_to_game_window(self) -> None:
        settings = CaptureSettings(
            hover_delay=0.0,
            move_duration=0.0,
            frames_per_slot=1,
            tooltip_width=80,
            tooltip_height=60,
        )
        screen = FakeScreen(tooltip_sequence(80, 60))
        capturer = TooltipCapturer(
            screen,
            FakeInput(),
            self.geometry,
            settings,
            clock=FakeClock(),
            bounds=(100, 200, 120, 100),
            marker_recognizer=marker_recognizer,
        )
        slot = ShopCapturer(
            FakeScreen(grid_image(self.geometry, {(2, 1)})),
            FakeInput(),
            self.geometry,
        ).occupied_slots(grid_image(self.geometry, {(2, 1)}))[0]

        result = capturer.capture(slot)

        self.assertEqual(screen.boxes[-1], (140, 240, 80, 60))
        self.assertEqual(len(result.frames), 1)

    def test_tooltip_pauses_when_game_loses_focus(self) -> None:
        class Focus:
            checks = 0
            waits = 0

            def is_foreground(self) -> bool:
                self.checks += 1
                return self.checks > 1

            def wait_until_foreground(self, timeout: float = 15.0) -> bool:
                self.waits += 1
                return True

        image = grid_image(self.geometry, {(1, 0)})
        slot = ShopCapturer(
            FakeScreen(image), FakeInput(), self.geometry
        ).occupied_slots(image)[0]
        focus = Focus()
        clock = FakeClock()
        capturer = TooltipCapturer(
            FakeScreen(tooltip_sequence(80, 60)),
            FakeInput(),
            self.geometry,
            CaptureSettings(
                hover_delay=0.0,
                move_duration=0.0,
                frames_per_slot=1,
                tooltip_width=80,
                tooltip_height=60,
            ),
            clock=clock,
            focus=focus,
            marker_recognizer=marker_recognizer,
        )

        capturer.capture(slot)

        self.assertEqual(focus.waits, 1)
        self.assertGreaterEqual(clock.now, 0.15)

    def test_missing_tooltip_returns_no_frames(self) -> None:
        image = Image.new("RGB", (80, 60), (70, 90, 55))
        input_backend = FakeInput()
        capturer = TooltipCapturer(
            FakeScreen(image),
            input_backend,
            self.geometry,
            CaptureSettings(
                hover_delay=0.0,
                move_duration=0.0,
                frames_per_slot=2,
                tooltip_timeout=0.15,
                tooltip_poll_interval=0.05,
                hover_attempts=2,
                tooltip_width=80,
                tooltip_height=60,
            ),
            clock=FakeClock(),
            marker_recognizer=lambda image: [],
        )
        slot = ShopCapturer(
            FakeScreen(grid_image(self.geometry, {(1, 0)})),
            FakeInput(),
            self.geometry,
        ).occupied_slots(grid_image(self.geometry, {(1, 0)}))[0]

        result = capturer.capture(slot)

        self.assertEqual(result.frames, ())
        self.assertEqual(
            [action for action in input_backend.actions if action[0] == "nudge"],
            [("nudge", 4), ("nudge", 5)],
        )

    def test_fast_capture_caps_only_first_pass_attempts(self) -> None:
        image = Image.new("RGB", (80, 60), (70, 90, 55))
        slot = ShopCapturer(
            FakeScreen(grid_image(self.geometry, {(1, 0)})),
            FakeInput(),
            self.geometry,
        ).occupied_slots(grid_image(self.geometry, {(1, 0)}))[0]

        def make_capturer(input_backend):
            return TooltipCapturer(
                FakeScreen(image),
                input_backend,
                self.geometry,
                CaptureSettings(
                    hover_delay=0.0,
                    move_duration=0.0,
                    frames_per_slot=1,
                    tooltip_timeout=0.05,
                    tooltip_poll_interval=0.05,
                    hover_attempts=3,
                    first_pass_hover_attempts=2,
                    tooltip_width=80,
                    tooltip_height=60,
                ),
                clock=FakeClock(),
                marker_recognizer=lambda image: [],
            )

        fast_input = FakeInput()
        full_input = FakeInput()
        self.assertEqual(make_capturer(fast_input).capture_fast(slot).frames, ())
        self.assertEqual(make_capturer(full_input).capture(slot).frames, ())

        fast_nudges = [
            action for action in fast_input.actions if action[0] == "nudge"
        ]
        full_nudges = [
            action for action in full_input.actions if action[0] == "nudge"
        ]
        self.assertEqual(fast_nudges, [("nudge", 4), ("nudge", 5)])
        self.assertEqual(
            full_nudges,
            [("nudge", 4), ("nudge", 5), ("nudge", 6)],
        )

    def test_tooltip_retries_full_rest_to_slot_cycle(self) -> None:
        baseline = Image.new("RGB", (80, 60), (70, 90, 55))
        post = tooltip_sequence(80, 60, 1)[-1]
        screen = FakeScreen(
            [
                baseline,
                baseline,
                baseline,
                baseline,
                baseline,
                baseline,
                baseline,
                post,
            ]
        )
        input_backend = FakeInput()
        capturer = TooltipCapturer(
            screen,
            input_backend,
            self.geometry,
            CaptureSettings(
                hover_delay=0.0,
                move_duration=0.0,
                frames_per_slot=1,
                tooltip_timeout=0.1,
                tooltip_poll_interval=0.05,
                hover_attempts=2,
                tooltip_width=80,
                tooltip_height=60,
            ),
            clock=FakeClock(),
            marker_recognizer=marker_recognizer,
        )
        slot = ShopCapturer(
            FakeScreen(grid_image(self.geometry, {(1, 0)})),
            FakeInput(),
            self.geometry,
        ).occupied_slots(grid_image(self.geometry, {(1, 0)}))[0]

        result = capturer.capture(slot)

        self.assertEqual(len(result.frames), 1)
        self.assertEqual(
            [action for action in input_backend.actions if action[0] == "nudge"],
            [("nudge", 4), ("nudge", 5)],
        )
        rest_moves = [
            action
            for action in input_backend.actions
            if action[:3] == ("move", 152, 216)
        ]
        self.assertEqual(len(rest_moves), 2)

    def test_tooltip_rejects_slot_outside_live_game_window(self) -> None:
        slot = ShopCapturer(
            FakeScreen(grid_image(self.geometry, {(2, 1)})),
            FakeInput(),
            self.geometry,
        ).occupied_slots(grid_image(self.geometry, {(2, 1)}))[0]
        capturer = TooltipCapturer(
            FakeScreen(Image.new("RGB", (80, 60), "black")),
            FakeInput(),
            self.geometry,
            CaptureSettings(tooltip_width=80, tooltip_height=60),
            clock=FakeClock(),
            bounds=(0, 0, 100, 100),
            marker_recognizer=marker_recognizer,
        )

        with self.assertRaisesRegex(RuntimeError, "slot_center_outside_game"):
            capturer.capture(slot)

    def test_moving_character_is_not_tooltip(self) -> None:
        baseline = Image.new("RGB", (320, 240), (70, 90, 55))
        moving = np.asarray(baseline).copy()
        moving[30:220, 80:250] = (90, 65, 45)
        moving[60:200, 125:205] = (150, 95, 70)
        moving = Image.fromarray(moving)
        capturer = TooltipCapturer(
            FakeScreen(baseline),
            FakeInput(),
            self.geometry,
            CaptureSettings(),
            clock=FakeClock(),
        )

        self.assertIsNone(capturer._tooltip_bbox(baseline, moving))

    def test_sales_marker_accepts_ocr_typo(self) -> None:
        capturer = TooltipCapturer(
            FakeScreen(Image.new("RGB", (80, 60), "black")),
            FakeInput(),
            self.geometry,
            CaptureSettings(),
            marker_recognizer=lambda image: [
                {"text": "[Cena sprze.dažy]", "box": (1, 1, 20, 8)}
            ],
        )

        self.assertTrue(
            capturer._has_sales_marker(Image.new("RGB", (80, 60), "black"))
        )


class InteractionTests(unittest.TestCase):
    def test_probe_detects_periodic_grid(self) -> None:
        geometry = GridGeometry(cell=32, columns=3, rows=2)
        probe = ShopWindowProbe(
            FakeScreen(grid_image(geometry, {(1, 0)})),
            geometry,
            minimum_grid_score=1.0,
        )
        self.assertTrue(probe.is_open())

    def test_probe_detects_grid_even_when_every_slot_has_bright_icon(self) -> None:
        geometry = GridGeometry(cell=32, columns=10, rows=10)
        occupied = {
            (column, row)
            for row in range(geometry.rows)
            for column in range(geometry.columns)
        }
        image = np.asarray(grid_image(geometry, occupied)).copy()
        for row in range(geometry.rows):
            for column in range(geometry.columns):
                x0 = column * geometry.cell + 7
                y0 = row * geometry.cell + 7
                image[y0 : y0 + 18, x0 : x0 + 9] = (
                    210 if (row + column) % 2 else 40,
                    180,
                    30,
                )
        probe = ShopWindowProbe(
            FakeScreen(Image.fromarray(image)),
            geometry,
            minimum_grid_score=15.0,
        )

        self.assertGreaterEqual(probe.score(), 15.0)
        self.assertTrue(probe.is_open())

    def test_probe_rejects_non_periodic_scene(self) -> None:
        geometry = GridGeometry(cell=32, columns=10, rows=10)
        rng = np.random.default_rng(42)
        scene = Image.fromarray(
            rng.integers(0, 150, (320, 320, 3), dtype=np.uint8)
        )
        probe = ShopWindowProbe(FakeScreen(scene), geometry)

        self.assertLess(probe.score(), probe.minimum_grid_score)
        self.assertFalse(probe.is_open())

    def test_interactor_retries_until_probe_opens(self) -> None:
        class Probe:
            calls = 0
            minimum_grid_score = 15.0

            def score(self) -> float:
                self.calls += 1
                return 20.0 if self.calls >= 5 else 0.0

        input_backend = FakeInput()
        result = ShopInteractor(
            input_backend, Probe(), clock=FakeClock(), poll_interval=0.1
        ).open((300, 400), timeout=0.25, attempts=2)

        self.assertTrue(result.opened)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(
            [action for action in input_backend.actions if action[0] == "move"],
            [("move", 300, 400, 0.12), ("move", 300, 400, 0.12)],
        )
        self.assertEqual(
            [action for action in input_backend.actions if action[0] == "nudge"],
            [("nudge", 2), ("nudge", 2)],
        )
        self.assertEqual(
            [action for action in input_backend.actions if action[0] == "click"],
            [("click", None, None), ("click", None, None)],
        )
        self.assertIn(("down", "s"), input_backend.actions)
        self.assertGreaterEqual(
            input_backend.actions.count(("up", "s")),
            2,
        )

    def test_interactor_uses_short_timeout_only_for_retry(self) -> None:
        class Probe:
            minimum_grid_score = 15.0

            def score(self) -> float:
                return 0.0

        events = []
        result = ShopInteractor(
            FakeInput(),
            Probe(),
            clock=FakeClock(),
            poll_interval=0.1,
            retry_timeout=0.3,
            observer=events.append,
        ).open((300, 400), timeout=1.0, attempts=2)

        timeouts = [
            event["timeout"]
            for event in events
            if event["name"] == "attempt_timeout"
        ]
        self.assertFalse(result.opened)
        self.assertEqual(timeouts, [1.0, 0.3])
        self.assertLess(result.elapsed, 1.6)

    def test_interactor_reports_click_position_and_probe_score(self) -> None:
        class Probe:
            minimum_grid_score = 15.0
            scores = iter((0.0, 4.0, 18.0))

            def score(self) -> float:
                return next(self.scores)

        events = []
        result = ShopInteractor(
            FakeInput(),
            Probe(),
            clock=FakeClock(),
            poll_interval=0.1,
            observer=events.append,
        ).open((300, 400), timeout=0.3, attempts=1)

        self.assertTrue(result.opened)
        self.assertEqual(events[0]["name"], "click")
        self.assertEqual(events[0]["target"], [300, 400])
        self.assertEqual(events[-1]["name"], "opened")
        self.assertEqual(events[-1]["score"], 18.0)

    def test_interactor_cancels_click_to_move_after_timeout(self) -> None:
        class Probe:
            minimum_grid_score = 15.0

            def score(self) -> float:
                return 0.0

        input_backend = FakeInput()
        events = []
        result = ShopInteractor(
            input_backend,
            Probe(),
            clock=FakeClock(),
            poll_interval=0.1,
            observer=events.append,
        ).open((300, 400), timeout=0.2, attempts=1)

        self.assertFalse(result.opened)
        self.assertIn(("down", "s"), input_backend.actions)
        self.assertEqual(input_backend.actions[-1], ("up", "s"))
        self.assertTrue(
            any(event["name"] == "navigation_cancelled" for event in events)
        )

    def test_interactor_cancels_navigation_when_operator_aborts(self) -> None:
        class Probe:
            minimum_grid_score = 15.0

            def score(self) -> float:
                return 0.0

        class InterruptingClock(FakeClock):
            def sleep(self, seconds: float) -> None:
                if seconds == 0.1:
                    raise KeyboardInterrupt
                super().sleep(seconds)

        input_backend = FakeInput()
        events = []
        interactor = ShopInteractor(
            input_backend,
            Probe(),
            clock=InterruptingClock(),
            poll_interval=0.1,
            observer=events.append,
        )

        with self.assertRaises(KeyboardInterrupt):
            interactor.open((300, 400), timeout=4.0, attempts=1)

        self.assertEqual(input_backend.actions[-1], ("up", "s"))
        self.assertTrue(
            any(
                event["name"] == "navigation_cancelled_on_abort"
                for event in events
            )
        )

    def test_interactor_waits_until_shop_is_closed(self) -> None:
        class Probe:
            states = iter((True, True, False))

            def is_open(self) -> bool:
                return next(self.states)

        clock = FakeClock()
        interactor = ShopInteractor(
            FakeInput(), Probe(), clock=clock, poll_interval=0.1
        )

        self.assertTrue(interactor.wait_closed(timeout=0.5))
        self.assertEqual(clock.now, 0.2)


if __name__ == "__main__":
    unittest.main()
