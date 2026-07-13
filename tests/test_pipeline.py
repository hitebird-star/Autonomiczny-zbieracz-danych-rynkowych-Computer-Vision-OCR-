from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scanner.analysis.coord_reader import CoordParse
from scanner.capture.shop_capture import OccupiedSlot
from scanner.capture.tooltip_capture import TooltipFrames
from scanner.config import GridGeometry
from scanner.detection import ShopCandidate, ShopTracker, TrackedShop
from scanner.models import ItemObservation, ScanStatus
from scanner.pipeline import (
    AnalysisWorker,
    AutonomousMarketLoop,
    CaptureOutcome,
    GameCapturePipeline,
)
from scanner.models import ShopScan
from scanner.navigation import MovementStep
from scanner.storage import CSVExporter, ScanRepository
from tests.fakes import FakeInput


class OpenInteractor:
    def open(self, position, *, timeout):
        class Result:
            opened = True
            attempts = 1
            elapsed = 0.1
            reason = None

        return Result()


class FailedOpenInteractor:
    class Probe:
        def score(self) -> float:
            return 7.0

    probe = Probe()

    def open(self, position, *, timeout):
        class Result:
            opened = False
            attempts = 1
            elapsed = 0.2
            reason = "shop_window_not_detected"

        return Result()


class FakeShopCapturer:
    geometry = GridGeometry()

    def capture_shop(self):
        return Image.new("RGB", (100, 100), (40, 30, 20))

    def capture_grid(self):
        return Image.new("RGB", (64, 64), "black")

    def occupied_slots(self, image):
        return [OccupiedSlot(slot=1, row=0, column=1, residual=20.0)]


class FakeScreen:
    def grab(self, box):
        return Image.new("RGB", (64, 64), "black")


class FakeShopCapturerWithScreen(FakeShopCapturer):
    screen = FakeScreen()


class FakeTooltipCapturer:
    def capture(self, slot):
        return TooltipFrames(
            slot,
            (
                Image.new("RGB", (80, 60), "black"),
                Image.new("RGB", (80, 60), "white"),
            ),
        )


class EmptyTooltipCapturer:
    def capture(self, slot):
        return TooltipFrames(slot, ())


class FlakyTooltipCapturer:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self, slot):
        self.calls += 1
        if self.calls == 1:
            return TooltipFrames(slot, ())
        return TooltipFrames(
            slot,
            (Image.new("RGB", (80, 60), "black"),),
        )


class VerifyingEngine:
    def analyze(self, scan, repository):
        observation = scan.slots[1]
        observation.ai = {
            "item": "Odłamek Metina",
            "unit_price": 7_000_000,
            "quantity": 10,
        }
        observation.validation = {"quantity": 10, "unit_price": 7_000_000}
        observation.status = ScanStatus.VERIFIED
        scan.transition(ScanStatus.PROVISIONAL)
        scan.transition(ScanStatus.VERIFIED)
        return scan


class PipelineTests(unittest.TestCase):
    def test_current_position_uses_startup_fallback_after_default_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            shop_capturer = FakeShopCapturer()
            shop_capturer.screen = FakeScreen()
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                shop_capturer,
                FakeTooltipCapturer(),
                ScanRepository(Path(temp) / "scans"),
                ShopTracker(),
                FakeInput(),
                window_box=(0, 0, 64, 64),
            )

            with patch(
                "scanner.analysis.coord_reader.read_image",
                side_effect=[
                    None,
                    CoordParse(x=485, y=720, channel=None, map_name=None),
                ],
            ) as read_image:
                self.assertEqual(pipeline.read_current_position(), (485, 720))

            self.assertEqual(read_image.call_count, 2)
            self.assertEqual(pipeline.position_source, "ocr")

    def test_second_pass_recovers_only_failed_tooltip_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            tracker = ShopTracker()
            capturer = FlakyTooltipCapturer()
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                FakeShopCapturer(),
                capturer,
                repository,
                tracker,
                FakeInput(),
            )

            outcome = pipeline.capture(TrackedShop("shop-1", (300, 400)))
            events = [
                json.loads(line)
                for line in (
                    repository.scan_dir(outcome.scan.scan_id)
                    / "raw_events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            recovery = [
                event
                for event in events
                if event["event"] == "recovery_pass"
            ]
            recovery_started = [
                event
                for event in events
                if event["event"] == "recovery_started"
            ]
            captured = [
                event
                for event in events
                if event["event"] == "slot_captured"
            ]

            self.assertEqual(outcome.scan.status, ScanStatus.CAPTURED)
            self.assertEqual(outcome.scan.captured_slots, 1)
            self.assertEqual(capturer.calls, 2)
            self.assertEqual(
                [(event["phase"], event.get("recovered")) for event in recovery],
                [("started", None), ("completed", 1)],
            )
            self.assertEqual(recovery_started[0]["queued"], 1)
            self.assertEqual(captured[0]["capture_pass"], 2)
            self.assertTrue(captured[0]["recovered"])
            self.assertTrue(captured[0]["recovery_pass"])
            self.assertIn(
                "tooltips/slot_001_1.png",
                outcome.scan.slots[1].images,
            )
            self.assertIn(
                "tooltip_recovered_on_pass_2",
                outcome.scan.slots[1].evidence,
            )

    def test_live_capturer_uses_fast_first_pass_and_full_recovery(self) -> None:
        class PassAwareCapturer:
            def __init__(self):
                self.calls = []

            def capture_fast(self, slot):
                self.calls.append("fast")
                return TooltipFrames(slot, ())

            def capture(self, slot):
                self.calls.append("full")
                return TooltipFrames(
                    slot,
                    (Image.new("RGB", (80, 60), "black"),),
                )

        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            capturer = PassAwareCapturer()
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                FakeShopCapturer(),
                capturer,
                repository,
                ShopTracker(),
                FakeInput(),
            )

            outcome = pipeline.capture(TrackedShop("shop-1", (300, 400)))

            self.assertEqual(outcome.scan.status, ScanStatus.CAPTURED)
            self.assertEqual(capturer.calls, ["fast", "full"])

    def test_successful_first_pass_does_not_run_recovery(self) -> None:
        class CountingCapturer(FakeTooltipCapturer):
            def __init__(self):
                self.calls = 0

            def capture(self, slot):
                self.calls += 1
                return super().capture(slot)

        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            capturer = CountingCapturer()
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                FakeShopCapturer(),
                capturer,
                repository,
                ShopTracker(),
                FakeInput(),
            )

            outcome = pipeline.capture(TrackedShop("shop-1", (300, 400)))
            events = (
                repository.scan_dir(outcome.scan.scan_id)
                / "raw_events.jsonl"
            ).read_text(encoding="utf-8")

            self.assertEqual(capturer.calls, 1)
            self.assertNotIn('"event":"recovery_pass"', events)
            self.assertNotIn('"event":"recovery_started"', events)

    def test_recovery_does_not_repeat_slots_already_captured(self) -> None:
        class TwoSlotShopCapturer(FakeShopCapturer):
            def occupied_slots(self, image):
                return [
                    OccupiedSlot(slot=0, row=0, column=0, residual=20.0),
                    OccupiedSlot(slot=1, row=0, column=1, residual=20.0),
                ]

        class SelectiveCapturer:
            def __init__(self):
                self.calls = []

            def capture(self, slot):
                self.calls.append(slot.slot)
                if slot.slot == 1 and self.calls.count(1) == 1:
                    return TooltipFrames(slot, ())
                return TooltipFrames(
                    slot,
                    (Image.new("RGB", (80, 60), "black"),),
                )

        with tempfile.TemporaryDirectory() as temp:
            capturer = SelectiveCapturer()
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                TwoSlotShopCapturer(),
                capturer,
                ScanRepository(Path(temp) / "scans"),
                ShopTracker(),
                FakeInput(),
            )

            outcome = pipeline.capture(TrackedShop("shop-1", (300, 400)))

            # Bramka ≥2: grupa 2 slotów, slot 0 OK, slot 1 pudło (pass 1).
            # Recovery dohoverowuje slot 1 (pass 2 OK). Bramka ≥2 widzi 2 udane
            # odczyty → grupa gotowa, bez deferred/stack_covered_by.
            self.assertEqual(capturer.calls, [0, 1, 1])
            self.assertEqual(outcome.scan.captured_slots, 2)

    def test_group_reads_topup_uses_buy_popup_without_scanning_whole_stack(self) -> None:
        slots = [
            OccupiedSlot(slot=0, row=0, column=0, residual=20.0),
            OccupiedSlot(slot=1, row=0, column=1, residual=20.0),
            OccupiedSlot(slot=2, row=0, column=2, residual=20.0),
        ]

        class ThreeSlotShopCapturer(FakeShopCapturer):
            def occupied_slots(self, image):
                return list(slots)

        class OneGoodTooltipCapturer:
            def __init__(self):
                self.calls = []

            def capture(self, slot):
                self.calls.append(slot.slot)
                if slot.slot == 0:
                    return TooltipFrames(
                        slot,
                        (Image.new("RGB", (80, 60), "black"),),
                    )
                return TooltipFrames(slot, ())

        popup_calls = []

        def fake_popup(self, scan, slot):
            popup_calls.append(slot.slot)
            obs = ItemObservation(
                slot=slot.slot,
                row=slot.row,
                column=slot.column,
                images=[],
                status=ScanStatus.CAPTURED,
                evidence=["buy_popup"],
            )
            obs.validation = {
                "status": "provisional",
                "source": "buy_popup",
                "item": "Skrzynia Testowa",
                "unit_price": 100,
                "quantity": 10,
            }
            return obs

        with tempfile.TemporaryDirectory() as temp:
            capturer = OneGoodTooltipCapturer()
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                ThreeSlotShopCapturer(),
                capturer,
                ScanRepository(Path(temp) / "scans"),
                ShopTracker(),
                FakeInput(),
                window_box=(0, 0, 400, 300),
            )
            pipeline.enable_phase_b(quorum=2)

            with patch("scanner.pipeline.group_slots_by_icon", return_value=[slots]):
                with patch.object(
                    GameCapturePipeline,
                    "_read_slot_from_buy_popup",
                    fake_popup,
                ):
                    outcome = pipeline.capture(TrackedShop("shop-1", (300, 400)))

            self.assertEqual(capturer.calls, [0, 1])
            self.assertEqual(popup_calls, [1])
            self.assertIn("buy_popup", outcome.scan.slots[1].evidence)
            self.assertIn(
                "popup_topup:group_reads_topup",
                outcome.scan.slots[1].evidence,
            )
            self.assertEqual(
                outcome.scan.slots[2].evidence,
                ["stack_representative"],
            )

            events = "\n".join(
                (Path(temp) / "scans" / outcome.scan.scan_id / "raw_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertIn('"event":"popup_topup_attempted"', events)
            self.assertIn('"source":"buy_popup"', events)

    def test_auto_logs_legacy_pick_from_same_selectable_tracks(self) -> None:
        class Detector:
            def detect(self, image, *, screen_offset):
                return [
                    ShopCandidate(
                        (600, 500),
                        (600, 500),
                        20,
                        180.0,
                        1.0,
                        False,
                    ),
                    ShopCandidate(
                        (500, 500),
                        (500, 500),
                        20,
                        100.0,
                        -1.0,
                        True,
                    ),
                ]

            def mask_image(self, image):
                return Image.new("L", image.size)

        class Pipeline:
            close_blocked = False
            stall_count = 0
            stall_blocked = False

            def capture(self, track):
                return CaptureOutcome(
                    ShopScan("scan-1", status=ScanStatus.CAPTURED)
                )

        class Diagnostics:
            def record_detection(
                self,
                image,
                mask,
                candidates,
                tracks,
                selected,
                *,
                screen_offset,
                legacy_pick,
            ):
                self.selected = selected.track_id
                self.legacy_pick = legacy_pick.track_id

            def record_capture(self, track, outcome):
                pass

        diagnostics = Diagnostics()
        loop = AutonomousMarketLoop(
            Detector(),
            ShopTracker(),
            Pipeline(),
            object(),
            lambda: (Image.new("RGB", (800, 600)), (0, 0)),
            diagnostics=diagnostics,
        )

        loop.scan_current_view(max_shops=1)

        # Faza 1 (RESCAN_SCATTER_PLAN): peek_unvisited bierze pierwszy eligible.
        # Detektor sortuje (likely_false, distance) → shop-00001 (False,~200) przed shop-00002 (True,~100)
        # Diagnostics (kontrfaktyczne) sortuje tylko po distance → shop-00002 (100) przed shop-00001 (180)
        self.assertEqual(diagnostics.selected, "shop-00001")
        self.assertEqual(diagnostics.legacy_pick, "shop-00002")

    def test_zero_tooltips_does_not_mark_shop_as_visited(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            tracker = ShopTracker()
            track = TrackedShop("shop-1", (300, 400))
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                FakeShopCapturer(),
                EmptyTooltipCapturer(),
                repository,
                tracker,
                FakeInput(),
            )

            outcome = pipeline.capture(track)

            self.assertEqual(outcome.scan.status, ScanStatus.FAILED)
            self.assertEqual(
                outcome.scan.error.reason, "insufficient_tooltip_yield"
            )
            self.assertFalse(track.visited)
            self.assertFalse(track.failed)
            self.assertEqual(track.attempts, 1)

    def test_failed_open_saves_passive_ui_probe_without_pressing_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            input_backend = FakeInput()
            pipeline = GameCapturePipeline(
                FailedOpenInteractor(),
                FakeShopCapturerWithScreen(),
                FakeTooltipCapturer(),
                repository,
                ShopTracker(),
                input_backend,
                window_box=(0, 0, 640, 480),
            )

            outcome = pipeline.capture(TrackedShop("npc-or-false", (300, 400)))
            scan_dir = repository.scan_dir(outcome.scan.scan_id)
            events = [
                json.loads(line)
                for line in (scan_dir / "raw_events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            probe_events = [
                event
                for event in events
                if event["event"] == "failed_open_ui_probe"
            ]

            self.assertEqual(outcome.scan.status, ScanStatus.FAILED)
            self.assertTrue((scan_dir / "frames" / "failed_open_ui.png").exists())
            self.assertEqual(len(probe_events), 1)
            self.assertEqual(probe_events[0]["probe_score"], 7.0)
            self.assertTrue(probe_events[0]["passive"])
            self.assertEqual(probe_events[0]["action"], "none")
            self.assertNotIn(("press", "esc"), input_backend.actions)

    def test_capture_finishes_before_analysis_and_closes_shop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            tracker = ShopTracker()
            track = TrackedShop("shop-1", (300, 400))
            input_backend = FakeInput()
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                FakeShopCapturer(),
                FakeTooltipCapturer(),
                repository,
                tracker,
                input_backend,
                seller_provider=lambda: "Kocur",
            )

            outcome = pipeline.capture(track)

            self.assertEqual(outcome.scan.status, ScanStatus.CAPTURED)
            self.assertEqual(outcome.scan.seller, "Kocur")
            self.assertEqual(outcome.scan.captured_slots, 1)
            self.assertEqual(outcome.scan.slots[1].icon_group, 1)
            self.assertEqual(len(outcome.scan.slots[1].images), 2)
            self.assertTrue(track.visited)
            self.assertIn(("press", "esc"), input_backend.actions)
            self.assertEqual(repository.load(outcome.scan.scan_id).status, ScanStatus.CAPTURED)

    def test_identical_stack_reuses_representative_tooltip(self) -> None:
        class TwoSlotShopCapturer(FakeShopCapturer):
            def occupied_slots(self, image):
                return [
                    OccupiedSlot(slot=0, row=0, column=0, residual=20.0),
                    OccupiedSlot(slot=1, row=0, column=1, residual=20.0),
                ]

        class StackTooltipCapturer:
            def capture(self, slot):
                return TooltipFrames(
                    slot, (Image.new("RGB", (80, 60), "black"),)
                )

            def capture_stack_member(self, slot, reference_frame):
                return TooltipFrames(
                    slot, (reference_frame.copy(),), matched_reference=True
                )

        with tempfile.TemporaryDirectory() as temp:
            repository = ScanRepository(Path(temp) / "scans")
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                TwoSlotShopCapturer(),
                StackTooltipCapturer(),
                repository,
                ShopTracker(),
                FakeInput(),
            )

            outcome = pipeline.capture(TrackedShop("shop-1", (300, 400)))

            first = outcome.scan.slots[0]
            second = outcome.scan.slots[1]
            self.assertEqual(outcome.scan.status, ScanStatus.CAPTURED)
            self.assertEqual(first.icon_group, second.icon_group)
            self.assertEqual(first.images, second.images)
            self.assertEqual(second.evidence, ["icon_duplicate_of:0"])
            tooltip_files = list(
                (repository.scan_dir(outcome.scan.scan_id) / "tooltips").glob("*.png")
            )
            self.assertEqual(len(tooltip_files), 1)

    def test_analysis_worker_uses_external_engine_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = ScanRepository(root / "scans")
            tracker = ShopTracker()
            worker = AnalysisWorker(
                repository,
                VerifyingEngine(),
                exporter=CSVExporter(root / "ceny.csv"),
            )
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                FakeShopCapturer(),
                FakeTooltipCapturer(),
                repository,
                tracker,
                FakeInput(),
                analysis_queue=worker,
            )

            outcome = pipeline.capture(TrackedShop("shop-1", (300, 400)))
            worker.join()
            worker.stop(1.0)
            restored = repository.load(outcome.scan.scan_id)

            self.assertEqual(outcome.scan.status, ScanStatus.QUEUED)
            self.assertEqual(restored.status, ScanStatus.VERIFIED)
            self.assertTrue((root / "ceny.csv").exists())

    def test_auto_limit_counts_captured_shops_not_failed_clicks(self) -> None:
        class Detector:
            def detect(self, image, *, screen_offset):
                return []

        class Tracker:
            def __init__(self):
                self.index = 0

            def update(self, candidates):
                return []

            def next_unvisited(self, visible):
                if self.index >= 2:
                    return None
                return TrackedShop(f"shop-{self.index}", (300, 400))

        class Pipeline:
            stall_count = 0
            stall_blocked = False

            def __init__(self, tracker):
                self.tracker = tracker

            def capture(self, track):
                status = (
                    ScanStatus.FAILED
                    if self.tracker.index == 0
                    else ScanStatus.CAPTURED
                )
                scan = ShopScan(
                    scan_id=f"scan-{self.tracker.index}",
                    status=status,
                )
                self.tracker.index += 1
                return CaptureOutcome(scan)

        class Movement:
            def __init__(self):
                self.steps = []

            def execute(self, key, duration, settle):
                self.steps.append((key, duration, settle))

        tracker = Tracker()
        movement = Movement()
        loop = AutonomousMarketLoop(
            Detector(),
            tracker,
            Pipeline(tracker),
            movement,
            lambda: (Image.new("RGB", (10, 10)), (0, 0)),
        )

        outcomes = loop.run(
            (MovementStep("d", 0.2, 0.1, 0, 0, "horizontal"),),
            max_shops=1,
        )

        self.assertEqual(len(outcomes), 2)
        self.assertFalse(outcomes[0].successful)
        self.assertTrue(outcomes[1].successful)

    def test_auto_walks_after_failed_shop_attempt(self) -> None:
        class Detector:
            def detect(self, image, *, screen_offset):
                return []

        class Tracker:
            offered = False

            def update(self, candidates):
                return []

            def next_unvisited(self, visible):
                if self.offered:
                    return None
                self.offered = True
                return TrackedShop("shop-failed", (300, 400))

        class Pipeline:
            stall_count = 0
            stall_blocked = False
            current_position = None

            def capture(self, track):
                return CaptureOutcome(
                    ShopScan("scan-failed", status=ScanStatus.FAILED)
                )

            def read_current_position(self, step_key=None):
                return None

        class Movement:
            def __init__(self):
                self.steps = []

            def execute(self, key, duration, settle):
                self.steps.append((key, duration, settle))

        movement = Movement()
        loop = AutonomousMarketLoop(
            Detector(),
            Tracker(),
            Pipeline(),
            movement,
            lambda: (Image.new("RGB", (10, 10)), (0, 0)),
        )

        loop.run(
            (MovementStep("d", 0.2, 0.1, 0, 0, "horizontal"),),
            max_shops=1,
        )

        self.assertEqual(movement.steps, [("d", 0.2, 0.1)])

    def test_auto_does_not_walk_when_shop_failed_to_close(self) -> None:
        class Detector:
            def detect(self, image, *, screen_offset):
                return []

        class Tracker:
            offered = False

            def update(self, candidates):
                return []

            def next_unvisited(self, visible):
                if self.offered:
                    return None
                self.offered = True
                return TrackedShop("shop-open", (300, 400))

        class Pipeline:
            close_blocked = False
            current_position = None

            def capture(self, track):
                self.close_blocked = True
                return CaptureOutcome(
                    ShopScan("scan-open", status=ScanStatus.CAPTURED)
                )

            def read_current_position(self, step_key=None):
                return None

        class Movement:
            def __init__(self):
                self.steps = []

            def execute(self, key, duration, settle):
                self.steps.append((key, duration, settle))

        movement = Movement()
        loop = AutonomousMarketLoop(
            Detector(),
            Tracker(),
            Pipeline(),
            movement,
            lambda: (Image.new("RGB", (10, 10)), (0, 0)),
        )

        loop.run(
            (MovementStep("d", 0.2, 0.1, 0, 0, "horizontal"),),
            max_shops=2,
        )

        # Po F4: close_blocked NIE przerywa sesji. Anti-stall wymusza
        # _force_step (s, 0.3, 0.5) przed normalnym krokiem WASD.
        self.assertEqual(movement.steps, [
            ("s", 0.3, 0.5),   # wymuszony krok anti-stall
            ("d", 0.2, 0.1),   # normalny krok trasy
        ])

    def test_coverage_drive_uses_same_anti_stall_without_walk_route(self) -> None:
        class Detector:
            def detect(self, image, *, screen_offset):
                return [
                    ShopCandidate(
                        screen_position=(300, 400),
                        local_position=(300, 400),
                        area=50,
                        distance=10,
                    )
                ]

            def mask_image(self, image):
                return image

        class Pipeline:
            close_blocked = False
            stall_blocked = False
            stall_count = 0
            current_position = (400, 720)
            position_source = "ocr"
            units_per_step = 3.2
            _odometry_vectors = {"s": (2.59, 1.87)}

            def capture(self, track):
                track.visited = True
                self.close_blocked = True
                return CaptureOutcome(
                    ShopScan("scan-open", status=ScanStatus.CAPTURED)
                )

            def read_current_position(self, step_key=None):
                self.current_position = (403, 722)
                return self.current_position

        class Movement:
            def __init__(self):
                self.steps = []

            def execute(self, key, duration, settle):
                self.steps.append((key, duration, settle))

        class CoverageMap:
            envelope = (348, 501, 672, 794)

            def blocked_ahead(self, current, next_pos, lookahead=1.0):
                return False

            def is_blocked(self, cell):
                return False

            def cell_of(self, pos):
                return (0, 0)

            def should_skip_click(self, pos, *, dup_floor=None):
                return False

            def dups_in_cell(self, cell):
                return 0

        movement = Movement()
        loop = AutonomousMarketLoop(
            Detector(),
            ShopTracker(),
            Pipeline(),
            movement,
            lambda: (Image.new("RGB", (10, 10)), (0, 0)),
        )
        loop.set_coverage_map(CoverageMap())

        outcomes = loop.run((), max_shops=1)

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(movement.steps, [("s", 0.3, 0.5)])

    def test_coverage_drive_saturates_touched_cells_before_uncovered(self) -> None:
        class Detector:
            def detect(self, image, *, screen_offset):
                return []

            def mask_image(self, image):
                return image

        class Pipeline:
            close_blocked = False
            stall_blocked = False
            stall_count = 0
            current_position = (400, 720)
            position_source = "ocr"

            def capture(self, track):
                raise AssertionError("no tracks should be captured")

        class CoverageMap:
            envelope = (348, 501, 672, 794)

            def __init__(self):
                self.prefer_uncovered_values = []
                self.dup_floors = []

            def path_to_next_target(self, pos, **kwargs):
                self.prefer_uncovered_values.append(kwargs.get("prefer_uncovered"))
                self.dup_floors.append(kwargs.get("dup_floor"))
                return None

            def all_done(self, *, dup_floor=None):
                return False

        coverage = CoverageMap()
        loop = AutonomousMarketLoop(
            Detector(),
            ShopTracker(),
            Pipeline(),
            object(),
            lambda: (Image.new("RGB", (10, 10)), (0, 0)),
        )
        loop.set_coverage_map(coverage)

        loop.run((), max_shops=1)

        self.assertEqual(coverage.prefer_uncovered_values, [False])
        self.assertEqual(coverage.dup_floors, [4])


class ForceCloseTests(unittest.TestCase):
    """Testy eskalacji _force_close (F4)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = ScanRepository(Path(self.temp.name) / "scans")
        self.input = FakeInput()
        self.geometry = GridGeometry(origin=(1000, 200))

    def tearDown(self):
        self.temp.cleanup()

    def _make_pipeline(self, *, wait_closed_result, is_foreground=True):
        """Zbuduj GameCapturePipeline z atrapami do testów _force_close."""

        class FakeFocus:
            def is_foreground(self_):
                return is_foreground

        class FakeInteractor:
            def __init__(self_):
                self_.wait_closed_calls = []
            def wait_closed(self_, timeout):
                self_.wait_closed_calls.append(timeout)
                if callable(wait_closed_result):
                    return wait_closed_result(len(self_.wait_closed_calls))
                return wait_closed_result

        class FakeCapturer:
            geometry = self.geometry

        pipeline = GameCapturePipeline(
            interactor=FakeInteractor(),
            shop_capturer=FakeCapturer(),
            tooltip_capturer=EmptyTooltipCapturer(),
            repository=self.repo,
            tracker=ShopTracker(),
            input_backend=self.input,
            close_key="esc",
            focus=FakeFocus(),
        )
        return pipeline, pipeline.interactor

    # ------------------------------------------------------------------
    def test_force_close_faza1_esc_otwiera(self) -> None:
        pipeline, interactor = self._make_pipeline(wait_closed_result=True)
        closed = pipeline._force_close("scan-01")
        self.assertTrue(closed)
        self.assertIn(("press", "esc"), self.input.actions)
        events = [json.loads(line) for line in
                  (self.repo.scan_dir("scan-01") / "raw_events.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["event"] == "shop_close_confirmed" for e in events))

    def test_force_close_faza1_esc_drugi_otwiera(self) -> None:
        results = [False, True]  # pierwszy Esc pada, drugi zamyka
        pipeline, _ = self._make_pipeline(wait_closed_result=lambda n: results[n-1])
        closed = pipeline._force_close("scan-02")
        self.assertTrue(closed)
        presses = [a for a in self.input.actions if a[0] == "press"]
        self.assertEqual(len(presses), 2)  # dwa Esc

    def test_force_close_faza2_klik_X_zamyka(self) -> None:
        results = [False, False, True]  # Esc×2 padły, klik X zamyka
        pipeline, interactor = self._make_pipeline(
            wait_closed_result=lambda n: results[n-1]
        )
        closed = pipeline._force_close("scan-03")
        self.assertTrue(closed)
        # Sprawdź, że klik trafił w przycisk X
        shop_x, shop_y, shop_w, shop_h = self.geometry.shop_box
        expected_x = shop_x + shop_w - 20
        expected_y = shop_y + 10
        moves = [a for a in self.input.actions if a[0] == "move"]
        self.assertTrue(any(
            abs(m[1] - expected_x) <= 2 and abs(m[2] - expected_y) <= 2
            for m in moves
        ), f"brak ruchu na X ({expected_x}, {expected_y}) w {moves}")
        events = [json.loads(line) for line in
                  (self.repo.scan_dir("scan-03") / "raw_events.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(
            e["event"] == "shop_close_confirmed" and e.get("method") == "button"
            for e in events
        ))

    def test_force_close_faza3_combo_esc_X_zamyka(self) -> None:
        results = [False, False, False, True]  # Esc×2, klik X, combo Esc+X zamyka
        pipeline, _ = self._make_pipeline(
            wait_closed_result=lambda n: results[n-1]
        )
        closed = pipeline._force_close("scan-04")
        self.assertTrue(closed)
        events = [json.loads(line) for line in
                  (self.repo.scan_dir("scan-04") / "raw_events.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(
            e["event"] == "shop_close_confirmed" and e.get("method") == "esc_button"
            for e in events
        ))

    def test_force_close_wszystko_pada(self) -> None:
        pipeline, _ = self._make_pipeline(wait_closed_result=False)
        closed = pipeline._force_close("scan-05")
        self.assertFalse(closed)
        events = [json.loads(line) for line in
                  (self.repo.scan_dir("scan-05") / "raw_events.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(
            e["event"] == "shop_close_escalation_failed" and e["attempts"] == 3
            for e in events
        ))

    def test_force_close_utrata_fokusa(self) -> None:
        pipeline, _ = self._make_pipeline(
            wait_closed_result=True, is_foreground=False
        )
        closed = pipeline._force_close("scan-06")
        self.assertFalse(closed)
        events = [json.loads(line) for line in
                  (self.repo.scan_dir("scan-06") / "raw_events.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(
            e["event"] == "shop_close_skipped" for e in events
        ))


    def test_durable_dedup_skips_known_fresh_shop(self) -> None:
        # v10.57: sklep znany z rejestru (poprzedni bieg) -> duplicate known_fresh
        # przed capture_grid(), bez hoverowania.
        known: set[str] = set()

        def known_fresh(fp: str) -> bool:
            return fp in known

        with tempfile.TemporaryDirectory() as temp:
            tracker = ShopTracker()
            pipeline = GameCapturePipeline(
                OpenInteractor(),
                FakeShopCapturer(),
                FakeTooltipCapturer(),
                ScanRepository(Path(temp) / "scans"),
                tracker,
                FakeInput(),
            )
            pipeline.set_known_fresh(known_fresh)

            # Pierwsze przejscie: sklep nieznany -> capture normalny
            outcome1 = pipeline.capture(TrackedShop("shop-1", (300, 400)))
            self.assertFalse(outcome1.duplicate)
            self.assertEqual(outcome1.scan.status, ScanStatus.CAPTURED)
            fp = outcome1.scan.shop_fingerprint
            self.assertIsNotNone(fp)
            known.add(fp)

            # Drugie przejscie: TEN SAM sklep, ale fresh tracker -> durable dedup
            tracker2 = ShopTracker()  # nowy tracker = sesyjnie nie wie
            pipeline2 = GameCapturePipeline(
                OpenInteractor(),
                FakeShopCapturer(),
                FakeTooltipCapturer(),
                ScanRepository(Path(temp) / "scans2"),
                tracker2,
                FakeInput(),
            )
            pipeline2.set_known_fresh(known_fresh)
            outcome2 = pipeline2.capture(TrackedShop("shop-1", (300, 400)))
            self.assertTrue(outcome2.duplicate)
            self.assertEqual(outcome2.scan.status, ScanStatus.FAILED)
            events2 = [
                json.loads(line) for line in
                (Path(temp) / "scans2" / outcome2.scan.scan_id / "raw_events.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(
                e.get("error", {}).get("reason") == "duplicate_known_fresh"
                for e in events2
            ), f"brak 'duplicate_known_fresh' w {events2}")


if __name__ == "__main__":
    unittest.main()
