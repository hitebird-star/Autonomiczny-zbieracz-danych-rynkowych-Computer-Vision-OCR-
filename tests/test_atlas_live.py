from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scanner.analysis.coord_reader import CoordParse
from scanner.atlas.calibrator_live import (
    AtlasLiveCalibrator,
    CalibrationMovePlan,
    match_shop_deltas,
)
from scanner.atlas.contracts import FrameSnapshot
from scanner.atlas.live_feed import (
    AtlasLiveFeed,
    _read_coord_attempts_for_atlas,
    _salvage_coord_from_texts,
)
from scanner.detection.shop_detector import ShopCandidate
from scanner.runtime import WindowRect


class FakeScreen:
    def grab(self, box):
        self.box = box
        return Image.new("RGB", (box[2], box[3]), "black")


class FakeWindow:
    def locate(self):
        return WindowRect(100, 200, 800, 600)


class FakeDetector:
    def detect(self, image, *, screen_offset=(0, 0)):
        return [
            ShopCandidate(
                screen_position=(screen_offset[0] + 20, screen_offset[1] + 30),
                local_position=(20, 30),
                area=50,
                distance=12.0,
                hybrid_score=0.25,
                likely_false=False,
            ),
            ShopCandidate(
                screen_position=(screen_offset[0] + 60, screen_offset[1] + 70),
                local_position=(60, 70),
                area=30,
                distance=40.0,
                hybrid_score=-0.5,
                likely_false=True,
            ),
        ]


class FakeInput:
    def __init__(self):
        self.events = []

    def key_down(self, key):
        self.events.append(("down", key))

    def key_up(self, key):
        self.events.append(("up", key))


class FakeClock:
    def __init__(self):
        self.now = 10.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class AtlasLiveFeedTests(unittest.TestCase):
    def test_capture_once_builds_frame_snapshot(self) -> None:
        screen = FakeScreen()
        feed = AtlasLiveFeed(
            screen=screen,
            window=FakeWindow(),
            detector=FakeDetector(),
        )

        with patch(
            "scanner.atlas.live_feed._read_coord_for_atlas_with_trace",
            return_value=(
                CoordParse(x=450, y=720, channel=1, map_name="Glevia2 Farm"),
                {"stages": []},
            ),
        ):
            snapshot = feed.capture_once()

        self.assertEqual(screen.box, (100, 200, 800, 600))
        self.assertEqual(snapshot.window_rect, (100, 200, 800, 600))
        self.assertEqual(snapshot.player_game, (450.0, 720.0))
        self.assertEqual(snapshot.shops_screen, ((20.0, 30.0), (60.0, 70.0)))
        self.assertEqual(snapshot.shop_observations[0].screen_position, (120.0, 230.0))
        self.assertTrue(snapshot.shop_observations[1].likely_false)
        self.assertEqual(feed.last_window_rect, (100, 200, 800, 600))
        self.assertIsNotNone(feed.last_client_image)
        self.assertEqual(feed.last_snapshot, snapshot)
        self.assertEqual(feed.last_coord_trace["final_accept"]["reason"], "accepted")

    def test_capture_once_rejects_implausible_coord_jump(self) -> None:
        feed = AtlasLiveFeed(
            screen=FakeScreen(),
            window=FakeWindow(),
            detector=FakeDetector(),
        )

        with patch(
            "scanner.atlas.live_feed._read_coord_for_atlas_with_trace",
            return_value=(
                CoordParse(x=450, y=720, channel=None, map_name=None),
                {"stages": []},
            ),
        ):
            self.assertEqual(feed.capture_once().player_game, (450.0, 720.0))
        with patch(
            "scanner.atlas.live_feed._read_coord_for_atlas_with_trace",
            return_value=(
                CoordParse(x=900, y=1200, channel=None, map_name=None),
                {"stages": []},
            ),
        ):
            self.assertIsNone(feed.capture_once().player_game)
        self.assertEqual(feed.last_coord_trace["final_accept"]["reason"], "jump_too_large")

    def test_capture_once_uses_startup_fallback_when_default_coord_ocr_misses(self) -> None:
        feed = AtlasLiveFeed(
            screen=FakeScreen(),
            window=FakeWindow(),
            detector=FakeDetector(),
        )
        calls = []

        def fake_attempts(image, attempts, bounds, **kwargs):
            calls.append(attempts)
            if len(calls) == 1:
                return None
            return CoordParse(x=445, y=706, channel=None, map_name=None)

        with patch(
            "scanner.atlas.live_feed._read_coord_attempts_for_atlas",
            side_effect=fake_attempts,
        ):
            snapshot = feed.capture_once()

        self.assertEqual(snapshot.player_game, (445.0, 706.0))
        self.assertEqual(len(calls), 2)

    def test_coord_salvage_combines_partial_ocr_reads(self) -> None:
        parsed = _salvage_coord_from_texts(
            [
                "0 (448, paz)",
                "707)",
                "1448",
                "/ (44B,frA7)",
                "707)",
            ],
            (348, 501, 672, 794),
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual((parsed.x, parsed.y), (448, 707))

    def test_atlas_coord_attempts_skip_parsed_but_out_of_bounds_read(self) -> None:
        attempts = (
            ((0.0, 0.0, 1.0, 1.0), 1, None),
            ((0.0, 0.0, 1.0, 1.0), 1, "white_outline"),
        )

        with patch(
            "scanner.atlas.live_feed.coord_reader._ocr_text",
            side_effect=["(457,927)", "(457* 727)"],
        ):
            parsed = _read_coord_attempts_for_atlas(
                Image.new("RGB", (20, 20), "black"),
                attempts,
                (300, 600, 650, 800),
            )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual((parsed.x, parsed.y), (457, 727))


class AtlasLiveCalibratorTests(unittest.TestCase):
    def test_match_shop_deltas_uses_nearest_unused_after_point(self) -> None:
        deltas = match_shop_deltas(
            ((100.0, 100.0), (200.0, 100.0), (500.0, 500.0)),
            ((92.0, 104.0), (192.0, 105.0), (900.0, 900.0)),
            max_distance_px=20.0,
        )

        self.assertEqual(deltas, [(-8.0, 4.0), (-8.0, 5.0)])

    def test_match_shop_deltas_prefers_dominant_motion_over_nearest_confusers(self) -> None:
        deltas = match_shop_deltas(
            (
                (0.0, 0.0),
                (100.0, 0.0),
                (200.0, 0.0),
                (300.0, 0.0),
            ),
            (
                (0.0, 0.0),      # confuser: nearest for first point
                (100.0, 0.0),    # confuser: nearest for second point
                (14.0, 6.0),
                (114.0, 6.0),
                (214.0, 6.0),
                (314.0, 6.0),
            ),
            max_distance_px=40.0,
            consensus_radius_px=8.0,
        )

        self.assertEqual(len(deltas), 4)
        self.assertTrue(all(abs(dx - 14.0) <= 1e-9 for dx, _ in deltas))
        self.assertTrue(all(abs(dy - 6.0) <= 1e-9 for _, dy in deltas))

    def test_match_shop_deltas_uses_expected_delta_to_avoid_grid_alias(self) -> None:
        before = ((0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0))
        after = (
            (50.0, 50.0),
            (150.0, 50.0),
            (250.0, 50.0),
            (350.0, 50.0),
            (-70.0, -80.0),
            (30.0, -80.0),
            (130.0, -80.0),
            (230.0, -80.0),
        )

        deltas = match_shop_deltas(
            before,
            after,
            max_distance_px=130.0,
            consensus_radius_px=12.0,
            expected_delta=(50.0, 50.0),
        )

        self.assertEqual(len(deltas), 4)
        self.assertTrue(all(abs(dx - 50.0) <= 1e-9 for dx, _ in deltas))
        self.assertTrue(all(abs(dy - 50.0) <= 1e-9 for _, dy in deltas))

    def test_match_shop_deltas_rejects_grid_alias_outside_tight_expected_radius(self) -> None:
        deltas = match_shop_deltas(
            ((0.0, 0.0), (100.0, 0.0)),
            ((60.0, 0.0), (160.0, 0.0)),
            max_distance_px=80.0,
            consensus_radius_px=8.0,
            expected_delta=(20.0, 0.0),
        )

        self.assertEqual(deltas, [])

    def test_one_track_match_does_not_drift_existing_hint(self) -> None:
        calibrator = AtlasLiveCalibrator(
            input_backend=FakeInput(),
            snapshot_provider=lambda: None,
            plan=CalibrationMovePlan(min_shop_tracks=2),
            clock=FakeClock(),
        )

        strong = calibrator.observation_from_snapshots(
            "w",
            FrameSnapshot(
                timestamp="before",
                window_rect=(0, 0, 800, 600),
                player_game=(450.0, 720.0),
                shops_screen=((100.0, 100.0), (200.0, 100.0)),
            ),
            FrameSnapshot(
                timestamp="after",
                window_rect=(0, 0, 800, 600),
                player_game=(454.0, 720.0),
                shops_screen=((90.0, 100.0), (190.0, 100.0)),
            ),
        )
        self.assertIsNotNone(strong)
        self.assertEqual(calibrator._screen_delta_hint_by_key["w"], (-10.0, 0.0))

        weak = calibrator.observation_from_snapshots(
            "w",
            FrameSnapshot(
                timestamp="before2",
                window_rect=(0, 0, 800, 600),
                player_game=(454.0, 720.0),
                shops_screen=((100.0, 100.0),),
            ),
            FrameSnapshot(
                timestamp="after2",
                window_rect=(0, 0, 800, 600),
                player_game=(458.0, 720.0),
                shops_screen=((88.0, 100.0),),
            ),
        )

        self.assertIsNotNone(weak)
        self.assertEqual(calibrator._screen_delta_hint_by_key["w"], (-10.0, 0.0))

    def test_one_track_grid_alias_is_rejected_even_after_hint_exists(self) -> None:
        calibrator = AtlasLiveCalibrator(
            input_backend=FakeInput(),
            snapshot_provider=lambda: None,
            plan=CalibrationMovePlan(min_shop_tracks=2),
            clock=FakeClock(),
        )
        self.assertIsNotNone(
            calibrator.observation_from_snapshots(
                "w",
                FrameSnapshot(
                    timestamp="before",
                    window_rect=(0, 0, 800, 600),
                    player_game=(450.0, 720.0),
                    shops_screen=((100.0, 100.0), (200.0, 100.0)),
                ),
                FrameSnapshot(
                    timestamp="after",
                    window_rect=(0, 0, 800, 600),
                    player_game=(454.0, 720.0),
                    shops_screen=((90.0, 100.0), (190.0, 100.0)),
                ),
            )
        )

        alias = calibrator.observation_from_snapshots(
            "w",
            FrameSnapshot(
                timestamp="before2",
                window_rect=(0, 0, 800, 600),
                player_game=(454.0, 720.0),
                shops_screen=((100.0, 100.0),),
            ),
            FrameSnapshot(
                timestamp="after2",
                window_rect=(0, 0, 800, 600),
                player_game=(458.0, 720.0),
                shops_screen=((50.0, 100.0),),
            ),
        )

        self.assertIsNone(alias)
        self.assertEqual(calibrator.last_failure_reason, "after_tracks_missing:0/1")

    def test_move_once_sends_key_and_returns_observation(self) -> None:
        frames = iter(
            [
                FrameSnapshot(
                    timestamp="before",
                    window_rect=(0, 0, 800, 600),
                    player_game=(450.0, 720.0),
                    shops_screen=((100.0, 100.0), (200.0, 100.0)),
                ),
                FrameSnapshot(
                    timestamp="after",
                    window_rect=(0, 0, 800, 600),
                    player_game=(453.0, 721.0),
                    shops_screen=((94.0, 102.0), (194.0, 103.0)),
                ),
                FrameSnapshot(
                    timestamp="after-stable",
                    window_rect=(0, 0, 800, 600),
                    player_game=(453.0, 721.0),
                    shops_screen=((94.0, 102.0), (194.0, 103.0)),
                ),
            ]
        )
        input_backend = FakeInput()
        clock = FakeClock()
        calibrator = AtlasLiveCalibrator(
            input_backend=input_backend,
            snapshot_provider=lambda: next(frames),
            plan=CalibrationMovePlan(keys=("w",), hold_s=0.4, settle_s=0.2, min_shop_tracks=2),
            clock=clock,
        )

        observation = calibrator.move_once("w")

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(input_backend.events, [("down", "w"), ("up", "w")])
        self.assertEqual(clock.sleeps, [0.4, 0.2, 0.2])
        self.assertEqual(observation.key, "w")
        self.assertEqual(observation.delta_game, (3.0, 1.0))
        self.assertEqual(observation.delta_screen, ((-6.0, 2.0), (-6.0, 3.0)))
        self.assertGreater(observation.confidence, 0.0)

    def test_move_once_uses_frame_registration_when_images_are_available(self) -> None:
        rng = np.random.default_rng(123)
        before_image = rng.integers(0, 255, (220, 240, 3), dtype=np.uint8)
        after_image = np.roll(before_image, shift=(4, 6), axis=(0, 1))
        current_image = {"image": before_image}
        frames = iter(
            [
                FrameSnapshot(
                    timestamp="before",
                    window_rect=(0, 0, 240, 220),
                    player_game=(450.0, 720.0),
                    shops_screen=(),
                ),
                FrameSnapshot(
                    timestamp="after",
                    window_rect=(0, 0, 240, 220),
                    player_game=(453.0, 721.0),
                    shops_screen=(),
                ),
            ]
        )

        def snapshot_provider():
            frame = next(frames)
            if frame.timestamp == "after":
                current_image["image"] = after_image
            return frame

        calibrator = AtlasLiveCalibrator(
            input_backend=FakeInput(),
            snapshot_provider=snapshot_provider,
            image_provider=lambda: current_image["image"],
            plan=CalibrationMovePlan(keys=("w",), hold_s=0.1, settle_s=0.1),
            clock=FakeClock(),
        )

        observation = calibrator.move_once("w")

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(len(observation.delta_screen), 1)
        dx, dy = observation.delta_screen[0]
        self.assertAlmostEqual(dx, 6.0, delta=1.0)
        self.assertAlmostEqual(dy, 4.0, delta=1.0)
        self.assertGreater(observation.confidence, 0.5)

    def test_move_once_returns_none_without_enough_tracked_shops(self) -> None:
        frames = iter(
            [
                FrameSnapshot(
                    timestamp="before",
                    window_rect=(0, 0, 800, 600),
                    player_game=(450.0, 720.0),
                    shops_screen=((100.0, 100.0),),
                ),
                FrameSnapshot(
                    timestamp="after",
                    window_rect=(0, 0, 800, 600),
                    player_game=(451.0, 721.0),
                    shops_screen=((300.0, 300.0),),
                ),
            ]
        )
        calibrator = AtlasLiveCalibrator(
            input_backend=FakeInput(),
            snapshot_provider=lambda: next(frames),
            plan=CalibrationMovePlan(min_shop_tracks=1, max_match_distance_px=20.0),
            clock=FakeClock(),
        )

        self.assertIsNone(calibrator.move_once("d"))


if __name__ == "__main__":
    unittest.main()
