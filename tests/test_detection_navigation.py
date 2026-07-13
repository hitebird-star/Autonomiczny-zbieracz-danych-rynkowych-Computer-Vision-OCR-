from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from scanner.config import DetectorSettings
from scanner.detection import ShopCandidate, ShopDetector, ShopTracker
from scanner.models import ScanError
from scanner.navigation import (
    MovementController,
    RecoveryAction,
    RecoveryPolicy,
    SerpentineRoutePlanner,
)
from tests.fakes import FakeClock, FakeInput


class DetectionTests(unittest.TestCase):
    def test_color_detector_finds_and_merges_shop_blob(self) -> None:
        array = np.zeros((200, 300, 3), dtype=np.uint8)
        array[95:105, 145:157] = (150, 90, 40)
        detector = ShopDetector(
            DetectorSettings(
                area_min=20,
                area_max=300,
                width_min=5,
                width_max=30,
                height_min=4,
                height_max=30,
                min_radius=0,
                max_radius=80,
            )
        )

        candidates = detector.detect(Image.fromarray(array), screen_offset=(10, 20))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].screen_position, (160, 125))
        mask = detector.mask_image(Image.fromarray(array))
        self.assertEqual(mask.mode, "L")
        self.assertGreater(np.asarray(mask).max(), 0)

    def test_tracker_preserves_identity_and_skips_visited_fingerprint(self) -> None:
        detector = ShopDetector(
            DetectorSettings(
                area_min=20,
                area_max=300,
                width_min=5,
                width_max=30,
                height_min=4,
                height_max=30,
                min_radius=0,
                max_radius=100,
            )
        )
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        frame[95:105, 145:157] = (150, 90, 40)
        tracker = ShopTracker()
        first = tracker.update(detector.detect(Image.fromarray(frame)))[0]
        tracker.attach_fingerprint(first, "same-shop")
        tracker.mark_visited(first)

        second = tracker.update(detector.detect(Image.fromarray(frame)))[0]

        self.assertEqual(first.track_id, second.track_id)
        self.assertIsNone(tracker.next_unvisited([second]))

    def test_tracker_preserves_visited_shop_after_scene_shift(self) -> None:
        tracker = ShopTracker()
        first = tracker.update(
            [ShopCandidate((500, 500), (500, 500), 100, 120.0)]
        )[0]
        tracker.attach_fingerprint(first, "seller-a")
        tracker.mark_visited(first)

        shifted = tracker.update(
            [ShopCandidate((565, 545), (565, 545), 100, 115.0)]
        )[0]

        self.assertEqual(shifted.track_id, first.track_id)
        self.assertTrue(shifted.visited)
        self.assertIsNone(tracker.next_unvisited([shifted]))

    def test_tracker_globally_matches_candidates_before_copying_visited_state(self) -> None:
        """Bliższa para nie może dostać statusu sąsiedniego, odwiedzonego sklepu."""

        tracker = ShopTracker(match_radius=90)
        first = tracker.update(
            [
                ShopCandidate((0, 0), (0, 0), 20, 100.0),
                ShopCandidate((50, 0), (50, 0), 20, 110.0),
            ]
        )
        tracker.mark_visited(first[0])

        # Detektor podaje dalszy punkt pierwszy. Matcher zachłanny przypisałby
        # go do tracka (0,0), a idealnie pasujący drugi punkt dostałby błędny
        # status. Globalne parowanie najpierw łączy (0,0) z (0,0).
        shifted = tracker.update(
            [
                ShopCandidate((20, 0), (20, 0), 20, 100.0),
                ShopCandidate((0, 0), (0, 0), 20, 110.0),
            ]
        )

        self.assertFalse(shifted[0].visited)
        self.assertTrue(shifted[1].visited)
        self.assertIs(tracker.next_unvisited(shifted), shifted[0])

    def test_tracker_skips_nearby_points_after_terminal_false_target(self) -> None:
        tracker = ShopTracker(failed_zone_radius=80)
        failed = tracker.update(
            [ShopCandidate((500, 500), (500, 500), 17, 170.0)]
        )[0]
        tracker.mark_failed(failed, terminal=True)

        nearby = tracker.update(
            [ShopCandidate((535, 540), (535, 540), 18, 205.0)]
        )

        self.assertIsNone(tracker.next_unvisited(nearby))

    def test_peek_unvisited_supports_counterfactual_order(self) -> None:
        tracker = ShopTracker()
        visible = tracker.update(
            [
                ShopCandidate((600, 500), (600, 500), 20, 180.0),
                ShopCandidate((500, 500), (500, 500), 20, 100.0),
            ]
        )
        visible[1].visited = True

        pick = tracker.peek_unvisited(list(reversed(visible)))

        self.assertEqual(pick.track_id, visible[0].track_id)
        self.assertFalse(visible[0].visited)

    def test_color_detector_ignores_player_center_zone(self) -> None:
        array = np.zeros((200, 300, 3), dtype=np.uint8)
        array[95:105, 145:157] = (150, 90, 40)
        detector = ShopDetector(
            DetectorSettings(
                area_min=20,
                area_max=300,
                width_min=5,
                width_max=30,
                height_min=4,
                height_max=30,
                min_radius=80,
                max_radius=200,
            )
        )

        self.assertEqual(detector.detect(Image.fromarray(array)), [])

    def test_color_detector_relaxes_shape_when_legacy_filter_has_too_few_targets(self) -> None:
        array = np.zeros((220, 400, 3), dtype=np.uint8)
        # Perspektywiczny wierzch straganu z realnych runow bywa szerszy niz
        # legacy width_max=24 i ma area >200. Taki komponent ma wejsc przez
        # fallback, gdy standardowy filtr nie daje wystarczajacej liczby celow.
        array[104:116, 180:216] = (150, 90, 40)
        detector = ShopDetector(
            DetectorSettings(
                area_min=12,
                area_max=200,
                width_min=5,
                width_max=24,
                height_min=4,
                height_max=22,
                min_radius=0,
                max_radius=200,
                max_results=18,
                hybrid_enabled=False,
            )
        )

        candidates = detector.detect(Image.fromarray(array))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].local_position, (197, 115))


class NavigationTests(unittest.TestCase):
    def test_movement_always_releases_key(self) -> None:
        input_backend = FakeInput()
        MovementController(input_backend, clock=FakeClock()).hold("d", 0.6)
        self.assertEqual(input_backend.actions, [("down", "d"), ("up", "d")])

    def test_serpentine_route_alternates_direction(self) -> None:
        steps = SerpentineRoutePlanner(
            steps_per_lane=2, lanes=3, settle=0.0
        ).steps()
        self.assertEqual(
            [(step.key, step.kind) for step in steps],
            [
                ("d", "horizontal"),
                ("d", "horizontal"),
                ("s", "lane_change"),
                ("a", "horizontal"),
                ("a", "horizontal"),
                ("s", "lane_change"),
                ("d", "horizontal"),
                ("d", "horizontal"),
            ],
        )

    def test_recovery_distinguishes_opening_from_analysis(self) -> None:
        policy = RecoveryPolicy()
        opening = policy.decide(
            ScanError("opening", "shop_window_not_detected", retry_count=0)
        )
        analysis = policy.decide(
            ScanError("analyzing", "ollama_timeout", retry_count=1)
        )
        self.assertEqual(opening.action, RecoveryAction.RETRY_INTERACTION)
        self.assertEqual(analysis.action, RecoveryAction.REVIEW)


if __name__ == "__main__":
    unittest.main()
