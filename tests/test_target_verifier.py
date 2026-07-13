from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scanner.config import DetectorSettings
from scanner.detection import ShopDetector
from scanner.detection.target_verifier import (
    ShopTargetVerifier,
    TargetAssessment,
    crop_target,
    target_features,
)
from scanner.detection.train_target_verifier import session_from_name


class _Verifier:
    def assess(self, image, center):
        # Blizszy kandydat jest podejrzany; dalszy ma isc pierwszy.
        likely_false = center[0] < 120
        return TargetAssessment(-1.0 if likely_false else 1.0, likely_false)


class TargetVerifierTests(unittest.TestCase):
    def test_crop_clamps_without_black_padding(self) -> None:
        image = Image.new("RGB", (120, 100), (12, 34, 56))
        crop = crop_target(image, (2, 2), size=96)

        self.assertEqual(crop.size, (96, 96))
        self.assertEqual(crop.getpixel((0, 0)), (12, 34, 56))

    def test_features_are_stable_one_dimensional_vector(self) -> None:
        image = Image.new("RGB", (96, 96), (140, 90, 40))
        first = target_features(image)
        second = target_features(image)

        self.assertEqual(first.ndim, 1)
        self.assertGreater(first.size, 100)
        np.testing.assert_array_equal(first, second)

    def test_npz_model_loads_and_scores(self) -> None:
        feature_count = target_features(Image.new("RGB", (96, 96))).size
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.npz"
            metadata = root / "model.json"
            np.savez_compressed(
                model,
                weights=np.zeros(feature_count, dtype=np.float32),
                rho=np.asarray(0.5, dtype=np.float32),
            )
            metadata.write_text(
                json.dumps(
                    {
                        "score_sign": 1.0,
                        "defer_threshold": 0.0,
                        "crop_size": 96,
                    }
                ),
                encoding="utf-8",
            )

            verifier = ShopTargetVerifier.load(model, metadata)
            self.assertIsNotNone(verifier)
            assessment = verifier.assess(
                Image.new("RGB", (120, 120)), (60, 60)
            )

        self.assertAlmostEqual(assessment.score, -0.5)
        self.assertTrue(assessment.likely_false)

    def test_hybrid_only_reorders_legacy_candidate_set(self) -> None:
        array = np.zeros((200, 300, 3), dtype=np.uint8)
        array[95:105, 95:107] = (150, 90, 40)
        array[95:105, 145:157] = (150, 90, 40)
        settings = DetectorSettings(
            area_min=20,
            area_max=300,
            width_min=5,
            width_max=30,
            height_min=4,
            height_max=30,
            min_radius=0,
            max_radius=200,
            max_results=2,
            hybrid_enabled=False,
        )
        detector = ShopDetector(settings, verifier=_Verifier())

        candidates = detector.detect(Image.fromarray(array))

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].local_position[0], 150)
        self.assertEqual(candidates[1].local_position[0], 100)
        self.assertFalse(candidates[0].likely_false)
        self.assertTrue(candidates[1].likely_false)

    def test_hybrid_does_not_pull_extra_candidate_into_max_results(self) -> None:
        array = np.zeros((220, 400, 3), dtype=np.uint8)
        for x in (155, 195, 300):
            array[105:115, x : x + 12] = (150, 90, 40)
        settings = DetectorSettings(
            area_min=20,
            area_max=300,
            width_min=5,
            width_max=30,
            height_min=4,
            height_max=30,
            min_radius=0,
            max_radius=300,
            max_results=2,
            hybrid_enabled=False,
        )
        detector = ShopDetector(settings, verifier=_Verifier())

        candidates = detector.detect(Image.fromarray(array))

        self.assertEqual(len(candidates), 2)
        self.assertNotIn(305, [candidate.local_position[0] for candidate in candidates])

    def test_session_id_is_taken_from_dataset_filename(self) -> None:
        path = Path("20260620_181623_004_shop-00007.png")
        self.assertEqual(session_from_name(path), "20260620_181623")


if __name__ == "__main__":
    unittest.main()
