"""Lekka, opcjonalna weryfikacja celu detektora koloru.

Model nie usuwa kandydatow. Dostarcza tylko wynik, ktory pozwala odsunac na
koniec kolejki cele podobne do postaci/broni. Brak pliku modelu zachowuje
dokladnie dotychczasowa kolejnosc po odleglosci.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_MODEL_PATH = Path(__file__).with_name("shop_target_svm.npz")
DEFAULT_METADATA_PATH = Path(__file__).with_name("shop_target_svm.json")


def crop_target(
    image: Image.Image,
    center: tuple[int, int],
    *,
    size: int = 96,
) -> Image.Image:
    """Wytnij wycinek zgodny z ``analysis.target_dataset``."""

    width, height = image.size
    actual_size = min(size, width, height)
    half = actual_size // 2
    left = max(0, min(center[0] - half, width - actual_size))
    top = max(0, min(center[1] - half, height - actual_size))
    return image.crop((left, top, left + actual_size, top + actual_size))


def target_features(image: Image.Image) -> np.ndarray:
    """HOG ksztaltu + male histogramy koloru; bez ciezkich zaleznosci ML."""

    bgr = cv2.cvtColor(
        np.asarray(image.convert("RGB").resize((64, 64))),
        cv2.COLOR_RGB2BGR,
    )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        (64, 64),
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    ).compute(gray).reshape(-1)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    histograms = []
    for channel, bins, value_range in (
        (0, 18, (0, 180)),
        (1, 8, (0, 256)),
        (2, 8, (0, 256)),
    ):
        histogram = cv2.calcHist(
            [hsv], [channel], None, [bins], value_range
        ).reshape(-1)
        histogram /= max(float(histogram.sum()), 1.0)
        histograms.append(histogram)
    return np.concatenate((hog, *histograms)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class TargetAssessment:
    score: float
    likely_false: bool


class ShopTargetVerifier:
    """Liniowy SVM OpenCV. Dodatni score oznacza cel bardziej sklepowy."""

    def __init__(
        self,
        weights: np.ndarray,
        rho: float,
        *,
        score_sign: float = 1.0,
        defer_threshold: float = 0.0,
        crop_size: int = 96,
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        self.rho = float(rho)
        self.score_sign = score_sign
        self.defer_threshold = defer_threshold
        self.crop_size = crop_size

    @classmethod
    def load(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> ShopTargetVerifier | None:
        model_file = Path(model_path)
        metadata_file = Path(metadata_path)
        if not model_file.exists() or not metadata_file.exists():
            return None
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            artifact = np.load(model_file)
            return cls(
                artifact["weights"],
                float(artifact["rho"]),
                score_sign=float(metadata.get("score_sign", 1.0)),
                defer_threshold=float(metadata.get("defer_threshold", 0.0)),
                crop_size=int(metadata.get("crop_size", 96)),
            )
        except (
            OSError,
            ValueError,
            KeyError,
            cv2.error,
            json.JSONDecodeError,
        ):
            return None

    def assess(
        self,
        image: Image.Image,
        center: tuple[int, int],
    ) -> TargetAssessment:
        crop = crop_target(image, center, size=self.crop_size)
        features = target_features(crop)
        raw = float(features @ self.weights - self.rho)
        score = self.score_sign * raw
        return TargetAssessment(score, score < self.defer_threshold)
