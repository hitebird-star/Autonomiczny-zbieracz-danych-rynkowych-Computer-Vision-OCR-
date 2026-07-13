"""Trening lekkiego rankera real/false na zbiorze wygenerowanym przez C2.

Uruchomienie:
    python -m scanner.detection.train_target_verifier
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .target_verifier import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    target_features,
)


@dataclass(frozen=True, slots=True)
class TrainingSample:
    path: Path
    session: str
    label: int
    features: np.ndarray


def session_from_name(path: Path) -> str:
    prefix = path.name.split("_shop-", 1)[0]
    return prefix.rsplit("_", 1)[0]


def load_samples(root: str | Path) -> list[TrainingSample]:
    samples = []
    for directory, label in (("real", 1), ("false", -1)):
        for path in sorted((Path(root) / directory).glob("*.png")):
            with Image.open(path) as image:
                features = target_features(image.convert("RGB"))
            samples.append(
                TrainingSample(path, session_from_name(path), label, features)
            )
    return samples


def train_svm(samples: list[TrainingSample]):
    if not samples or not {sample.label for sample in samples} == {-1, 1}:
        raise ValueError("dataset musi zawierac klasy real i false")
    features = np.stack([sample.features for sample in samples])
    labels = np.asarray([sample.label for sample in samples], dtype=np.int32)
    positives = int((labels == 1).sum())
    negatives = int((labels == -1).sum())

    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.setC(0.5)
    # OpenCV mapuje wiersze wag do posortowanych etykiet: -1, +1.
    # Klasa false (-1) jest rzadka, wiec to ona dostaje wage ~8.4x.
    model.setClassWeights(
        np.asarray(
            [[positives / max(1, negatives)], [1.0]],
            dtype=np.float32,
        )
    )
    model.train(features, cv2.ml.ROW_SAMPLE, labels)

    _, raw = model.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    raw = raw.reshape(-1)
    score_sign = (
        1.0
        if float(raw[labels == 1].mean()) > float(raw[labels == -1].mean())
        else -1.0
    )
    return model, score_sign


def cross_validate(samples: list[TrainingSample], threshold: float) -> dict:
    """Leave-one-session-out; metryka odporna na duplikaty z jednego runu."""

    results: list[tuple[float, int]] = []
    sessions = sorted({sample.session for sample in samples})
    for held_out in sessions:
        test = [sample for sample in samples if sample.session == held_out]
        if not any(sample.label == -1 for sample in test):
            continue
        training = [
            sample for sample in samples if sample.session != held_out
        ]
        if {sample.label for sample in training} != {-1, 1}:
            continue
        model, sign = train_svm(training)
        matrix = np.stack([sample.features for sample in test])
        _, raw = model.predict(matrix, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        results.extend(
            (sign * float(value), sample.label)
            for value, sample in zip(raw.reshape(-1), test)
        )

    real = [score for score, label in results if label == 1]
    false = [score for score, label in results if label == -1]
    pairs = max(1, len(real) * len(false))
    auc = (
        sum(left > right for left in real for right in false)
        + 0.5 * sum(left == right for left in real for right in false)
    ) / pairs
    real_kept = sum(score >= threshold for score in real)
    false_deferred = sum(score < threshold for score in false)
    return {
        "sessions": len(sessions),
        "evaluated_real": len(real),
        "evaluated_false": len(false),
        "auc": round(auc, 4),
        "real_kept_rate": round(real_kept / max(1, len(real)), 4),
        "false_deferred_rate": round(
            false_deferred / max(1, len(false)), 4
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "--data",
        dest="dataset",
        default="dataset/targets",
        help="katalog real/ i false/ z harvest_targets (domyślnie dataset/targets)",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args(argv)

    samples = load_samples(args.dataset)
    metrics = cross_validate(samples, args.threshold)
    model, score_sign = train_svm(samples)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    support_vectors = model.getSupportVectors()
    rho, alpha, _ = model.getDecisionFunction(0)
    weights = alpha.reshape(-1) @ support_vectors
    np.savez_compressed(
        args.model,
        weights=weights.astype(np.float32),
        rho=np.asarray(rho, dtype=np.float32),
    )
    metadata = {
        "version": 1,
        "kind": "opencv_linear_svm_hog_hsv",
        "crop_size": 96,
        "defer_threshold": args.threshold,
        "score_sign": score_sign,
        "training_samples": len(samples),
        "real_samples": sum(sample.label == 1 for sample in samples),
        "false_samples": sum(sample.label == -1 for sample in samples),
        "validation": metrics,
    }
    Path(args.metadata).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
