"""Validate KANADDetector with the provided SMD machine-1-1 data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Support direct execution: ``python tests/validate_smd.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.kan_ad_detector import KANADDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end KAN-AD validation on SMD machine-1-1."
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--n-harmonics", type=int, default=5)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--skip-stream",
        action="store_true",
        help="Skip the full point-by-point streaming consistency check.",
    )
    return parser.parse_args()


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = PROJECT_ROOT / "data"
    train = np.loadtxt(data_dir / "train" / "machine-1-1.txt", delimiter=",").astype(
        np.float32
    )
    test = np.loadtxt(data_dir / "test" / "machine-1-1.txt", delimiter=",").astype(
        np.float32
    )
    labels = np.loadtxt(
        data_dir / "test_label" / "machine-1-1.txt", delimiter=","
    ).astype(np.int32)
    return train, test, labels


def main() -> int:
    args = parse_args()
    train, test, labels = load_data()

    if train.ndim != 2 or train.shape[1] != 38:
        raise AssertionError(f"unexpected train shape: {train.shape}")
    if test.ndim != 2 or test.shape[1] != 38:
        raise AssertionError(f"unexpected test shape: {test.shape}")
    if labels.shape != (test.shape[0],):
        raise AssertionError(
            f"label shape {labels.shape} does not match test length {test.shape[0]}"
        )
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise AssertionError("train/test data contains non-finite values")

    detector = KANADDetector(
        window_size=args.window_size,
        n_harmonics=args.n_harmonics,
        percentile=args.percentile,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    detector.fit(train)

    scores = detector.decision_function(test)
    predictions = detector.predict(test)
    if scores.shape != (test.shape[0],):
        raise AssertionError(f"unexpected score shape: {scores.shape}")
    if predictions.shape != (test.shape[0],):
        raise AssertionError(f"unexpected prediction shape: {predictions.shape}")
    if not np.isfinite(scores).all():
        raise AssertionError("detector returned non-finite scores")

    stream_matches = "skipped"
    if not args.skip_stream:
        detector.reset_stream()
        stream_predictions = np.asarray(
            [detector.step_stream(point) for point in test], dtype=np.int32
        )
        stream_matches = str(np.array_equal(stream_predictions, predictions))
        if stream_matches != "True":
            raise AssertionError("stream predictions do not match offline predictions")

    print(f"train_shape={train.shape}")
    print(f"test_shape={test.shape}")
    print(f"threshold={detector.threshold_}")
    print(f"scores_finite={np.isfinite(scores).all()}")
    print(f"stream_matches_offline={stream_matches}")
    print(f"precision={precision_score(labels, predictions, zero_division=0):.6f}")
    print(f"recall={recall_score(labels, predictions, zero_division=0):.6f}")
    print(f"f1={f1_score(labels, predictions, zero_division=0):.6f}")
    print(f"auroc={roc_auc_score(labels, scores):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
