import unittest
from pathlib import Path
import sys

import numpy as np

# Support both ``python -m unittest`` and direct execution of this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.kan_ad_detector import KANADDetector


class _FakeModel:
    is_fitted = False

    def fit(self, x_windows, y_next):
        self.is_fitted = True

    def predict(self, x_windows):
        return x_windows[:, -1, :].astype(np.float32)


class KANADDetectorTests(unittest.TestCase):
    def make_detector(self):
        detector = KANADDetector(window_size=3, epochs=1)
        detector._new_model = _FakeModel
        return detector

    def test_offline_alignment_and_streaming(self):
        train = np.zeros((8, 38), dtype=np.float32)
        detector = self.make_detector()
        detector.fit(train)

        test = train.copy()
        test[5] = 10.0
        scores = detector.decision_function(test)
        labels = detector.predict(test)

        self.assertEqual(scores.shape, (8,))
        self.assertTrue(np.all(scores[:3] == 0))
        self.assertEqual(scores.dtype, np.float32)
        self.assertEqual(labels.shape, (8,))
        self.assertEqual(labels[5], 1)

        detector.reset_stream()
        stream_labels = [detector.step_stream(point) for point in test]
        self.assertEqual(stream_labels[:3], [0, 0, 0])
        self.assertEqual(stream_labels[5], 1)

    def test_invalid_shape_and_unfitted_state(self):
        detector = self.make_detector()
        with self.assertRaises(RuntimeError):
            detector.predict(np.zeros((4, 38), dtype=np.float32))
        with self.assertRaises(ValueError):
            detector.fit(np.zeros((4, 37), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
