"""Stable public wrapper around the repository's KAN-AD model.

The upstream implementation is integrated with EasyTSAD and exposes a
framework-specific training/evaluation lifecycle.  This module adapts its
KANADModel regressor to a small, array-based API for 38-dimensional SMD data.
"""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


# ``KAN-AD`` contains a hyphen, so it cannot be imported as a normal Python
# package from ``models``.  Add the checked-out upstream project to the import
# path without requiring callers to modify PYTHONPATH.
_UPSTREAM_ROOT = Path(__file__).resolve().parent / "KAN-AD"
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised by installation errors
    raise ImportError(
        "KAN-AD dependencies are unavailable. Activate the project venv and "
        "install the dependencies listed in 封装说明.md."
    ) from exc


def _load_upstream_kanad_model() -> type[torch.nn.Module]:
    """Load only the upstream network class, without EasyTSAD's CLI imports.

    The checked-out ``kanad.py`` also imports EasyTSAD's experiment stack at
    module import time.  The published EasyTSAD wheel currently contains a
    syntax error in one of those optional modules, while the neural network
    class itself only needs torch.  Extracting the unchanged network class
    keeps this wrapper independent of that unrelated experiment stack.
    """
    source_path = _UPSTREAM_ROOT / "kanad" / "kanad.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    model_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KANADModel"
    )
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="torch", asname="th")]),
            ast.ImportFrom(
                module="torch",
                names=[ast.alias(name="nn", asname=None)],
                level=0,
            ),
            model_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["KANADModel"]


try:
    KANADModel = _load_upstream_kanad_model()
except (OSError, SyntaxError, KeyError) as exc:  # pragma: no cover
    raise ImportError("Unable to load the KAN-AD network from models/KAN-AD") from exc


class _KANADRegressor:
    """Train/predict adapter for the upstream single-series KANADModel.

    The upstream network predicts one scalar from one window.  A shared model
    is trained over all 38 channels, treating each channel/window pair as one
    regression sample.  Predictions are reshaped back to ``(n_windows, 38)``
    for the public detector.
    """

    def __init__(
        self,
        window_size: int,
        n_harmonics: int,
        device: torch.device,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        self.window_size = window_size
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.network = KANADModel(window=window_size, order=n_harmonics).to(device)
        self.is_fitted = False

    @staticmethod
    def _as_output_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        return output.reshape(-1)

    def fit(self, x_windows: np.ndarray, y_next: np.ndarray) -> None:
        n_windows, _, n_features = x_windows.shape
        x = torch.from_numpy(
            x_windows.transpose(0, 2, 1).reshape(n_windows * n_features, self.window_size)
        )
        y = torch.from_numpy(y_next.reshape(n_windows * n_features))
        dataset = TensorDataset(x, y)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            drop_last=len(dataset) > 1,
        )

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()

        self.network.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self._as_output_tensor(self.network(batch_x))
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()

        self.network.eval()
        self.is_fitted = True

    def predict(self, x_windows: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("KAN-AD model has not been fitted")

        n_windows, _, n_features = x_windows.shape
        x = torch.from_numpy(
            x_windows.transpose(0, 2, 1).reshape(n_windows * n_features, self.window_size)
        )
        outputs: list[np.ndarray] = []
        loader = DataLoader(x, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_x in loader:
                prediction = self._as_output_tensor(
                    self.network(batch_x.to(self.device))
                )
                outputs.append(prediction.cpu().numpy())

        return np.concatenate(outputs).reshape(n_windows, n_features).astype(np.float32)


class KANADDetector:
    """Unified offline and streaming anomaly detector for 38-D time series.

    Parameters
    ----------
    window_size:
        Number of historical points used to predict the next point.
    n_harmonics:
        Fourier/KAN order passed to the upstream KANAD network.
    percentile:
        Training-score percentile used as the fixed anomaly threshold.
    device:
        ``"cpu"`` or ``"cuda"``.  CUDA falls back to CPU when unavailable.
    epochs, batch_size, learning_rate:
        Optional training controls.  They keep the required public defaults
        intact while making small local tests and deployment tuning practical.
    """

    N_FEATURES = 38

    def __init__(
        self,
        window_size: int = 10,
        n_harmonics: int = 5,
        percentile: float = 99.0,
        device: str = "cpu",
        *,
        epochs: int = 10,
        batch_size: int = 1024,
        learning_rate: float = 0.01,
    ) -> None:
        if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(n_harmonics, (int, np.integer)) or n_harmonics <= 0:
            raise ValueError("n_harmonics must be a positive integer")
        if not 0.0 <= float(percentile) <= 100.0:
            raise ValueError("percentile must be between 0 and 100")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.window_size = int(window_size)
        self.n_harmonics = int(n_harmonics)
        self.percentile = float(percentile)
        self.requested_device = device
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._torch_device = torch.device(self.device)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)

        self.model = self._new_model()
        self.scaler = StandardScaler()
        self.threshold_: float | None = None
        self.buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def _new_model(self) -> _KANADRegressor:
        return _KANADRegressor(
            window_size=self.window_size,
            n_harmonics=self.n_harmonics,
            device=self._torch_device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

    @classmethod
    def _validate_matrix(cls, X: np.ndarray, name: str) -> np.ndarray:
        try:
            array = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric numpy array") from exc
        if array.ndim != 2 or array.shape[1] != cls.N_FEATURES:
            raise ValueError(f"{name} must have shape (n, {cls.N_FEATURES})")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _make_windows(X: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
        n_windows = X.shape[0] - window_size
        if n_windows <= 0:
            raise ValueError(
                "at least window_size + 1 observations are required to construct a window"
            )
        windows = np.stack(
            [X[index : index + window_size] for index in range(n_windows)], axis=0
        )
        targets = X[window_size:]
        return windows.astype(np.float32), targets.astype(np.float32)

    def _require_fitted(self) -> None:
        if self.threshold_ is None or not self.model.is_fitted:
            raise RuntimeError("KANADDetector.fit must be called before inference")

    def fit(self, X_train: np.ndarray) -> None:
        """Fit scaling, the KAN-AD predictor, and the training threshold."""
        X_train = self._validate_matrix(X_train, "X_train")
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train).astype(np.float32)
        X_windows, y_next = self._make_windows(X_scaled, self.window_size)

        self.model = self._new_model()
        self.model.fit(X_windows, y_next)
        y_pred = self.model.predict(X_windows)
        scores = np.mean(np.square(y_pred - y_next), axis=1, dtype=np.float32)
        self.threshold_ = float(np.percentile(scores, self.percentile))
        self.reset_stream()

    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        """Return one MSE anomaly score per input point."""
        self._require_fitted()
        X_test = self._validate_matrix(X_test, "X_test")
        scores = np.zeros(X_test.shape[0], dtype=np.float32)
        if X_test.shape[0] <= self.window_size:
            return scores

        X_scaled = self.scaler.transform(X_test).astype(np.float32)
        X_windows, y_next = self._make_windows(X_scaled, self.window_size)
        y_pred = self.model.predict(X_windows)
        scores[self.window_size :] = np.mean(
            np.square(y_pred - y_next), axis=1, dtype=np.float32
        )
        return scores

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return integer labels: 0 for normal and 1 for anomalous."""
        scores = self.decision_function(X_test)
        return (scores > self.threshold_).astype(int)

    def reset_stream(self) -> None:
        """Clear the historical observations used by :meth:`step_stream`."""
        self.buffer.clear()

    def step_stream(self, point: np.ndarray) -> int:
        """Score one new observation using only the preceding history."""
        self._require_fitted()
        try:
            point_array = np.asarray(point, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("point must be a numeric numpy array with shape (38,)") from exc
        if point_array.shape != (self.N_FEATURES,):
            raise ValueError(f"point must have shape ({self.N_FEATURES},)")
        if not np.isfinite(point_array).all():
            raise ValueError("point must contain only finite values")

        label = 0
        if len(self.buffer) == self.window_size:
            history = np.stack(tuple(self.buffer), axis=0).astype(np.float32)
            scaled_history = self.scaler.transform(history).astype(np.float32)
            scaled_point = self.scaler.transform(point_array[None, :])[0].astype(np.float32)
            predicted = self.model.predict(scaled_history[None, :, :])[0]
            score = float(np.mean(np.square(predicted - scaled_point)))
            label = int(score > self.threshold_)

        self.buffer.append(point_array.copy())
        return label


__all__ = ["KANADDetector"]
