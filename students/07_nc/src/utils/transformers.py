"""Hand-written transformer classes for Week 10."""
from __future__ import annotations

import numpy as np


class CustomStandardScaler:
    """A minimal Transformer-style standard scaler.

    - fit(X) learns only the column means and standard deviations.
    - transform(X) reuses the saved parameters without looking at new data.
    - fit_transform(X) is only for training data or intentionally global preprocessing.
    """

    def __init__(self, epsilon: float = 1e-12) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "CustomStandardScaler":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1-D or 2-D numeric array")

        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("CustomStandardScaler must be fitted before transform")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1-D or 2-D numeric array")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("X has a different number of features from the fitted data")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
