import numpy as np


class CustomStandardScaler:
    """
    Custom standard scaler.

    fit(X):
        Learn the mean and standard deviation from X.

    transform(X):
        Fill missing values using the learned mean and standardize X.

    fit_transform(X):
        Fit first, then transform.
    """

    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)

        # Handle columns that are all NaN.
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)

        # Avoid division by zero or NaN standard deviation.
        self.std_ = np.where((self.std_ == 0) | np.isnan(self.std_), 1.0, self.std_)

        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("CustomStandardScaler must be fitted before transform().")

        X = np.asarray(X, dtype=float).copy()

        # Fill NaN using the mean learned in fit().
        nan_rows, nan_cols = np.where(np.isnan(X))

        if len(nan_rows) > 0:
            X[nan_rows, nan_cols] = self.mean_[nan_cols]

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)