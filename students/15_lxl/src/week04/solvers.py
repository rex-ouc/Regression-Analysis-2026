"""
Module: solvers
Purpose: Custom implementation of OLS estimators using OOP principles.
CS Concept: We mimic the `sklearn` API design (.fit() and .predict() methods).
"""

import numpy as np
import time


class AnalyticalSolver:
    """Solver using the exact Normal Equation."""

    def __init__(self):
        self.coef_ = None  # To store the estimated betas
        self.fit_time_ = 0.0  # To track computation time

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model using X^T X beta = X^T y.

        CS Tip: DO NOT use `np.linalg.inv(X.T @ X) @ X.T @ y`.
        Matrix inversion is numerically unstable and slow.
        Instead, solve the linear system Ax = b using `np.linalg.solve()`.
        """
        start_time = time.perf_counter()

        # Add intercept column to X
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate A = X.T @ X
        A = X_with_intercept.T @ X_with_intercept
        # Calculate b = X.T @ y
        b = X_with_intercept.T @ y
        # Solve for beta using np.linalg.solve(A, b)
        self.coef_ = np.linalg.solve(A, b)

        self.fit_time_ = time.perf_counter() - start_time

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Add intercept column to X for prediction
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        # Return X @ self.coef_
        return X_with_intercept @ self.coef_


class GradientDescentSolver:
    """Solver using numerical optimization (Batch Gradient Descent)."""

    def __init__(
        self, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.fit_time_ = 0.0
        self.loss_history_ = []  # Useful for plotting convergence

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model by iteratively moving against the gradient.
        """
        start_time = time.perf_counter()
        n_samples, n_features = X.shape

        # Add intercept column to X
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        # Initialize self.coef_ with zeros
        self.coef_ = np.zeros(n_features + 1)  # +1 for intercept

        for epoch in range(self.max_iter):
            # Compute predictions: y_pred = X @ self.coef_
            y_pred = X_with_intercept @ self.coef_
            # Compute the gradient vector: grad = (2 / n_samples) * X.T @ (y_pred - y)
            grad = (2 / n_samples) * X_with_intercept.T @ (y_pred - y)
            # Update weights: self.coef_ -= self.lr * grad
            self.coef_ -= self.lr * grad
            # (Optional) Check convergence: if magnitude of grad < self.tol, break early
            if np.linalg.norm(grad) < self.tol:
                break
            # Log the MSE loss to self.loss_history_
            mse = np.mean((y_pred - y) ** 2)
            self.loss_history_.append(mse)

        self.fit_time_ = time.perf_counter() - start_time

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Add intercept column to X for prediction
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        # Return X @ self.coef_
        return X_with_intercept @ self.coef_