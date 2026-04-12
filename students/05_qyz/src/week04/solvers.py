"""
Module: solvers
Purpose: Custom implementation of OLS estimators using OOP principles.
CS Concept: We mimic the `sklearn` API design (.fit() and .predict() methods).
"""

import numpy as np  # 科学计算库，用于矩阵运算
import time  # 高精度计时器，用于测量运行时间

class AnalyticalSolver:  # 正规方程求解析解
    """Solver using the exact Normal Equation."""

    def __init__(self):
        self.coef_ = None  # To store the estimated betas
        self.fit_time_ = 0.0  # To track computation time

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model using X^T X beta = X^T y.

        CS Tip: DO NOT use `np.linalg.inv(X.T @ X) @ X.T @ y`.
        Matrix inversion is numerically unstable and slow.
        Instead, solve the linear system Ax = b using `np.linalg.solve()`.
        """
        start_time = time.perf_counter()

        # 1. Calculate A = X.T @ X
        A = X.T @ X
        # 2. Calculate b = X.T @ y
        b = X.T @ y
        # 3. Solve for beta using np.linalg.solve(A, b)
        self.coef_ = np.linalg.solve(A, b)
        # 4. Store result in self.coef_

        self.fit_time_ = time.perf_counter() - start_time

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Return X @ self.coef_
        return X @ self.coef_

class GradientDescentSolver:  # 使用全批量梯度下降法迭代求解
    """Solver using numerical optimization (Batch Gradient Descent)."""

    def __init__(
        self, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.fit_time_ = 0.0
        self.loss_history_ = []  # Useful for plotting convergence

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model by iteratively moving against the gradient.
        """
        start_time = time.perf_counter()
        n_samples, n_features = X.shape

        # 1. Initialize self.coef_ with zeros
        self.coef_ = np.zeros(n_features)

        for epoch in range(self.max_iter):
            # a. Compute predictions: y_pred = X @ self.coef_
            y_pred = X @ self.coef_
            # b. Compute the gradient vector: grad = (2 / n_samples) * X.T @ (y_pred - y)
            grad = (2 / n_samples) * X.T @ (y_pred - y)
            # c. Update weights: self.coef_ -= self.lr * grad
            self.coef_ -= self.lr * grad
            # d. (Optional) Check convergence: if magnitude of grad < self.tol, break early
            if np.linalg.norm(grad) < self.tol:
                break
            # e. Log the MSE loss to self.loss_history_
            mse = np.mean((y_pred - y) ** 2)
            self.loss_history_.append(mse)

        self.fit_time_ = time.perf_counter() - start_time

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Return X @ self.coef_
        return X @ self.coef_
