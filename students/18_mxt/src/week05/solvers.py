import numpy as np


class AnalyticalSolver:
    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)
        return self

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.beta


class GradientDescentSolver:
    def __init__(self, learning_rate=0.1, max_iter=50000, tol=1e-6):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        n, p = X.shape
        self.beta = np.zeros(p)

        for _ in range(self.max_iter):
            y_pred = X @ self.beta
            grad = (1 / n) * X.T @ (y_pred - y)
            new_beta = self.beta - self.lr * grad

            if np.linalg.norm(new_beta - self.beta) < self.tol:
                break
            self.beta = new_beta
        return self

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.beta
