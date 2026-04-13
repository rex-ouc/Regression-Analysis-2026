import numpy as np

class AnalyticalSolver:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        XTX = X_b.T @ X_b
        XTy = X_b.T @ y
        self.beta_ = np.linalg.solve(XTX, XTy)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta_

class GradientDescentSolver:
    def __init__(self, learning_rate: float = 1e-3, epochs: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        N, P = X.shape
        X_b = np.c_[np.ones((N, 1)), X]
        self.beta_ = np.zeros(P + 1)
        
        for _ in range(self.epochs):
            y_pred = X_b @ self.beta_
            grad = (1 / N) * X_b.T @ (y_pred - y)
            self.beta_ -= self.lr * grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta_