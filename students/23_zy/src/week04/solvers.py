import numpy as np

class AnalyticalSolver:
    def __init__(self):
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        XTX = X.T @ X
        XTy = X.T @ y
        self.beta = np.linalg.solve(XTX, XTy)
        return self

    def predict(self, X: np.ndarray):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.beta

class GradientDescentSolver:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, P = X.shape
        X = np.hstack([np.ones((N, 1)), X])
        self.beta = np.zeros(P + 1)
        for _ in range(self.epochs):
            y_pred = X @ self.beta
            error = y - y_pred
            grad = -(X.T @ error) / N
            self.beta -= self.learning_rate * grad
        return self

    def predict(self, X: np.ndarray):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.beta