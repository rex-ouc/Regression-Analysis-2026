import numpy as np

class AnalyticalSolver:
    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)
        return self

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.beta