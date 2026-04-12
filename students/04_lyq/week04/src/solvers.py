import numpy as np


class AnalyticalSolver:
    """
    使用正规方程的解析解
    beta = (X^T X)^(-1) X^T 
    """

    def fit(self, X, y):
        # 添加 bias 项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        A = X_b.T @ X_b
        b = X_b.T @ y

        self.beta = np.linalg.solve(A, b)
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta


class GradientDescentSolver:
    """
    批量梯度下降
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        N, P = X.shape
        X_b = np.c_[np.ones((N, 1)), X]

        self.beta = np.zeros(P + 1)

        for _ in range(self.epochs):
            y_pred = X_b @ self.beta
            error = y_pred - y

            # 梯度公式
            grad = (2 / N) * (X_b.T @ error)

            self.beta -= self.lr * grad

        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta