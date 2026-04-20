import numpy as np


class AnalyticalSolver:
    """
    使用正规方程求解线性回归参数
    beta = (X^T X)^(-1) X^T y
    但用 np.linalg.solve 提高数值稳定性
    """

    def fit(self, X, y):
        """
        X: (n, p)
        y: (n,)
        """
        # 加一列1（截距项）
        X = self._add_intercept(X)

        # 正规方程： (X^T X) beta = X^T y
        A = X.T @ X
        b = X.T @ y

        # 用 solve 解线性方程组
        self.beta = np.linalg.solve(A, b)
        return self

    def predict(self, X):
        X = self._add_intercept(X)
        return X @ self.beta

    def _add_intercept(self, X):
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)), X])


# =====================================================

class GradientDescentSolver:
    """
    批量梯度下降求解线性回归
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = self._add_intercept(X)

        n, p = X.shape

        # 初始化参数
        self.beta = np.zeros(p)

        for _ in range(self.epochs):
            y_pred = X @ self.beta

            # 梯度： (1/n) X^T (Xβ - y)
            grad = (1 / n) * X.T @ (y_pred - y)

            # 更新
            self.beta -= self.lr * grad

        return self

    def predict(self, X):
        X = self._add_intercept(X)
        return X @ self.beta

    def _add_intercept(self, X):
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)), X])