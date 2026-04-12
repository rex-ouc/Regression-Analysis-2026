import numpy as np


class AnalyticalSolver:
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # 为 X 添加偏置项（常数列）
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # 使用正规方程解析解: (X^T * X)^-1 * X^T * y
        # 老师要求使用 np.linalg.solve 以保证数值稳定性
        XTX = X_b.T.dot(X_b)
        XTy = X_b.T.dot(y)
        self.beta = np.linalg.solve(XTX, XTy)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.beta)


class GradientDescentSolver:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.beta = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.c_[np.ones((n_samples, 1)), X]
        y = y.reshape(-1, 1)

        # 初始化权重
        self.beta = np.zeros((n_features + 1, 1))

        # 批量梯度下降迭代
        for _ in range(self.epochs):
            # 计算梯度: (2/n) * X^T * (X * beta - y)
            prediction = X_b.dot(self.beta)
            errors = prediction - y
            gradient = (2 / n_samples) * X_b.T.dot(errors)
            # 更新参数
            self.beta = self.beta - self.lr * gradient

        self.beta = self.beta.flatten()

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.beta)
