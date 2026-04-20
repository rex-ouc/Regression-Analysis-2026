import numpy as np

class AnalyticalSolver:
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # 添加常数项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # 正规方程: (X^T * X) * beta = X^T * y
        XTX = X_b.T.dot(X_b)
        XTy = X_b.T.dot(y)
        # 使用老师推荐的稳定解法
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
        y_vec = y.reshape(-1, 1)
        
        # 初始化
        self.beta = np.zeros((n_features + 1, 1))
        
        for _ in range(self.epochs):
            # 梯度计算公式: (2/n) * X^T * (X * beta - y)
            error = X_b.dot(self.beta) - y_vec
            gradient = (2 / n_samples) * X_b.T.dot(error)
            self.beta -= self.lr * gradient
            
        self.beta = self.beta.flatten()

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.beta)