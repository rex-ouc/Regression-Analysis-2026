import numpy as np

class AnalyticalSolver:
    """使用正规方程求解：(X^T X) \beta = X^T Y"""
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # 添加偏置项 (Intercept)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # 计算 A = X^T X 和 b = X^T y
        A = X_b.T @ X_b
        b = X_b.T @ y
        
        # 使用 np.linalg.solve 替代 inv，计算更稳定
        self.beta = np.linalg.solve(A, b)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta

class GradientDescentSolver:
    """全批量梯度下降求解器"""
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.beta = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.c_[np.ones((n_samples, 1)), X]
        # 随机初始化权重
        self.beta = np.zeros((n_features + 1, 1))
        
        for _ in range(self.epochs):
            # 预测值
            y_pred = X_b @ self.beta
            # 计算梯度: (1/N) * X^T * (X*beta - y)
            gradient = (1 / n_samples) * (X_b.T @ (y_pred - y))
            # 更新参数
            self.beta -= self.lr * gradient
            
            # 记录 MSE 损失
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta