import numpy as np

class AnalyticalSolver:
    def __init__(self):
        self.beta = None  # 系数向量（不含截距）
    
    def fit(self, X, y):
        """使用正规方程求解解析解：β = (X^T X)^{-1} X^T y"""
        # 添加截距项（X = [1, x1, x2, ..., xp]）
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        # 转换为矩阵（确保数值稳定性）
        X_mat = np.matrix(X_with_intercept)
        y_mat = np.matrix(y.reshape(-1, 1))
        # 求解 (X^T X) β = X^T y（避免直接求逆，用 solve 更稳定）
        XTX = X_mat.T @ X_mat
        XTy = X_mat.T @ y_mat
        self.beta = np.array(np.linalg.solve(XTX, XTy)).flatten()
        return self
    
    def predict(self, X):
        """预测：y_pred = Xβ（含截距）"""
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_with_intercept @ self.beta


class GradientDescentSolver:
    def __init__(self, learning_rate=0.01, epochs=1000, tol=1e-6):
        self.learning_rate = learning_rate  # 学习率
        self.epochs = epochs                # 迭代次数
        self.tol = tol                      # 收敛阈值
        self.beta = None                    # 系数向量（不含截距）
    
    def _compute_gradient(self, X, y, beta):
        """计算全批量梯度：∇L(β) = (2/N) X^T (Xβ - y)"""
        n = X.shape[0]
        X_with_intercept = np.hstack([np.ones((n, 1)), X])
        y_pred = X_with_intercept @ beta
        error = y_pred - y
        gradient = (2 / n) * X_with_intercept.T @ error
        return gradient
    
    def fit(self, X, y):
        """全批量梯度下降迭代求解"""
        n, p = X.shape
        # 初始化系数（含截距，共 p+1 个参数）
        self.beta = np.zeros(p + 1)
        prev_loss = float('inf')
        
        for epoch in range(self.epochs):
            gradient = self._compute_gradient(X, y, self.beta)
            self.beta -= self.learning_rate * gradient
            
            # 计算损失（MSE）
            X_with_intercept = np.hstack([np.ones((n, 1)), X])
            y_pred = X_with_intercept @ self.beta
            loss = np.mean((y_pred - y) ** 2)
            
            # 提前停止（损失变化小于阈值）
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        
        # 分离截距和特征系数
        self.beta = self.beta  # 保留截距，预测时会自动处理
        return self
    
    def predict(self, X):
        """预测：y_pred = Xβ（含截距）"""
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_with_intercept @ self.beta