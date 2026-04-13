"""
自定义求解器模块
实现解析解和梯度下降两种方法
"""

import numpy as np
import time


class AnalyticalSolver:
    """使用正规方程求解析解（数值稳定版本）"""

    def fit(self, X, y):
        """
        拟合模型: beta = (X^T X)^{-1} X^T y
        使用 np.linalg.solve 避免显式求逆

        参数:
            X (np.ndarray): 特征矩阵 (N, P)
            y (np.ndarray): 目标向量 (N,)

        返回:
            beta (np.ndarray): 系数估计值 (P,)
        """
        # 添加截距项（第一列全1）
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        # 使用最小二乘法求解（数值稳定）
        # 求解 (X^T X) beta = X^T y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        beta = np.linalg.solve(XtX, Xty)

        return beta


class GradientDescentSolver:
    """全批量梯度下降求解器"""

    def __init__(self, learning_rate=0.01, n_epochs=1000, tol=1e-6):
        """
        参数:
            learning_rate (float): 学习率
            n_epochs (int): 最大迭代次数
            tol (float): 梯度范数阈值，若小于则提前收敛
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.tol = tol

    def fit(self, X, y, verbose=False):
        """
        拟合模型: 使用梯度下降更新系数

        参数:
            X (np.ndarray): 特征矩阵 (N, P)
            y (np.ndarray): 目标向量 (N,)
            verbose (bool): 是否打印收敛信息

        返回:
            beta (np.ndarray): 系数估计值 (P+1,)，包含截距项
        """
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        n_samples, n_features = X_with_intercept.shape

        # 初始化系数（全零）
        beta = np.zeros(n_features)

        # 梯度下降迭代
        for epoch in range(self.n_epochs):
            # 计算预测值
            y_pred = X_with_intercept @ beta
            # 计算梯度: (2/N) * X^T (y_pred - y)
            gradient = (2.0 / n_samples) * X_with_intercept.T @ (y_pred - y)

            # 更新系数
            beta -= self.learning_rate * gradient

            # 检查收敛
            if np.linalg.norm(gradient) < self.tol:
                if verbose:
                    print(f"梯度下降收敛于第 {epoch + 1} 次迭代")
                break

        return beta
