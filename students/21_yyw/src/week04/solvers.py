"""
模块：solvers.py
作用：实现两种线性回归求解器
- AnalyticalSolver: 正规方程解析解
- GradientDescentSolver: 梯度下降迭代解

不依赖 sklearn、statsmodels，纯 NumPy 实现
"""

import numpy as np


class AnalyticalSolver:
    """
    解析解求解器：使用正规方程 (X^T X)^{-1} X^T y
    
    数值稳定性技巧：
    - 不使用 np.linalg.inv，而是用 np.linalg.solve
    - solve 求解线性方程组，比求逆更稳定
    """
    
    def __init__(self):
        self.beta_ = None  # 存储估计的参数
    
    def fit(self, X, y):
        """
        拟合模型
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵（不含截距列）
        y : np.ndarray, shape (n_samples,)
            目标向量
        
        Returns
        -------
        self
        """
        # 添加截距列（第一列为 1）
        X_with_const = np.column_stack([np.ones(X.shape[0]), X])
        
        # 正规方程：β = (X^T X)^{-1} X^T y
        # 使用 np.linalg.solve 求解 (X^T X) β = X^T y
        XtX = X_with_const.T @ X_with_const
        Xty = X_with_const.T @ y
        
        # 数值稳定的求解方法
        self.beta_ = np.linalg.solve(XtX, Xty)
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            预测值
        """
        if self.beta_ is None:
            raise ValueError("请先调用 fit 方法")
        
        X_with_const = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_const @ self.beta_
    
    def get_params(self):
        """返回估计的参数 [β₀, β₁, ...]"""
        return self.beta_


class GradientDescentSolver:
    """
    梯度下降求解器：批量梯度下降 (Batch Gradient Descent)
    
    梯度公式推导：
    L(β) = (1/2n) * Σ (y_i - X_iβ)^2
    ∇L(β) = -(1/n) * X^T (y - Xβ)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 tolerance=1e-6, verbose=False):
        """
        Parameters
        ----------
        learning_rate : float
            学习率（步长）
        n_iterations : int
            最大迭代次数
        tolerance : float
            收敛阈值（梯度变化小于该值时停止）
        verbose : bool
            是否打印迭代过程
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.beta_ = None
        self.loss_history_ = []  # 记录损失函数历史
    
    def fit(self, X, y):
        """
        拟合模型（批量梯度下降）
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵
        y : np.ndarray, shape (n_samples,)
            目标向量
        
        Returns
        -------
        self
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # 添加截距列
        X_with_const = np.column_stack([np.ones(n_samples), X])
        n_params = n_features + 1
        
        # 初始化参数（全零）
        beta = np.zeros(n_params)
        
        # 梯度下降迭代
        for i in range(self.n_iterations):
            # 计算预测值
            y_pred = X_with_const @ beta
            
            # 计算误差
            error = y_pred - y
            
            # 计算损失 (MSE)
            loss = np.mean(error ** 2) / 2
            self.loss_history_.append(loss)
            
            # 计算梯度：∇L = (1/n) * X^T (Xβ - y)
            gradient = (X_with_const.T @ error) / n_samples
            
            # 更新参数
            beta_new = beta - self.learning_rate * gradient
            
            # 检查收敛
            if np.linalg.norm(beta_new - beta) < self.tolerance:
                if self.verbose:
                    print(f"收敛于第 {i+1} 次迭代")
                beta = beta_new
                break
            
            beta = beta_new
            
            # 打印进度
            if self.verbose and (i + 1) % 100 == 0:
                print(f"迭代 {i+1}/{self.n_iterations}, 损失: {loss:.6f}")
        
        self.beta_ = beta
        return self
    
    def predict(self, X):
        """
        预测
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵
        
        Returns
        -------
        y_pred : np.ndarray
            预测值
        """
        if self.beta_ is None:
            raise ValueError("请先调用 fit 方法")
        
        X_with_const = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_const @ self.beta_
    
    def get_params(self):
        """返回估计的参数"""
        return self.beta_
    
    def get_loss_history(self):
        """返回损失函数历史"""
        return self.loss_history_