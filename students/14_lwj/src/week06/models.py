import numpy as np
from scipy.stats import f

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.beta_hat_ = None
        self.n_ = 0
        self.p_ = 0
        self.X_b_ = None
        self.y_ = None

    def fit(self, X, y):
        """拟合模型"""
        self.n_, self.p_ = X.shape
        self.y_ = y.copy()
        # 添加截距列
        self.X_b_ = np.c_[np.ones((self.n_, 1)), X]
        
        # 使用lstsq解决奇异矩阵问题
        self.beta_hat_ = np.linalg.lstsq(self.X_b_, y, rcond=None)[0]
        
        self.intercept_ = self.beta_hat_[0]
        self.coef_ = self.beta_hat_[1:]
        return self

    def predict(self, X):
        """预测"""
        if self.beta_hat_ is None:
            raise RuntimeError("请先调用fit方法")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta_hat_

    def score(self, X, y):
        """计算R²"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0
        return 1 - (ss_res / ss_tot)

    def f_test(self, C, d):
        """
        F检验：H0: C @ beta = d
        """
        if self.beta_hat_ is None:
            raise RuntimeError("请先调用fit方法")
        
        # 无约束残差平方和
        y_pred_unres = self.X_b_ @ self.beta_hat_
        rss_unres = np.sum((self.y_ - y_pred_unres) ** 2)
        
        # 求解受限模型
        XTX = self.X_b_.T @ self.X_b_
        XTX_inv = np.linalg.pinv(XTX)  # 使用伪逆
        
        try:
            
            temp = C @ XTX_inv @ C.T
            temp_inv = np.linalg.inv(temp)
            lambda_ = temp_inv @ (C @ self.beta_hat_ - d)
            beta_res = self.beta_hat_ - XTX_inv @ C.T @ lambda_
            
            # 受限残差平方和
            y_pred_res = self.X_b_ @ beta_res
            rss_res = np.sum((self.y_ - y_pred_res) ** 2)
            
            # F统计量
            q = C.shape[0]
            df_res = self.n_ - (self.p_ + 1)
            F_stat = ((rss_res - rss_unres) / q) / (rss_unres / df_res)
            p_value = 1 - f.cdf(F_stat, q, df_res)
            
            return F_stat, p_value
        except np.linalg.LinAlgError:
            return np.nan, np.nan