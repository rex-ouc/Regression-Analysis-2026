"""
Module: utils.transformers
Purpose: Custom transformer classes (StandardScaler, etc.)
"""
import numpy as np


class CustomStandardScaler:
    """
    手写标准化转换器，严格遵循大厂 Transformer 接口规范
    - fit(X): 计算均值和标准差
    - transform(X): 使用保存的参数标准化数据
    - fit_transform(X): 合并 fit 和 transform
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted = False
    
    def fit(self, X):
        """
        计算并保存 X 的均值和标准差
        
        Args:
            X: shape (n_samples, n_features)
        
        Returns:
            self
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # 避免除以 0
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        self._fitted = True
        return self
    
    def transform(self, X):
        """
        使用保存的均值和标准差标准化数据
        
        Args:
            X: shape (n_samples, n_features)
        
        Returns:
            X_scaled: 标准化后的数据
        """
        if not self._fitted:
            raise ValueError("必须先调用 fit() 再调用 transform()")
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """
        先 fit 再 transform
        """
        return self.fit(X).transform(X)
    

class SimpleImputer:
    """
    简单的缺失值填补器
    - fit(X): 计算每列的填补值（均值）
    - transform(X): 使用保存的值填补缺失值
    """
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.fill_values_ = None
        self._fitted = False
    
    def fit(self, X):
        n_features = X.shape[1]
        self.fill_values_ = np.zeros(n_features)
        
        for j in range(n_features):
            col = X[:, j]
            # 过滤掉 NaN
            valid_col = col[~np.isnan(col)]
            
            if self.strategy == 'mean':
                self.fill_values_[j] = np.mean(valid_col) if len(valid_col) > 0 else 0
            elif self.strategy == 'median':
                self.fill_values_[j] = np.median(valid_col) if len(valid_col) > 0 else 0
            else:
                self.fill_values_[j] = 0
        
        self._fitted = True
        return self
    
    def transform(self, X):
        if not self._fitted:
            raise ValueError("必须先调用 fit() 再调用 transform()")
        
        X_copy = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X_copy[:, j])
            X_copy[mask, j] = self.fill_values_[j]
        
        return X_copy
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
