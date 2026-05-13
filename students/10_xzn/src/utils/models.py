import numpy as np


class CustomOLS:
    """
    自定义普通最小二乘线性回归模型。

    Parameters
    ----------
    fit_intercept : bool, default=True
        是否在拟合时添加截距项（常数项列）。
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        使用正规方程拟合模型参数。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵，形状为 (n_samples, n_features)，不包含常数项列。
        y : np.ndarray
            目标变量，形状为 (n_samples,)。
        """
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        XTX = X.T @ X
        XTy = X.T @ y
        self.beta = np.linalg.solve(XTX, XTy)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合好的模型进行预测。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵，形状为 (n_samples, n_features)，不包含常数项列。

        Returns
        -------
        np.ndarray
            预测值，形状为 (n_samples,)。
        """
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), X]

        return X @ self.beta