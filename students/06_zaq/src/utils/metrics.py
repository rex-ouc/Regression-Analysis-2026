"""
Module: utils.metrics
Purpose: Core evaluation metrics (RMSE, MAE, MAPE)
"""
import numpy as np


def calculate_rmse(y_true, y_pred):
    """
    Root Mean Square Error
    RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error
    MAE = mean(|y_true - y_pred|)
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true, y_pred, eps=1e-8):
    """
    Mean Absolute Percentage Error
    MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    
    Args:
        eps: 避免除以 0 的小常数
    """
    # 避免分母为 0
    y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def calculate_all_metrics(y_true, y_pred):
    """一次性计算所有指标"""
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }
