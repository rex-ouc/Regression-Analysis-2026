"""
Module: utils.metrics
Purpose: Model evaluation metrics (RMSE, MAE, MAPE).
"""
import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    float : RMSE value
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = mean(|y_true - y_pred|)
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    float : MAE value
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = np.mean(np.abs(y_true - y_pred))
    
    return float(mae)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100%
    
    Note: Handles edge cases where y_true ≈ 0 using epsilon.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values (must not contain 0 or very small values)
    y_pred : np.ndarray
        Predicted values
    epsilon : float
        Small value to avoid division by zero
    
    Returns:
    --------
    float : MAPE value (as percentage)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Avoid division by zero: replace very small y_true with epsilon
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    # Calculate percentage errors
    percentage_errors = np.abs((y_true - y_pred) / y_true_safe) * 100
    
    # Handle any remaining inf/nan values
    percentage_errors = np.where(np.isfinite(percentage_errors), percentage_errors, 0)
    
    mape = np.mean(percentage_errors)
    
    return float(mape)
