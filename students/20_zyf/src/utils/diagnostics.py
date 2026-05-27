"""
Module: utils.diagnostics
Purpose: Statistical diagnostic tools for model quality assessment.
"""
import numpy as np
import sys
import os

# Handle imports for both direct execution and package import
try:
    from .models import AnalyticalOLS
except ImportError:
    # Add parent directory to path for direct script execution
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, parent_dir)
    from models import AnalyticalOLS


def calculate_vif(X: np.ndarray) -> list:
    """
    Calculate Variance Inflation Factor (VIF) for each feature.
    
    For each feature j, fit an OLS model with feature j as the target
    and all other features as predictors. The VIF is calculated as:
    VIF_j = 1 / (1 - R_j^2)
    
    Parameters:
    -----------
    X : np.ndarray
        Design matrix of shape (n_samples, n_features)
    
    Returns:
    --------
    list : VIF values for each feature
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    vif_values = []
    
    for j in range(n_features):
        # Extract target: j-th column
        y_j = X[:, j].astype(np.float64)
        
        # Extract predictors: all columns except j
        X_excluding_j = np.delete(X, j, axis=1).astype(np.float64)
        
        # Fit OLS model: y_j ~ X_excluding_j
        model = AnalyticalOLS()
        model.fit(X_excluding_j, y_j)
        
        # Calculate R-squared
        r_squared = model.score(X_excluding_j, y_j)
        
        # Avoid division by zero and negative R-squared
        r_squared = max(0, r_squared)
        r_squared = min(0.9999, r_squared)  # Cap to avoid inf
        
        # Calculate VIF
        vif = 1 / (1 - r_squared)
        vif_values.append(vif)
    
    return vif_values
