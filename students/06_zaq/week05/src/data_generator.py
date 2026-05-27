"""
Data generating process for multicollinearity experiment.
Task 1: Generate correlated features X1 and X2.
"""

import numpy as np


def generate_design_matrix(n_samples: int = 100, rho: float = 0.0, random_seed: int = 42):
    """
    Generate design matrix X with two correlated features.
    
    Args:
        n_samples: Number of samples
        rho: Correlation coefficient between X1 and X2 (-1 to 1)
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Design matrix of shape (n_samples, 2)
        correlation_matrix: The correlation matrix used
    """
    np.random.seed(random_seed)
    
    # Create covariance matrix for desired correlation
    # Cov(X1, X2) = rho, Var(X1) = Var(X2) = 1
    cov_matrix = np.array([[1.0, rho],
                           [rho, 1.0]])
    
    # Generate correlated samples using Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    Z = np.random.randn(n_samples, 2)
    X = Z @ L.T
    
    return X, cov_matrix


def generate_data_fixed_design(n_samples: int = 100, rho: float = 0.0, 
                                true_beta: np.ndarray = None, 
                                sigma: float = 2.0,
                                random_seed: int = 42):
    """
    Generate data with fixed design matrix X.
    
    According to the statistics rule: X is generated once and fixed,
    only new noise ε is generated each time.
    
    Args:
        n_samples: Number of samples
        rho: Correlation between X1 and X2
        true_beta: True coefficients (default: [5.0, 3.0])
        sigma: Standard deviation of noise
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Fixed design matrix (generated once)
        y_generator: Function that generates new y's given X
        true_beta: True coefficients
    """
    if true_beta is None:
        true_beta = np.array([5.0, 3.0])
    
    # Generate fixed design matrix X (only once!)
    X, cov_matrix = generate_design_matrix(n_samples, rho, random_seed)
    
    def generate_y(noise_seed: int = None):
        """Generate new y values using the fixed X and new noise."""
        if noise_seed is not None:
            np.random.seed(noise_seed)
        epsilon = sigma * np.random.randn(n_samples)
        y = X @ true_beta + epsilon
        return y
    
    return X, generate_y, true_beta, cov_matrix


def generate_data_for_simulation(n_samples: int = 100, rho: float = 0.0,
                                   n_simulations: int = 1000,
                                   true_beta: np.ndarray = None,
                                   sigma: float = 2.0,
                                   random_seed: int = 42):
    """
    Generate data for Monte Carlo simulation.
    
    Args:
        n_samples: Number of samples per simulation
        rho: Correlation between X1 and X2
        n_simulations: Number of Monte Carlo runs
        true_beta: True coefficients
        sigma: Noise standard deviation
        random_seed: Base random seed
        
    Returns:
        X_fixed: Fixed design matrix (same for all simulations)
        betas_hat: Array of estimated coefficients (n_simulations, 2)
        true_beta: True coefficients
    """
    # Generate fixed design matrix once
    X_fixed, generate_y, true_beta, _ = generate_data_fixed_design(
        n_samples, rho, true_beta, sigma, random_seed
    )
    
    betas_hat = []
    
    for i in range(n_simulations):
        # Generate new noise each time
        y = generate_y(noise_seed=random_seed + i + 1000)
        
        # OLS estimate: β_hat = (X^T X)^(-1) X^T y
        XtX = X_fixed.T @ X_fixed
        Xty = X_fixed.T @ y
        beta_hat = np.linalg.solve(XtX, Xty)
        betas_hat.append(beta_hat)
    
    return X_fixed, np.array(betas_hat), true_beta