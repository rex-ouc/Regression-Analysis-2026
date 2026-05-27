"""
Monte Carlo simulation for multicollinearity experiment.
Task 2 & 3: Run simulations and compute empirical/theoretical covariance.
"""

import numpy as np
from data_generator import generate_data_for_simulation


def run_simulation(rho: float, n_samples: int = 100, 
                   n_simulations: int = 1000,
                   true_beta: np.ndarray = None,
                   sigma: float = 2.0,
                   random_seed: int = 42):
    """
    Run Monte Carlo simulation for given correlation.
    
    Args:
        rho: Correlation between X1 and X2
        n_samples: Number of samples per simulation
        n_simulations: Number of Monte Carlo runs
        true_beta: True coefficients
        sigma: Noise standard deviation
        random_seed: Random seed
        
    Returns:
        results: Dictionary containing X, betas_hat, true_beta, 
                 empirical_cov, theoretical_cov
    """
    print(f"\n{'='*60}")
    print(f"Running simulation with ρ = {rho}")
    print(f"Samples: {n_samples}, Simulations: {n_simulations}")
    print(f"True β: {true_beta if true_beta is not None else [5.0, 3.0]}")
    print(f"Noise σ: {sigma}")
    print(f"{'='*60}")
    
    # Generate data and run simulation
    X, betas_hat, true_beta = generate_data_for_simulation(
        n_samples, rho, n_simulations, true_beta, sigma, random_seed
    )
    
    # Calculate empirical covariance matrix from simulation results
    empirical_cov = np.cov(betas_hat.T)
    
    # Calculate theoretical covariance matrix: σ² (X^T X)^(-1)
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    theoretical_cov = sigma**2 * XtX_inv
    
    # Calculate mean of estimated betas
    beta_hat_mean = np.mean(betas_hat, axis=0)
    beta_hat_std = np.std(betas_hat, axis=0)
    
    # Calculate bias
    bias = beta_hat_mean - true_beta
    
    results = {
        'rho': rho,
        'X': X,
        'betas_hat': betas_hat,
        'true_beta': true_beta,
        'beta_hat_mean': beta_hat_mean,
        'beta_hat_std': beta_hat_std,
        'bias': bias,
        'empirical_cov': empirical_cov,
        'theoretical_cov': theoretical_cov,
        'n_simulations': n_simulations,
        'n_samples': n_samples,
        'sigma': sigma
    }
    
    # Print results
    print(f"\n--- Results for ρ = {rho} ---")
    print(f"Mean of β̂: [{beta_hat_mean[0]:.4f}, {beta_hat_mean[1]:.4f}]")
    print(f"True β:     [{true_beta[0]:.4f}, {true_beta[1]:.4f}]")
    print(f"Bias:       [{bias[0]:.4f}, {bias[1]:.4f}]")
    print(f"Std of β̂:   [{beta_hat_std[0]:.4f}, {beta_hat_std[1]:.4f}]")
    
    print(f"\nEmpirical Covariance Matrix:")
    print(empirical_cov)
    print(f"\nTheoretical Covariance Matrix (σ² (X^T X)^(-1)):")
    print(theoretical_cov)
    
    # Check alignment
    cov_diff = np.abs(empirical_cov - theoretical_cov)
    max_diff = np.max(cov_diff)
    print(f"\nMax difference between empirical and theoretical: {max_diff:.6e}")
    
    if max_diff < 1e-2:
        print("✓ Matrices align well!")
    else:
        print("⚠ Matrices show some discrepancy")
    
    return results


def compare_simulations(rho_a: float = 0.0, rho_b: float = 0.99,
                        n_samples: int = 100, n_simulations: int = 1000,
                        random_seed: int = 42):
    """
    Run both simulations (orthogonal vs collinear) and compare.
    
    Args:
        rho_a: Correlation for Experiment A (orthogonal)
        rho_b: Correlation for Experiment B (collinear)
        n_samples: Number of samples per simulation
        n_simulations: Number of Monte Carlo runs
        random_seed: Random seed
        
    Returns:
        results_a, results_b: Results from both simulations
    """
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION: Orthogonal vs Collinear Features")
    print("="*80)
    
    # Experiment A: Orthogonal features (ρ = 0)
    results_a = run_simulation(rho_a, n_samples, n_simulations, 
                                true_beta=np.array([5.0, 3.0]),
                                sigma=2.0, random_seed=random_seed)
    
    # Experiment B: Highly collinear features (ρ = 0.99)
    results_b = run_simulation(rho_b, n_samples, n_simulations,
                                true_beta=np.array([5.0, 3.0]),
                                sigma=2.0, random_seed=random_seed + 1)
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'ρ = 0.0':<25} {'ρ = 0.99':<25}")
    print("-" * 80)
    print(f"{'Std of β₁':<30} {results_a['beta_hat_std'][0]:<25.4f} {results_b['beta_hat_std'][0]:<25.4f}")
    print(f"{'Std of β₂':<30} {results_a['beta_hat_std'][1]:<25.4f} {results_b['beta_hat_std'][1]:<25.4f}")
    print(f"{'Corr(β₁, β₂) from emp_cov':<30} {results_a['empirical_cov'][0,1]/results_a['beta_hat_std'][0]/results_a['beta_hat_std'][1]:<25.4f} "
          f"{results_b['empirical_cov'][0,1]/results_b['beta_hat_std'][0]/results_b['beta_hat_std'][1]:<25.4f}")
    
    # Variance inflation factor
    var_ratio_1 = results_b['beta_hat_std'][0]**2 / results_a['beta_hat_std'][0]**2
    var_ratio_2 = results_b['beta_hat_std'][1]**2 / results_a['beta_hat_std'][1]**2
    print(f"\n{'Variance Inflation Factor (VIF) for β₁':<30} {1.0:<25.1f} {var_ratio_1:<25.2f}")
    print(f"{'Variance Inflation Factor (VIF) for β₂':<30} {1.0:<25.1f} {var_ratio_2:<25.2f}")
    
    return results_a, results_b


if __name__ == "__main__":
    # Quick test
    results_a, results_b = compare_simulations()