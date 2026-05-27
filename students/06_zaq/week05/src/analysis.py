"""
Visualization and analysis for multicollinearity experiment.
Task 4: Create scatter plots of β̂ estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import compare_simulations


def plot_beta_estimates(results_a, results_b, save_path: str = None):
    """
    Create scatter plot of β̂ estimates from both simulations.
    
    Args:
        results_a: Results from ρ = 0.0 simulation
        results_b: Results from ρ = 0.99 simulation
        save_path: Path to save the figure
    """
    betas_a = results_a['betas_hat']
    betas_b = results_b['betas_hat']
    true_beta = results_a['true_beta']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Experiment A: Orthogonal features (ρ = 0)
    ax1 = axes[0]
    ax1.scatter(betas_a[:, 0], betas_a[:, 1], alpha=0.5, s=10, c='steelblue', edgecolors='none')
    ax1.scatter(true_beta[0], true_beta[1], c='red', s=100, marker='*', 
                edgecolors='black', linewidths=1, label='True β')
    ax1.axhline(y=true_beta[1], color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=true_beta[0], color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('β₁ (Coefficient for X₁)', fontsize=12)
    ax1.set_ylabel('β₂ (Coefficient for X₂)', fontsize=12)
    ax1.set_title(f'Experiment A: Orthogonal Features (ρ = 0.0)\n'
                  f'Std(β₁) = {results_a["beta_hat_std"][0]:.3f}, '
                  f'Std(β₂) = {results_a["beta_hat_std"][1]:.3f}', 
                  fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for fair comparison
    x_range = np.ptp(betas_a[:, 0])
    y_range = np.ptp(betas_a[:, 1])
    max_range = max(x_range, y_range)
    x_center = np.mean(betas_a[:, 0])
    y_center = np.mean(betas_a[:, 1])
    ax1.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax1.set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    # Plot Experiment B: Highly collinear features (ρ = 0.99)
    ax2 = axes[1]
    ax2.scatter(betas_b[:, 0], betas_b[:, 1], alpha=0.5, s=10, c='coral', edgecolors='none')
    ax2.scatter(true_beta[0], true_beta[1], c='red', s=100, marker='*',
                edgecolors='black', linewidths=1, label='True β')
    ax2.axhline(y=true_beta[1], color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=true_beta[0], color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('β₁ (Coefficient for X₁)', fontsize=12)
    ax2.set_ylabel('β₂ (Coefficient for X₂)', fontsize=12)
    ax2.set_title(f'Experiment B: Highly Collinear Features (ρ = 0.99)\n'
                  f'Std(β₁) = {results_b["beta_hat_std"][0]:.3f}, '
                  f'Std(β₂) = {results_b["beta_hat_std"][1]:.3f}',
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set same aspect ratio for comparison
    x_range = np.ptp(betas_b[:, 0])
    y_range = np.ptp(betas_b[:, 1])
    max_range = max(x_range, y_range)
    x_center = np.mean(betas_b[:, 0])
    y_center = np.mean(betas_b[:, 1])
    ax2.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax2.set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    return fig


def print_covariance_matrices(results_a, results_b):
    """
    Print empirical and theoretical covariance matrices for both experiments.
    """
    print("\n" + "="*80)
    print("COVARIANCE MATRICES")
    print("="*80)
    
    for results in [results_a, results_b]:
        rho = results['rho']
        print(f"\n--- Experiment: ρ = {rho} ---")
        
        print(f"\nEmpirical Covariance Matrix (from {results['n_simulations']} simulations):")
        print(np.round(results['empirical_cov'], 6))
        
        print(f"\nTheoretical Covariance Matrix (σ² (X^T X)^(-1)):")
        print(np.round(results['theoretical_cov'], 6))
        
        # Calculate correlation from covariance
        emp_corr = results['empirical_cov'][0,1] / (results['beta_hat_std'][0] * results['beta_hat_std'][1])
        theo_corr = results['theoretical_cov'][0,1] / np.sqrt(results['theoretical_cov'][0,0] * results['theoretical_cov'][1,1])
        
        print(f"\nEmpirical Correlation between β̂₁ and β̂₂: {emp_corr:.6f}")
        print(f"Theoretical Correlation between β̂₁ and β̂₂: {theo_corr:.6f}")


def main():
    """
    Run complete analysis: simulations, plotting, and reporting.
    """
    print("\n" + "="*80)
    print("WEEK 05: Seeing the Invisible - Covariance & Multicollinearity")
    print("="*80)
    
    # Run simulations
    results_a, results_b = compare_simulations(
        rho_a=0.0,      # Orthogonal features
        rho_b=0.99,     # Highly collinear features
        n_samples=100,
        n_simulations=1000,
        random_seed=42
    )
    
    # Print covariance matrices
    print_covariance_matrices(results_a, results_b)
    
    # Create and save plot
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'beta_estimates_scatter.png')
    
    plot_beta_estimates(results_a, results_b, save_path=plot_path)
    
    # Analysis explanation
    print("\n" + "="*80)
    print("ANALYSIS: Why are β̂₁ and β̂₂ negatively correlated when X₁ and X₂ are positively correlated?")
    print("="*80)
    print("""
    When X₁ and X₂ are highly positively correlated (ρ = 0.99):
    
    1. The total sum (X₁ + X₂) is almost fixed for a given sample.
    
    2. If β̂₁ increases (say by +δ), to maintain the same prediction ŷ = β̂₁X₁ + β̂₂X₂,
       β̂₂ must decrease (approximately by -δ) because X₁ ≈ X₂.
    
    3. This creates a "budget allocation" effect: the coefficients trade off against
       each other, resulting in a strong negative correlation between β̂₁ and β̂₂.
    
    4. Mathematically, this is reflected in the covariance matrix:
       Var(β̂) = σ² (X^T X)^(-1)
       
       When X₁ and X₂ are correlated, (X^T X) is nearly singular,
       its inverse has large off-diagonal elements with opposite signs.
    
    5. This is why the scatter plot changes from a circle (ρ = 0, independent estimates)
       to a tilted ellipse (ρ = 0.99, negatively correlated estimates).
    """)
    
    return results_a, results_b


if __name__ == "__main__":
    results_a, results_b = main()