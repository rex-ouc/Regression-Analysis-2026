from simulation import loop, analysis

if __name__ == "__main__":
    df = loop(n_sim=100, n=100)
    analysis(df)

from simulation import monte_carlo, covariance_analysis
from analysis import plot_results
import numpy as np

if __name__ == "__main__":
    beta_true = np.array([5.0, 3.0])

    betas_A, X_A = monte_carlo(rho=0.0)
    betas_B, X_B = monte_carlo(rho=0.99)

    emp_B, theo_B = covariance_analysis(betas_B, X_B)

    print("Empirical Covariance:\n", emp_B)
    print("Theoretical Covariance:\n", theo_B)

    plot_results(betas_A, betas_B, beta_true)
from simulation import monte_carlo, covariance_analysis
from analysis import plot_results
import numpy as np

if __name__ == "__main__":
    beta_true = np.array([5.0, 3.0])

    betas_A, X_A = monte_carlo(rho=0.0)
    betas_B, X_B = monte_carlo(rho=0.99)

    emp_B, theo_B = covariance_analysis(betas_B, X_B)

    print("Empirical Covariance:\n", emp_B)
    print("Theoretical Covariance:\n", theo_B)

    plot_results(betas_A, betas_B, beta_true)