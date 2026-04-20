# analysis.py
import matplotlib.pyplot as plt
from simulation import run_simulation
import numpy as np

def plot_betas(betas_A, betas_B, beta_true=np.array([5.0, 3.0])):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(betas_A[:,0], betas_A[:,1], alpha=0.5, label='Orthogonal (rho=0.0)')
    plt.scatter(betas_B[:,0], betas_B[:,1], alpha=0.5, label='High Collinearity (rho=0.99)')
    
    plt.scatter(beta_true[0], beta_true[1], color='red', marker='x', s=100, label='True β')
    
    plt.xlabel('β1 estimates')
    plt.ylabel('β2 estimates')
    plt.title('Monte Carlo β Estimates: Orthogonal vs Collinear Features')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    betas_A, _ = run_simulation(rho=0.0)
    betas_B, _ = run_simulation(rho=0.99)
    plot_betas(betas_A, betas_B)