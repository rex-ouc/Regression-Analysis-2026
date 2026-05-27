import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from simulation import betas_A, betas_B, beta_true

plt.figure(figsize=(8, 8))

plt.scatter(betas_A[:, 0], betas_A[:, 1], 
            c='tab:blue', alpha=0.4, s=12, label='ρ=0.0 (Orthogonal)')

plt.scatter(betas_B[:, 0], betas_B[:, 1], 
            c='tab:red', alpha=0.4, s=12, label='ρ=0.99 (Multicollinear)')

plt.scatter(beta_true[0], beta_true[1], 
            c='black', marker='*', s=200, label='True β = (5, 3)')

plt.xlabel(r'$\hat{\beta}_1$')
plt.ylabel(r'$\hat{\beta}_2$')
plt.title('Monte Carlo Simulation: β Estimates')
plt.legend()
plt.grid(alpha=0.3)
plt.axis('equal')
plt.tight_layout()

plt.savefig('covariance_scatter.png', dpi=300)
plt.close()

print("✅ Figure saved as covariance_scatter.png")