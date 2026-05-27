import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['axes.unicode_minus'] = False 
# =====================================================================

results = np.load("simulation_results.npz")
beta_hat_A = results["beta_A"]
beta_hat_B = results["beta_B"]
TRUE_BETA = np.array([5.0, 3.0])

plt.figure(figsize=(10, 8), dpi=100)

plt.scatter(beta_hat_A[:, 0], beta_hat_A[:, 1], 
            color="#1f77b4", alpha=0.6, s=2, label="A (orthogonal)")

plt.scatter(beta_hat_B[:, 0], beta_hat_B[:, 1], 
            color="#ff4b5c", alpha=0.6, s=2, label="B (collinear)")

plt.scatter(TRUE_BETA[0], TRUE_BETA[1], 
            color="black", s=100, marker="*", label="True Beta")


plt.xlabel(r"$\hat{\beta}_1$", fontsize=14)
plt.ylabel(r"$\hat{\beta}_2$", fontsize=14)
plt.title("Beta Estimation Distribution under Multicollinearity", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.axis("equal")

plt.savefig("beta_estimation_scatter.png", bbox_inches="tight")
plt.close()
print("✅ 图片已生成为beta_estimation_scatter.png")