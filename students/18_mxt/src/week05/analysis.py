import numpy as np
import matplotlib.pyplot as plt

def calculate_covariance_matrices(
    X: np.ndarray, beta_hat: np.ndarray, sigma: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    empirical_cov = np.cov(beta_hat.T)
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    XTX = X_b.T @ X_b
    theoretical_cov = sigma**2 * np.linalg.inv(XTX)[1:, 1:]
    return empirical_cov, theoretical_cov

def plot_covariance_scatter(
    beta_hat_A: np.ndarray,
    beta_hat_B: np.ndarray,
    beta_true: np.ndarray = np.array([5.0, 3.0]),
    save_path: str = "covariance_scatter.png",
) -> None:

    plt.figure(figsize=(10, 8), dpi=150)

    plt.scatter(
        beta_hat_A[:, 0],
        beta_hat_A[:, 1],
        alpha=0.6,
        s=15,
        label=r"Exp A ($\rho=0.0$)",
        color="#1f77b4",
    )

    plt.scatter(
        beta_hat_B[:, 0],
        beta_hat_B[:, 1],
        alpha=0.6,
        s=15,
        label=r"Exp B ($\rho=0.99$)",
        color="#ff7f0e",
    )

    plt.scatter(
        beta_true[0],
        beta_true[1],
        marker="*",
        s=300,
        color="red",
        label=r"True $\beta$",
        zorder=5,
    )

    plt.xlabel(r"$\hat{\beta}_1$", fontsize=14)
    plt.ylabel(r"$\hat{\beta}_2$", fontsize=14)
    plt.title(r"$\hat{\beta}_1$ vs $\hat{\beta}_2$ (Orthogonal vs Collinear)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()