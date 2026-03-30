import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
np.random.seed(42)
x = np.linspace(0, 10, 100)
beta0_true = 1
beta1_true = 2
epsilon = np.random.normal(0, 1, size=100)
y = beta0_true + beta1_true * x + epsilon
x_bar = np.mean(x)
y_bar = np.mean(y)
numerator = np.sum((x - x_bar) * (y - y_bar))
denominator = np.sum((x - x_bar)**2)
beta1_hat = numerator / denominator
beta0_hat = y_bar - beta1_hat * x_bar
print(f"手动估计: beta0 = {beta0_hat:.4f}, beta1 = {beta1_hat:.4f}")
print(f"真实值:    beta0 = {beta0_true}, beta1 = {beta1_true}")
sk_model = LinearRegression()
sk_model.fit(x.reshape(-1, 1), y)
sk_beta0 = sk_model.intercept_
sk_beta1 = sk_model.coef_[0]
print(f"\nsklearn 估计: beta0 = {sk_beta0:.4f}, beta1 = {sk_beta1:.4f}")
X_sm = sm.add_constant(x)
sm_model = sm.OLS(y, X_sm).fit()
print("\nstatsmodels 结果:")
print(sm_model.summary())
plt.figure(figsize=(8, 5))
plt.scatter(x, y, label="原始数据", alpha=0.6)
plt.plot(x, beta0_hat + beta1_hat * x, color="red", label="拟合直线")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("一元线性回归拟合")
plt.savefig("regression_plot.png")
plt.show()