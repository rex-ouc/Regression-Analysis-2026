import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# 设置随机种子
np.random.seed(42)

# 参数设置
beta_0_true = 1
beta_1_true = 2
n_samples = 100

# 生成数据
X = np.random.uniform(0, 10, n_samples)
epsilon = np.random.normal(0, 1, n_samples)
y = beta_0_true + beta_1_true * X + epsilon

# 整理为二维数组
X_reshaped = X.reshape(-1, 1)

# 手动计算
x_bar, y_bar = np.mean(X), np.mean(y)
beta_1_hat = np.sum((X - x_bar) * (y - y_bar)) / np.sum((X - x_bar)**2)
beta_0_hat = y_bar - beta_1_hat * x_bar

# 计算方差
y_pred_manual = beta_0_hat + beta_1_hat * X
residual_var = np.sum((y - y_pred_manual)**2) / (n_samples - 2)
beta_1_var = residual_var / np.sum((X - x_bar)**2)

print(f"手动估计: beta_0 = {beta_0_hat:.4f}, beta_1 = {beta_1_hat:.4f}")
print(f"Bias: beta_0: {beta_0_hat - beta_0_true:.4f}, beta_1: {beta_1_hat - beta_1_true:.4f}")

# 1. Sklearn
sk_model = LinearRegression().fit(X_reshaped, y)

# 2. Statsmodels 
X_with_const = sm.add_constant(X)
sm_model = sm.OLS(y, X_with_const).fit()

# 对比表格
results = {
    "Method": ["True Value", "Manual OLS", "Sklearn", "Statsmodels"],
    "Beta_0": [beta_0_true, beta_0_hat, sk_model.intercept_, sm_model.params[0]],
    "Beta_1": [beta_1_true, beta_1_hat, sk_model.coef_[0], sm_model.params[1]]
}
df_compare = pd.DataFrame(results)
print(df_compare)

import statsmodels.formula.api as smf

df = pd.DataFrame({'x': X, 'y': y})
smf_model = smf.ols('y ~ x', data=df).fit()

from statsmodels.stats.anova import anova_lm
anova_results = anova_lm(smf_model)
print(anova_results)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data Points')
plt.plot(X, y_pred_manual, color='red', label='Fitted Line')
plt.title("Linear Regression Analysis")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()