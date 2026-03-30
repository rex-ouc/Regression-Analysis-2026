import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.stats.anova import anova_lm

# 生成数据
np.random.seed(42)
beta_0, beta_1 = 1, 2
n = 100
X = np.random.normal(0, 1, n)
y = beta_0 + beta_1 * X + np.random.normal(0, 1, n)

# 手动计算
X_const = sm.add_constant(X)
beta_hat = np.linalg.inv(X_const.T @ X_const) @ X_const.T @ y
print(f"手动: beta0={beta_hat[0]:.4f}, beta1={beta_hat[1]:.4f}")

# sklearn
sk_model = LinearRegression().fit(X.reshape(-1,1), y)
print(f"sklearn: beta0={sk_model.intercept_:.4f}, beta1={sk_model.coef_[0]:.4f}")

# statsmodels
sm_model = sm.OLS(y, X_const).fit()
print(f"statsmodels: beta0={sm_model.params[0]:.4f}, beta1={sm_model.params[1]:.4f}")

# 假设检验与方差分析
print(f"\nbeta1=0 的 p 值: {sm_model.pvalues[1]:.4f}")
print(f"整体 F 检验 p 值: {sm_model.f_pvalue:.4f}")

null_model = sm.OLS(y, sm.add_constant(np.ones(n))).fit()
print("\n方差分析表:")
print(anova_lm(null_model, sm_model))