# -*- coding: utf-8 -*-
"""
第二周作业：一元回归分析
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1、生成随机数
np.random.seed(42)
beta_0_true = 1
beta_1_true = 2
n_samples = 100
X = np.random.uniform(0,10,n_samples)
epsilon = np.random.normal(0, 1, n_samples)
y = beta_0_true + beta_1_true * X + epsilon
data = pd.DataFrame({'X': X, 'y': y})
print(data.head())

# 2、估计参数
X_mean = np.mean(X)
y_mean = np.mean(y)
Sxx = np.sum((X - X_mean)**2)
Sxy = np.sum((X - X_mean) * (y - y_mean))
beta_1_manual = Sxy / Sxx
beta_0_manual = y_mean - beta_1_manual * X_mean

print("【公式手动计算结果】")
print(f"β₀ = {beta_0_manual:.6f}")
print(f"β₁ = {beta_1_manual:.6f}")

# 残差方差的无偏估计 σ² = RSS/(n-2)
residuals = y - (beta_0_manual + beta_1_manual * X)
RSS = np.sum(residuals ** 2)
sigma_sq = RSS / (n_samples - 2)
var_beta_1 = sigma_sq / Sxx
print(f"β₁ 估计值的方差: {var_beta_1:.8f}")

# 计算偏差 (bias)
bias_0 = beta_0_manual - beta_0_true
bias_1 = beta_1_manual - beta_1_true
print(f"β₀ 的偏差: {bias_0:.6f}")
print(f"β₁ 的偏差: {bias_1:.6f}")
print("="*60)

# 3、使用 sklearn 进行线性回归
X_sklearn = X.reshape(-1, 1)
model_sk = LinearRegression()
model_sk.fit(X_sklearn, y)

beta_0_sk = model_sk.intercept_
beta_1_sk = model_sk.coef_[0]

print("【sklearn 结果】")
print(f"β₀ = {beta_0_sk:.6f}")
print(f"β₁ = {beta_1_sk:.6f}")
print("="*60)

# 4、statsmodels（公式版，用于比较）
model_smf = smf.ols('y ~ X', data=data)
results_smf = model_smf.fit()

beta_0_smf = results_smf.params['Intercept']
beta_1_smf = results_smf.params['X']

print("【statsmodels (公式) 结果】")
print(f"β₀ = {beta_0_smf:.6f}")
print(f"β₁ = {beta_1_smf:.6f}")
print("="*60)

# 5、statsmodels（数组版，用于比较）
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm)
results_sm = model_sm.fit()

beta_0_sm = results_sm.params[0]
beta_1_sm = results_sm.params[1]

print("【statsmodels (数组) 结果】")
print(f"β₀ = {beta_0_sm:.6f}")
print(f"β₁ = {beta_1_sm:.6f}")
print("="*60)

# 6、最终对比
print("\n【方法对比总结】")
print(f"{'方法':<20} {'β₀':<12} {'β₁':<12}")
print(f"{'真实值':<20} {beta_0_true:<12.6f} {beta_1_true:<12.6f}")
print(f"{'手动公式':<20} {beta_0_manual:<12.6f} {beta_1_manual:<12.6f}")
print(f"{'sklearn':<20} {beta_0_sk:<12.6f} {beta_1_sk:<12.6f}")
print(f"{'statsmodels(公式)':<20} {beta_0_smf:<12.6f} {beta_1_smf:<12.6f}")
print(f"{'statsmodels(数组)':<20} {beta_0_sm:<12.6f} {beta_1_sm:<12.6f}")