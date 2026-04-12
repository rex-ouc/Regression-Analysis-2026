import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.stats.anova import anova_lm
import matplotlib
matplotlib.use('Agg')

# 设置路径
report_dir = "12_wxy/docs/week02_report"
report_file = os.path.join(report_dir, "week02_report.md")
os.makedirs(report_dir, exist_ok=True)

# 生成数据
np.random.seed(42)
beta_0, beta_1 = 1, 2
n = 100
X = np.random.normal(0, 1, n)
y = beta_0 + beta_1 * X + np.random.normal(0, 1, n)

# 手动计算
X_const = sm.add_constant(X)
beta_hat = np.linalg.inv(X_const.T @ X_const) @ X_const.T @ y
b0_manual, b1_manual = beta_hat

# sklearn
sk_model = LinearRegression().fit(X.reshape(-1, 1), y)
b0_sk, b1_sk = sk_model.intercept_, sk_model.coef_[0]

# statsmodels
sm_model = sm.OLS(y, X_const).fit()
b0_sm, b1_sm = sm_model.params[0], sm_model.params[1]

# 方差分析
null_model = sm.OLS(y, sm.add_constant(np.ones(n))).fit()
anova_table = anova_lm(null_model, sm_model)

# 创建结果表格
results_df = pd.DataFrame({
    '方法': ['手动计算', 'sklearn', 'statsmodels'],
    'β₀ 估计值': [b0_manual, b0_sk, b0_sm],
    'β₁ 估计值': [b1_manual, b1_sk, b1_sm],
    'β₁ 标准误': [np.sqrt(sm_model.cov_params().iloc[1,1])] * 3,
    'β₁ p值': [sm_model.pvalues[1]] * 3
})

# 绘制图像
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：数据点与回归线
axes[0].scatter(X, y, alpha=0.6, label='观测数据', color='steelblue')
X_sorted = np.sort(X)
y_fit = b0_sm + b1_sm * X_sorted
axes[0].plot(X_sorted, y_fit, 'r-', linewidth=2, label='拟合回归线')
axes[0].axhline(y=beta_0, color='gray', linestyle='--', alpha=0.5, label=f'真实截距 β₀={beta_0}')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('线性回归拟合结果', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 右图：残差分布
residuals = sm_model.resid
axes[1].scatter(sm_model.fittedvalues, residuals, alpha=0.6, color='steelblue')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
axes[1].set_xlabel('拟合值', fontsize=12)
axes[1].set_ylabel('残差', fontsize=12)
axes[1].set_title('残差图', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'regression_plots.png'), dpi=150, bbox_inches='tight')
plt.close()

# 生成报告内容
report_content = f"""# 线性回归参数估计实验报告

## 1. 实验目的

- 比较手动计算、scikit-learn、statsmodels 三种方法在线性回归参数估计中的一致性
- 进行参数显著性检验与整体模型的方差分析
- 通过可视化展示回归结果

## 2. 数据生成

实验数据由以下线性模型生成：

$$y = \\beta_0 + \\beta_1 X + \\varepsilon, \\quad \\varepsilon \\sim N(0, 1)$$

**参数设置**：
- 真实截距：$\\beta_0 = 1$
- 真实斜率：$\\beta_1 = 2$
- 样本量：$n = 100$

自变量 $X$ 从标准正态分布中随机生成，误差项 $\\varepsilon$ 服从均值为 0、方差为 1 的正态分布。

## 3. 参数估计结果对比

三种方法的参数估计结果如下表所示：

{results_df.round(4).to_markdown(index=False)}

**结果分析**：
- 三种方法得到的参数估计值完全一致，验证了手动计算与库函数实现的等价性
- $\\beta_1$ 的标准误为 {results_df['β₁ 标准误'][0]:.4f}，p 值远小于 0.05，表明 $\\beta_1$ 在统计上显著不为零

## 4. 假设检验

### 4.1 斜率显著性检验

检验假设：$H_0: \\beta_1 = 0$ vs $H_1: \\beta_1 \\neq 0$

- t 统计量：{sm_model.tvalues[1]:.4f}
- p 值：{sm_model.pvalues[1]:.4f}

由于 p 值 < 0.05，拒绝原假设，认为 $\\beta_1$ 显著不为零，自变量 $X$ 对因变量 $y$ 有显著影响。

### 4.2 整体模型显著性检验

检验假设：$H_0$: 模型整体不显著 vs $H_1$: 模型整体显著

- F 统计量：{sm_model.fvalue:.4f}
- p 值：{sm_model.f_pvalue:.4f}

整体模型显著，说明回归方程具有解释力。

## 5. 方差分析

方差分析表如下：

{anova_table.round(4).to_markdown()}

**解释**：
- **df_model**：模型自由度（1个自变量）
- **df_resid**：残差自由度（n - 2 = 98）
- **F**：F 统计量，与整体检验一致
- **PR(>F)**：p 值，表明模型显著

## 6. 可视化结果

![回归结果图](regression_plots.png)

**左图**：散点图展示了原始数据点，红色实线为拟合的回归线，灰色虚线为真实截距位置。拟合线较好地捕捉了数据的线性趋势。

**右图**：残差图展示了拟合值与残差的关系。残差随机分布在零线两侧，无明显模式，说明线性模型假设合理。

## 7. 结论

1. **方法一致性**：手动计算、scikit-learn、statsmodels 三种方法得到的参数估计结果完全一致，验证了 OLS 解析解的正确性。

2. **统计显著性**：$\\beta_1$ 的 p 值远小于 0.05，表明 $X$ 对 $y$ 有显著影响，这与真实数据生成过程一致（$\\beta_1 = 2 \\neq 0$）。

3. **模型有效性**：整体 F 检验显著，方差分析结果与 t 检验结论一致，模型整体具有统计意义。

4. **诊断检验**：残差图未发现明显异方差性或非线性模式，模型拟合良好。

## 附录：代码说明

实验使用 Python 完成，主要依赖库包括：
- `numpy`：数据生成与矩阵运算
- `pandas`：结果整理与表格展示
- `matplotlib`：图形绘制
- `statsmodels`：统计建模与假设检验
- `sklearn`：机器学习方法实现线性回归

### 关键结果总结

- **回归方程**：$\\hat{{y}} = {b0_sm:.4f} + {b1_sm:.4f}X$
- **拟合优度 R²**：{sm_model.rsquared:.4f}
- **调整 R²**：{sm_model.rsquared_adj:.4f}
- **样本量**：{n}


