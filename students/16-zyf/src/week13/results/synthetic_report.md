
# Week 13 Task A Synthetic Report

## 1. Data Generating Process

本任务自己生成了一份高度共线性的模拟回归数据。

真实 DGP 为：

y = 5 + 3.0 * x1_marketing - 2.0 * x4_price + 1.5 * x5_service + noise

其中：

- x1_marketing 是真实有效变量
- x4_price 是真实有效变量
- x5_service 是真实有效变量
- x2_marketing_copy 和 x3_marketing_copy 是 x1 的高度相关复制变量
- noise1、noise2、noise3 是纯噪声变量

高度相关特征组为：

- x1_marketing
- x2_marketing_copy
- x3_marketing_copy

## 2. OLS vs Ridge 系数稳定性

下面是 50 次随机切分后，共线变量系数标准差：

| model | feature | coef_std |
| --- | --- | --- |
| OLS | x1_marketing | 0.4178 |
| OLS | x2_marketing_copy | 0.5813 |
| OLS | x3_marketing_copy | 0.6273 |
| Ridge_alpha_10 | x1_marketing | 0.0456 |
| Ridge_alpha_10 | x2_marketing_copy | 0.0543 |
| Ridge_alpha_10 | x3_marketing_copy | 0.0628 |

从结果可以看到，OLS 在共线变量上的系数波动更明显，而 Ridge 通过 penalty 收缩系数后，通常更加稳定。

对应箱线图保存在：

src/week13/results/figures/synthetic_ols_vs_ridge_boxplot.png

## 3. 为什么正则化前必须标准化

Ridge、Lasso 和 Elastic Net 都会对系数大小进行惩罚。

如果不同特征的量纲差异很大，例如一个变量范围是 0 到 1，另一个变量范围是 0 到 10000，那么 penalty 对不同变量就不公平。

所以在正则化模型前必须标准化。

## 4. GridSearchCV 最优参数

| Model | Best Params | Best CV RMSE |
|---|---|---:|
| Ridge | {'model__alpha': np.float64(0.7196856730011514)} | 0.9854 |
| Lasso | {'model__alpha': np.float64(0.026826957952797246)} | 0.9838 |
| ElasticNet | {'model__alpha': np.float64(0.019306977288832496), 'model__l1_ratio': 0.9} | 0.9835 |

CV 曲线保存在 figures 文件夹中。

## 5. 测试集表现对比

| model | RMSE | MAE |
| --- | --- | --- |
| OLS | 1.1231 | 0.8977 |
| Ridge | 1.1249 | 0.8994 |
| Lasso | 1.1095 | 0.8855 |
| ElasticNet | 1.1157 | 0.8915 |

## 6. Ridge 系数

| feature | coef |
| --- | --- |
| x1_marketing | 1.738 |
| x2_marketing_copy | 0.6095 |
| x3_marketing_copy | 0.5223 |
| x4_price | -2.0166 |
| x5_service | 1.6475 |
| noise1 | -0.1206 |
| noise2 | 0.0436 |
| noise3 | -0.03 |

## 7. Lasso 系数

| feature | coef |
| --- | --- |
| x1_marketing | 2.2203 |
| x2_marketing_copy | 0.4514 |
| x3_marketing_copy | 0.1699 |
| x4_price | -1.99 |
| x5_service | 1.627 |
| noise1 | -0.0989 |
| noise2 | 0.0136 |
| noise3 | -0.0 |

## 8. Elastic Net 系数

| feature | coef |
| --- | --- |
| x1_marketing | 1.7171 |
| x2_marketing_copy | 0.6549 |
| x3_marketing_copy | 0.4782 |
| x4_price | -1.9963 |
| x5_service | 1.6326 |
| noise1 | -0.1058 |
| noise2 | 0.0236 |
| noise3 | -0.0093 |

## 9. Lasso 与前向选择变量名单对比

Lasso 选择出的非零变量：

['x1_marketing', 'x2_marketing_copy', 'x3_marketing_copy', 'x4_price', 'x5_service', 'noise1', 'noise2']

前向选择 Top-5 选择出的变量：

['x2_marketing_copy', 'x4_price', 'x5_service', 'x1_marketing', 'noise3']

## 10. 解释

Ridge 倾向于保留所有变量，只是把系数整体缩小，因此在共线变量组中通常会保留多个相关变量。

Lasso 更像变量筛选工具，可能只保留共线变量组中的一个，把其他变量压缩为 0。

Elastic Net 是 Ridge 和 Lasso 的折中，既有变量筛选能力，也不会像 Lasso 那样过于激进地只保留一个变量。
