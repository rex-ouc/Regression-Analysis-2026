
# Week 13 Summary Comparison Report

## 1. Synthetic Data Results

| model | RMSE | MAE |
| --- | --- | --- |
| OLS | 1.1231 | 0.8977 |
| Ridge | 1.1249 | 0.8994 |
| Lasso | 1.1095 | 0.8855 |
| ElasticNet | 1.1157 | 0.8915 |

模拟数据中，DGP 是已知的，所以我们可以直接判断模型是否找到了真正有效的变量。

Lasso 选择变量：

['x1_marketing', 'x2_marketing_copy', 'x3_marketing_copy', 'x4_price', 'x5_service', 'noise1', 'noise2']

Forward Selection 选择变量：

['x2_marketing_copy', 'x4_price', 'x5_service', 'x1_marketing', 'noise3']

## 2. Kaggle Data Results

| model | RMSE | MAE |
| --- | --- | --- |
| OLS | 587.2208 | 201.7243 |
| Ridge | 587.1699 | 201.2009 |
| Lasso | 587.1888 | 201.4617 |
| ElasticNet | 587.1681 | 201.1803 |

Kaggle 房价数据中，真实 DGP 不知道，因此我们只能做推测解释。

Lasso 给出的 Top 5 重要变量：

['SQUARE_FT', 'BHK_NO.', 'POSTED_BY_Dealer', 'POSTED_BY_Owner', 'RESALE']

## 3. Lasso 在共线变量组中的业务风险

Lasso 面对高度相关变量时，可能只保留其中一个变量，而把其他相关变量压缩为 0。

这在业务上有风险。

因为被压缩为 0 的变量不一定真的没有价值，只可能是它和另一个变量信息重复。

## 4. Elastic Net 如何缓解这个问题

Elastic Net 同时结合 Ridge 和 Lasso。

它既可以筛变量，也可以在相关变量组中保留一部分整体结构。

所以在高度相关变量成组出现时，Elastic Net 往往比 Lasso 更稳健。

## 5. GridSearchCV 与主观偏好

GridSearchCV 的目标是找到验证误差最低的超参数。

但是业务上有时还会追求：

- 模型更稀疏
- 模型更稳定
- 模型更容易解释

所以最低验证误差不一定等于最符合业务解释需求。

## 6. Forward Selection 与 Lasso 对比

Forward Selection 是一步一步加入变量，逻辑直观，但计算成本较高。

Lasso 通过正则化自动实现变量筛选，效率更高。

但是 Lasso 在共线变量组中可能选择不稳定，因此结果解释需要谨慎。

## 7. Final Conclusion

Week13 的核心结论是：

OLS 在共线性场景下系数容易不稳定。

Ridge 适合提升稳定性。

Lasso 适合变量筛选。

Elastic Net 适合处理成组相关变量。

传统变量选择方法可以作为对照，帮助我们理解 Lasso 的筛选结果是否合理。
