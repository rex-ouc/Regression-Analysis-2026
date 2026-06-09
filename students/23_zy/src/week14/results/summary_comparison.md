# Week 14 Summary Comparison：Lasso vs PCR

## 1. 两种数据世界

本次比较了两种不同的数据生成机制：

1. Sparse truth：只有少数原始变量真正直接决定 y，其他变量主要是噪声。
2. Latent-factor truth：大量原始变量由少数潜在因子生成，y 也主要由这些潜在因子驱动。

## 2. Lasso 与 PCR 结果对比

| 场景 | 方法 | test_RMSE | test_MAE | 复杂度指标 | 复杂度数值 | 稳定性指标_test_RMSE标准差 |
| --- | --- | --- | --- | --- | --- | --- |
| Sparse truth | Lasso | 0.9790 | 0.7804 | 非零系数个数 | 11 | 0.0839 |
| Sparse truth | PCR | 7.3917 | 5.7958 | 保留主成分个数k | 16 | 0.4611 |
| Latent-factor truth | Lasso | 1.0808 | 0.8476 | 非零系数个数 | 13 | 0.1019 |
| Latent-factor truth | PCR | 1.0866 | 0.8207 | 保留主成分个数k | 18 | 0.0871 |

对比图保存位置：

```text
src/week14/figures/lasso_vs_pcr_comparison.png
```

这张图把两个场景分开展示，横轴是数据场景，纵轴是测试集 RMSE，每组柱子分别代表 Lasso 和 PCR。

## 3. 核心问题回答

### 3.1 当数据真的是 sparse truth 时，为什么 Lasso 更自然？

因为 sparse truth 的特点是只有少数原始变量真正有用。Lasso 通过 L1 正则化可以把很多无关变量的系数压缩为 0，所以它天然适合做变量筛选。

### 3.2 当数据更像 latent-factor truth 时，为什么 PCR 更自然？

因为 latent-factor truth 里，很多原始变量都在重复表达少数潜在因子的信息。这时问题的重点不是从原始变量中挑出某几列，而是把这些重复信息压缩成少数几个主成分。PCR 正好适合这种信息压缩任务。

### 3.3 Lasso 回答“谁留下”，PCR 回答什么？

Lasso 更像是在回答：哪些原始变量应该留下。PCR 更像是在回答：能不能把原始高维信息压缩成几个更稳定的综合方向。

### 3.4 如果业务方要求一个更短的变量名单，用哪个方法？

更可能使用 Lasso。因为 Lasso 可以直接给出非零系数变量，也就是一个更短的变量名单。

### 3.5 如果业务方要求一个更稳的预测器，用哪个方法？

如果数据存在明显共线性或潜在因子结构，我更可能使用 PCR。因为 PCR 通过主成分压缩可以减少冗余信息带来的不稳定。

### 3.6 为什么本周主线更适合比较 Lasso vs PCR？

因为本周的核心是 selection vs compression。Lasso 是典型的变量筛选方法，PCR 是典型的信息压缩方法。前向/后向选择也属于 selection 路线，但它们不能很好地体现 PCA/PCR 这种主成分压缩思想。

如果一定要加入前向/后向选择，它们更接近 selection 路线，而不是 compression 路线。