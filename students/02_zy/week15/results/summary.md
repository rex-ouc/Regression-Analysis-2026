# 第 15 周总结

## 1. 为什么逻辑回归不只是“线性回归加 sigmoid”

逻辑回归和线性回归的统计假设不同。逻辑回归假设目标变量服从 Bernoulli 分布，用 sigmoid 把线性得分映射为概率，并通过最大化 Bernoulli likelihood 来估计参数。这个过程自然会导出 log loss。

在模拟数据实验中，ROC-AUC 更高的模型是 `LinearRegression`。

## 2. Sigmoid、Bernoulli Likelihood 与 Log Loss 的关系

sigmoid 函数把任意实数线性得分转换成 0 到 1 之间的概率。Bernoulli likelihood 衡量在这个概率下，真实观测到的 0/1 标签有多可能发生。对 likelihood 取负对数，就得到 log loss。

## 3. 为什么分类模型不能只看 Accuracy

accuracy 会掩盖 false positive 和 false negative 的差异。阈值分析说明，同一个概率模型在不同阈值下会得到不同的 precision 和 recall。因此，分类任务不能只看 accuracy，指标选择必须结合业务场景。

## 4. L1 与 L2 逻辑回归分别适合什么目标

L1 更适合变量筛选和给出较短的特征名单。L2 更适合在变量相关性较强时追求稳定预测，因为它倾向于平滑收缩系数，而不是直接删除大量变量。

在本次实验中，L1 保留了 `19` 个非零系数，L2 保留了 `25` 个非零系数。

## 5. 为什么逻辑回归仍然是很强的 baseline

逻辑回归可以输出概率，支持根据业务目标调整阈值，系数方向也比较容易解释。同时，它还能通过 L1 或 L2 正则化处理高维数据。因此在很多二分类任务中，逻辑回归仍然是一个透明、稳定、很有价值的 baseline 模型。
