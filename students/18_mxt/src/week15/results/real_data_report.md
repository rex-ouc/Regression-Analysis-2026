# Week 15 任务E：真实数据挑战

## E1. 真实二分类数据

我使用 sklearn 内置的 Breast Cancer Wisconsin 数据集作为真实二分类任务。原始数据包含 569 个样本和 30 个数值特征。本报告把 malignant 设为正类，数据保存为 `src/week15/data/real_binary_breast_cancer.csv`。

正类比例为 `0.3726`。这说明数据并非完全均衡，因此不能只盯着 accuracy。

## E2. 完整逻辑回归流程

流程包括：数据读取、正类定义、训练/测试划分、标准化、普通 LogisticRegression 训练、阈值扫描和图形分析。阈值 0.5 下的主要结果如下：

| 指标 | 数值 |
|:--|--:|
| accuracy | 0.9708 |
| precision | 0.9836 |
| recall | 0.9375 |
| F1 | 0.9600 |
| ROC-AUC | 0.9975 |
| log loss | 0.0675 |

图 `figures/real_threshold_metrics.png` 展示真实数据上的阈值权衡：

- 横轴是分类阈值 threshold。
- 纵轴是指标值。
- 四条线分别是 accuracy、precision、recall、F1。

## E3. 业务解释

在这个数据里，单看 accuracy 可能误导判断。医疗筛查更怕漏诊，所以 recall 和 ROC-AUC 更值得关注。accuracy 高不代表模型没有漏掉恶性样本。

我最后更信任 recall、F1 和 ROC-AUC 的组合。ROC-AUC 反映整体排序能力，recall 反映能否抓住真正恶性病例，F1 则在 precision 和 recall 之间做折中。

如果向业务方解释模型输出，我会强调“概率”而不只是“类别”。因为概率可以配合不同阈值服务不同风险偏好：初筛阶段可以降低阈值提高召回，复核阶段可以提高阈值减少误报。本次 F1 最优阈值约为 `0.30`，对应 F1 为 `0.9764`。
