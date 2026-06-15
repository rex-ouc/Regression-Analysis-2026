# Week15 Synthetic Report：逻辑回归与二分类

## 1. 数据生成机制 DGP

本次模拟数据样本量为 600，特征数为 5。

我先生成 5 个连续特征 x1 到 x5，然后构造线性得分 eta：

```text
eta = -0.2 + 1.6*x1 - 1.3*x2 + 0.9*x3 - 0.6*x4 + 0.0*x5
```

之后通过 sigmoid 函数把 eta 转成正类概率 p：

```text
p = 1 / (1 + exp(-eta))
```

最后从 Bernoulli(p) 抽样得到 0/1 标签 y。

其中，x1 和 x3 会提高正类概率，x2 和 x4 会降低正类概率，x5 基本没有影响。

## 2. LinearRegression 与 LogisticRegression 对比

| threshold | TP | TN | FP | FN | accuracy | precision | recall | F1 | 模型 | ROC_AUC | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 63 | 82 | 12 | 23 | 0.8056 | 0.8400 | 0.7326 | 0.7826 | LinearRegression clipped to [0,1] | 0.8858 | 0.6292 |
| 0.5000 | 64 | 82 | 12 | 22 | 0.8111 | 0.8421 | 0.7442 | 0.7901 | LogisticRegression | 0.8852 | 0.4356 |

LinearRegression 的输出如果硬解释成概率，会出现不自然的问题。
在测试集中，LinearRegression 输出小于 0 的样本数为 14，大于 1 的样本数为 6。
但是概率应该被限制在 0 到 1 之间，所以线性回归的输出没有天然的概率意义。

LogisticRegression 通过 sigmoid 函数把线性得分映射到 0 到 1 之间，因此它的输出更容易解释成正类概率。

核心对比图保存位置：

```text
src/week15/figures/linear_vs_logistic_output.png
```

这张图的横轴是主要特征 x1，纵轴是模型输出。散点代表测试集真实 0/1 标签，一条线代表 LinearRegression 的输出，另一条线代表 LogisticRegression 的预测概率。图中想说明的是：线性回归输出可以超过概率范围，而逻辑回归输出始终在 0 到 1 之间。

## 3. 核心问题回答

### 3.1 LinearRegression 在二分类任务里最不自然的地方是什么？

最不自然的是它的输出不是概率。它可以小于 0，也可以大于 1，所以即使它可以通过阈值做分类，也不能自然解释为正类发生概率。

### 3.2 为什么逻辑回归的输出更容易解释成概率？

因为逻辑回归先计算线性得分，再通过 sigmoid 映射到 0 到 1 之间。这个范围正好符合概率的定义。

### 3.3 关键区别是能不能分类，还是输出是否有概率意义？

关键不是能不能分类。LinearRegression 加一个阈值也能给出 0/1 分类。真正的区别是输出有没有概率意义，以及训练目标是否适合二分类问题。