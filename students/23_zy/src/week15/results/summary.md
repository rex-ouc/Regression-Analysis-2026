# Week15 Summary：逻辑回归、概率解释与阈值权衡

## 1. 为什么逻辑回归不是“线性回归后面接一个 sigmoid”这么简单？

逻辑回归不只是在线性回归输出后接 sigmoid。它背后对应的是二分类概率建模：目标变量服从 Bernoulli 分布，模型估计的是正类概率，训练目标来自 Bernoulli likelihood 的最大化。

## 2. sigmoid、Bernoulli likelihood、log loss 三者之间是什么关系？

sigmoid 把线性得分变成 0 到 1 之间的概率 p。Bernoulli likelihood 用这个 p 来描述观察到标签 y 的可能性。log loss 则是 Bernoulli likelihood 取负对数后得到的损失函数。

## 3. 为什么分类模型不能只看 accuracy？

因为 accuracy 只看整体预测对了多少，但不区分 FP 和 FN 的代价。在疾病初筛、信用违约、用户流失等场景中，不同错误的业务成本不同，所以必须结合 precision、recall、F1、ROC-AUC 等指标一起看。

## 4. L1 和 L2 逻辑回归分别更适合什么目标？

L1 更适合变量筛选，因为它能产生稀疏系数，把一些变量系数压缩为 0。L2 更适合稳定建模，因为它整体压缩系数，减少模型对单个变量的过度依赖。

## 5. 为什么逻辑回归仍然是很强的 baseline？

逻辑回归可以输出概率，模型结构简单，训练稳定，解释性强，还可以通过 L1/L2 正则化处理高维和共线性问题。如果业务方需要一个稳定、可解释、能输出概率的模型，逻辑回归仍然是一个很强的 baseline。