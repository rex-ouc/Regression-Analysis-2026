# Week 04 Assignment: The Tale of Two Solvers

**姓名**：张梦宇  
**学号**：21251113008  
**日期**：2026年3月30日  

## 实验背景
在多元线性回归中，解析解 $\hat{\beta} = (X^T X)^{-1} X^T y$ 在数学上简洁优美，但其计算复杂度为 $O(P^3)$，当特征维度 $P$ 很高时（如基因数据、高频交易），会面临内存爆炸和算力瓶颈。现代工业界常采用梯度下降类算法迭代求解。本次实验旨在对比解析解与梯度下降在低维和高维场景下的性能表现。

## 实验设计
- **数据生成**：N=10,000 样本，P 分别为 10（低维）和 2000（高维）。特征矩阵 X 由标准正态分布生成，真实系数 $\beta$ 在 $[-2,2]$ 均匀分布，噪声 $\epsilon \sim N(0,1)$。
- **求解器**：
  1. `AnalyticalSolver (custom)`：使用正规方程，通过 `np.linalg.solve` 数值稳定求解。
  2. `GradientDescentSolver (custom)`：全批量梯度下降，学习率 0.01，最大迭代 500 次。
  3. `statsmodels.OLS`：统计库的普通最小二乘。
  4. `sklearn.LinearRegression`：机器学习库的解析解。
  5. `sklearn.SGDRegressor`：随机梯度下降（默认小批量），最大迭代 500 次。
- **评估指标**：运行时间（秒）和均方误差（MSE）。

## 实验结果

### 低维场景（N=10000, P=10）
| 求解器                      | 时间 (秒) | MSE   |
|----------------------------|----------|-------|
| AnalyticalSolver (custom)  | 0.0011   | 1.0039|
| GradientDescentSolver (custom) | 0.0460 | 1.0039 |
| statsmodels.OLS             | 0.0038   | 1.0039 |
| sklearn.LinearRegression    | 0.0176   | 1.0039 |
| sklearn.SGDRegressor        | 0.0035   | 1.0067 |

在低维下，所有求解器都能快速完成，MSE 几乎一致（SGD 略高，因未完全收敛）。解析解明显快于自定义梯度下降（0.0011 vs 0.0460）。

### 高维场景（N=10000, P=2000）
| 求解器                      | 时间 (秒) | MSE   |
|----------------------------|----------|-------|
| AnalyticalSolver (custom)  | 0.8373   | 0.8000|
| GradientDescentSolver (custom) | 20.4916 | 0.8553 |
| statsmodels.OLS             | 30.3733  | 0.8000 |
| sklearn.LinearRegression    | 13.7588  | 0.8000 |
| sklearn.SGDRegressor        | 2.3693   | 1.0903 |

在高维下，解析解（自定义、statsmodels、sklearn）耗时明显增加，其中 **statsmodels.OLS 最慢（30.37秒）**，**自定义梯度下降也极慢（20.49秒）**，而 **SGDRegressor 仅需 2.37 秒**，速度优势显著。

## 分析与结论
1. **解析解的瓶颈**：求解 $O(P^3)$ 的矩阵求逆或解线性方程组，当 $P=2000$ 时计算量巨大。`np.linalg.solve` 虽然优化良好，但依然需要处理 $2000 \times 2000$ 矩阵，耗时约 0.84 秒。statsmodels 额外计算了更多统计量（如 p值、标准误），因此更慢。
2. **全批量梯度下降**：自定义实现每次迭代遍历全部样本，更新一次需 $O(NP)$。500 次迭代导致总复杂度 $O(500 \cdot NP)$，在 $N=10000, P=2000$ 时非常耗时（20.49秒），且未充分收敛（MSE 0.8553 略高于解析解的 0.8000）。
3. **随机梯度下降（SGDRegressor）**：每次迭代只使用一个样本（或小批量），计算量大幅降低。虽然 500 次迭代后 MSE 稍高（1.0903），但时间仅 2.37 秒，体现了“在可接受精度下用时间换精度”的工程智慧。

## 思考题
> **在高维场景下，哪个 API 崩溃了或极其缓慢？为什么 SGDRegressor 能在极短时间内完成任务？**

- **最慢的 API**：`statsmodels.OLS` 耗时最长（30.37秒），因为它不仅求解参数，还计算了大量统计推断内容（如协方差矩阵、p值、置信区间），增加了额外开销。
- **SGDRegressor 的高效原因**：它使用**随机梯度下降**，每次迭代只处理一个样本（或小批量），而非全部样本。算法复杂度为 $O(\text{iterations} \cdot P)$，远低于解析解的 $O(P^3)$。通过适当选择迭代次数，可以在几秒内获得一个足够好的近似解，非常适合高维大规模数据场景。

## 代码与运行说明
- 所有代码位于 `students/08_zmy/src/week04/` 目录下。
- 依赖库：`numpy`, `pandas`, `statsmodels`, `scikit-learn`。
- 运行方式：激活虚拟环境后执行 `python src/week04/main.py`。

## 参考文献
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Scikit-learn: SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
- [Statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
