### 低维场景 (N=10,000, P=10)

| Solver | Time (s) | MSE |
|--------|--------|-----|
| AnalyticalSolver |  0.2033s | 0.009 |
| GradientDescentSolver | 0.0345s | 0.009 |
| LinearRegression |  0.0304s | 0.009 |
| SGDRegressor | 0.0744s | 0.009 |
| Statsmodels OLS | 0.1850s | 0.009 |

---

### 高维场景 (N=10,000, P=2,000)

| Solver | Time (s) | MSE |
|--------|--------|-----|
| AnalyticalSolver | ❗非常慢/崩溃 8.9197s | 0.008233 |
| GradientDescentSolver | 7.7891s | 0.054041 |
| LinearRegression | 16.3559s | 0.008233 |
| SGDRegressor | 🚀最快1.0890s | 0.017673 |
| Statsmodels OLS | ❗崩溃/内存爆炸16.9214s | 0.008233 |

---

## 思考题

### 1️⃣ 哪个 API 崩溃或极慢？

在高维情况下：

- Statsmodels OLS 极慢甚至崩溃
- AnalyticalSolver 也明显变慢

原因：

- 都依赖矩阵运算 `(X^T X)`，复杂度为 O(P³)
- 当 P=2000 时：
  - 计算量巨大
  - 内存消耗爆炸

---

### 2️⃣ 为什么 SGDRegressor 很快？

SGD（随机梯度下降）的优势：

- 不需要矩阵求逆
- 每次只用一小部分数据（甚至单样本）
- 复杂度约为 O(NP)

核心原因：

**避免了 O(P³)，改为线性复杂度 O(NP)**

---

### 3️⃣ 总结

| 方法 | 复杂度 | 适用场景 |
|------|--------|--------|
| 解析解 | O(P³) | 低维 |
| 梯度下降 | O(NP) | 高维 |
| SGD | O(NP)（更快） | 超高维 / 大数据 |

---

## 最终结论

工业界更倾向使用 SGD，因为：

- 可扩展性强
- 内存友好
- 适用于大规模数据