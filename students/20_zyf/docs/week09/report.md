# 📊 第九周实操报告：数据急救员与病态模型诊断

**提交日期**: 2026年5月12日  
**学生**: 20_zyf  
**课程**: 回归分析实操 (2026)

---

## 🎯 一、背景与目标

### 作业背景
现实世界的数据充满了"**外伤**"（缺失、异常）和"**内科病**"（多重共线性）。第九周的核心任务是：
- 编写**支持命令行参数的预处理脚本**（CLI 工具化）
- 开发**多重共线性诊断工具**
- 实现**5折交叉验证评估**
- **识别并讨论数据泄漏问题**

### 关键概念
- **虚拟变量陷阱**: One-Hot 编码必须 `drop_first=True`，否则矩阵奇异不可逆
- **Winsorization**: 将极端值限制到某个分位数（此处为99%）
- **VIF诊断**: 衡量特征间多重共线性的指标
- **数据泄漏**: 训练集信息"泄露"到测试集的问题

---

## 📂 二、代码实现结构

### 2.1 文件组织
```
students/20_zyf/
├── src/
│   ├── utils/
│   │   ├── models.py          # 核心OLS模型
│   │   └── diagnostics.py     # ✨ 新增：VIF计算工具
│   └── week09/
│       ├── data_prep.py       # ✨ 新增：数据清洗CLI脚本
│       └── evaluate.py        # 交叉验证与诊断入口
├── results/week09/
│   ├── clean_marketing.csv    # 清洗后的数据
│   └── diagnostics_report.md  # 诊断结果报告
└── docs/week09/
    └── report.md              # 本报告
```

### 2.2 核心模块功能

#### A. 数据预处理脚本 (`data_prep.py`)
**功能**: CLI工具，支持命令行传参

```bash
python data_prep.py --input homework/week09/data/dirty_marketing.csv \
                    --output students/20_zyf/results/week09/clean_marketing.csv
```

**处理流程**:
1. **缺失值填补**: 使用全局均值（本周要求；下周改进）
2. **异常值处理**: Winsorization at 99th percentile
3. **分类变量编码**: One-Hot with `drop_first=True` ✅ 防止虚拟变量陷阱
4. **数据验证**: 确保矩阵可逆

**处理结果**:
```
[STAGE 1] Handling missing values with mean imputation...
  - Imputed TV_Budget with mean: 180.3883

[STAGE 2] Applying Winsorization at 99th percentile...
  - Winsorized TV_Budget: 10 outliers capped at 297.8401
  - Winsorized Online_Video_Budget: 10 outliers capped at 269.8575
  - Winsorized Radio_Budget: 10 outliers capped at 99.7409
  - Winsorized Sales: 10 outliers capped at 1034.1014

[STAGE 3] Processing categorical variables with One-Hot encoding...
  - One-Hot encoding 'Region' (drop_first=True)...
  - Removed original categorical columns: ['Region']

Final data shape: (1000, 7)
```

#### B. 诊断工具箱 (`utils/diagnostics.py`)
**函数**: `calculate_vif(X: np.ndarray) -> list`

**VIF计算原理**:
$$VIF_j = \frac{1}{1 - R_j^2}$$

其中 $R_j^2$ 是将第 $j$ 个特征作为目标，其他特征作为预测器的OLS回归的 $R^2$。

**实现逻辑**:
- 对每个特征 $j$：
  1. 从 $X$ 中提取 $j$ 列作为 $y_j$
  2. 提取除 $j$ 外的所有列作为 $X_{\text{rest}}$
  3. 拟合 `AnalyticalOLS()` 模型
  4. 计算 $R_j^2$ 并反演为VIF
  5. 避免除零和负数问题

#### C. 评估脚本 (`week09/evaluate.py`)
**功能**: 综合诊断和交叉验证

---

## 📈 三、诊断结果分析

### 3.1 数据概览

| 指标 | 值 |
|------|-----|
| **样本数** | 1000 |
| **特征数** | 7 (3个预算+1个销售+3个地区虚拟变量) |
| **缺失值** | 50个（TV_Budget） |
| **异常值** | 各列10个@99分位数 |

**清洗后的列**:
- `TV_Budget`: 电视广告预算
- `Online_Video_Budget`: 在线视频预算
- `Radio_Budget`: 收音机广告预算
- `Sales`: **目标变量** 💰
- `Region_North`, `Region_South`, `Region_West`: 地区虚拟变量（East被drop_first丢弃）

### 3.2 多重共线性诊断结果

| 特征 | VIF值 | 评级 | 解释 |
|------|-------|------|------|
| **TV_Budget** | **16.76** | 🔴 严重 | TV和Online预算高度相关 |
| **Online_Video_Budget** | **17.32** | 🔴 严重 | 同上 |
| **Radio_Budget** | 1.00 | 🟢 正常 | 独立于其他预算特征 |
| **Region_North** | 1.31 | 🟢 正常 | 虚拟变量间无共线性 |
| **Region_South** | 1.31 | 🟢 正常 | - |
| **Region_West** | 1.31 | 🟢 正常 | - |

### 3.3 关键发现

#### ⚠️ 严重的多重共线性问题
```
WARNING: HIGH MULTICOLLINEARITY DETECTED!
- TV_Budget: VIF = 16.7642
- Online_Video_Budget: VIF = 17.3208
```

**业务解释**:
- 公司的 TV 和 Online 视频广告预算紧密关联
- 这在现实中是合理的（可能来自同一个营销部门的预算配置）
- **不影响预测能力** ✓，但影响系数解释 ✗

**系数解释问题示例**:
- 无法单独回答"TV预算增加$1,单独带来的销售增加"
- 因为TV预算变化通常伴随Online预算的变化

#### ✅ 数据有效性
- 虚拟变量编码正确（drop_first=True防止陷阱）
- 矩阵可逆，无奇异性问题
- 地区虚拟变量间无共线性

---

## 🔬 四、交叉验证结果

### 4.1 5折CV详细结果

| Fold | R² | 评论 |
|------|------|------|
| 1 | 0.990323 | ⭐⭐⭐⭐⭐ 优秀 |
| 2 | 0.990102 | ⭐⭐⭐⭐⭐ 优秀 |
| 3 | 0.990191 | ⭐⭐⭐⭐⭐ 优秀 |
| 4 | 0.989613 | ⭐⭐⭐⭐⭐ 优秀 |
| 5 | 0.988447 | ⭐⭐⭐⭐⭐ 优秀 |

### 4.2 统计汇总

```
┌─────────────────────────┐
│  Cross-Validation Stats │
├─────────────────────────┤
│ Mean R²:   0.989735     │
│ Std Dev:   0.000688     │
│ Min R²:    0.988447     │
│ Max R²:    0.990323     │
│ Range:     0.001876     │
└─────────────────────────┘
```

### 4.3 结果解读

✅ **优点**:
- 平均 $R^2 = 0.9897$ **非常高**（接近1）
- 5个fold间差异极小（std = 0.0007）
- 模型稳定性很好 → 低方差
- 在测试集上表现一致 → 低过拟合风险

❌ **隐患**（见第五部分）:
- 高 $R^2$ 部分源于**数据泄漏**，真实性能可能被高估

---

## 🚨 五、关键问题：数据泄漏分析

### 5.1 问题发现

**问题陈述**:
> 在 `data_prep.py` 中，我们使用**整个数据集的均值**进行缺失值填补。  
> 在 5 折交叉验证中，**测试集真的是"完全未见过"的陌生数据吗？**

**答案**: **否** 🚨

### 5.2 泄漏机制

#### 当前（错误）做法:
```python
# 第一步：在整个数据集上计算统计量
mean_TV = df['TV_Budget'].mean()  # 基于全部1000条数据

# 第二步：用这个均值填补缺失值
df_clean = df.fillna(mean_TV)

# 第三步：然后进行交叉验证
for fold in cv.splits():
    X_train, X_test = split(df_clean)  # ❌ 测试集已被污染！
```

#### 为什么有问题?
1. **统计量计算**: 
   - 使用了全部 1000 条数据（包含测试集）
   - 使用了**测试集的缺失值信息**
   - 虽然没有直接看到测试数据，但统计量"泄露"了信息

2. **测试集污染**:
   - 测试集中的缺失值被均值填补
   - 这个均值来自"未来数据"（测试集自己）
   - 降低了填补值的随机性和噪声

3. **性能高估**:
   - 测试集"人为变得更有规律"
   - 模型在这样的"干净"数据上拟合更好
   - 真实环境中的数据可能更混乱，模型性能下降

### 5.3 泄漏的量化影响

虽然我们没有直接对比，但可以推测：

- **实际 $R^2$ 可能**: 0.95 ~ 0.98（估计下降1~5%）
- **真实泛化误差**: 可能略高于报告的0.000688

### 5.4 正确做法（Week 10）

```python
# ✅ 正确的流程：fit on train, apply to test
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 关键：只在训练数据上计算统计量
    mean_train = X_train.mean(axis=0)
    
    # 用训练均值填补训练数据
    X_train_clean = fillna(X_train, mean_train)
    
    # 用同一个均值填补测试数据（不重新计算！）
    X_test_clean = fillna(X_test, mean_train)
    
    # 拟合和评估
    model.fit(X_train_clean, y_train)
    score = model.score(X_test_clean, y_test)
```

**关键原则**:
> **⚡ 所有数据预处理统计量必须从训练集计算，然后应用到测试集**

这防止了测试集"提前知道"训练集的特征，保证了交叉验证的真实性。

---

## 📋 六、虚拟变量陷阱回顾

### 问题背景
原始 `Region` 列有4个类别：East, North, South, West

### ❌ 错误的One-Hot编码:
```python
pd.get_dummies(df['Region'], drop_first=False)
# 输出: East_0/1, North_0/1, South_0/1, West_0/1 (4列)
# 问题: 完全共线性 → X'X不可逆
```

矩阵会有线性相关的列:
$$[\text{East}, \text{North}, \text{South}, \text{West}] \cdot (1,1,1,1)^T = 1$$
（每行的4个虚拟变量之和恒为1）

### ✅ 正确的做法:
```python
pd.get_dummies(df['Region'], drop_first=True)
# 输出: North_0/1, South_0/1, West_0/1 (3列)
# 原因: East变成"参考类别"，隐含在截距项中
```

现在列线性独立，$(X^TX)$ 可逆 ✓

### 验证结果
```
Final columns: ['TV_Budget', 'Online_Video_Budget', 'Radio_Budget', 
                'Sales', 'Region_North', 'Region_South', 'Region_West']
```
✅ 只有3个虚拟变量（East被丢弃）→ 矩阵可逆

---

## ✅ 七、红线检查与验收

### 红线1: ❌ Hardcode数据路径
```python
# ❌ 错误
df = pd.read_csv('homework/week09/data/dirty_marketing.csv')

# ✅ 正确
python data_prep.py --input homework/week09/data/dirty_marketing.csv \
                    --output students/20_zyf/results/week09/clean_marketing.csv
```
**状态**: ✅ 通过 - 使用argparse完全参数化

### 红线2: ❌ 奇异矩阵错误（虚拟变量陷阱）
```
Singular Matrix Error!
This indicates a Dummy Variable Trap violation.
```
**状态**: ✅ 通过 - drop_first=True防止陷阱，矩阵可逆

---

## 🎓 八、关键学习成果

### 技能获得
- ✅ 编写**CLI数据预处理工具**（工程实践）
- ✅ 实现**VIF诊断系统**（统计诊断）
- ✅ 进行**5折交叉验证**（模型评估）
- ✅ 识别**数据泄漏问题**（科学严谨性）

### 概念理解
- **多重共线性**: 特征间高度相关 → VIF > 10
- **虚拟变量陷阱**: One-Hot编码需drop_first → 否则矩阵奇异
- **数据泄漏**: 统计量必须从训练集计算 → 测试集独立性
- **模型诊断**: VIF检查 + 交叉验证相结合

### 代码工程
- 模块化设计：utils/diagnostics.py, week09/data_prep.py
- CLI工具化：argparse参数解析
- 日志报告：markdown格式输出
- 错误处理：dtype转换、路径创建

---

## 📊 九、总结与展望

### 本周成就
| 任务 | 状态 | 备注 |
|------|------|------|
| 数据预处理CLI | ✅ | 支持命令行参数 |
| 虚拟变量处理 | ✅ | drop_first=True |
| VIF诊断工具 | ✅ | 所有特征计算完成 |
| 5折交叉验证 | ✅ | 平均R²=0.9897 |
| 多重共线性检测 | ✅ | TV/Online预算VIF>10 |
| 数据泄漏识别 | ✅ | 问题已指出，下周改进 |

### 下周改进方向（Week 10）
- **防止数据泄漏**: 在CV内部分别fit预处理器
- **更先进的填补**: KNN填补、迭代填补
- **特征工程**: 降维、特征选择解决共线性
- **模型诊断**: 残差分析、异方差检验

### 代码质量
- **可复现性**: ✅ 完整的种子和参数控制
- **可维护性**: ✅ 模块化，有清晰的函数接口
- **可扩展性**: ✅ 易添加新的诊断指标

---

## 📁 十、交付物清单

```
students/20_zyf/
├── src/
│   ├── utils/
│   │   ├── models.py              (维持，无改动)
│   │   ├── diagnostics.py         ✨ NEW
│   │   └── __init__.py            (已更新)
│   └── week09/
│       ├── data_prep.py           ✨ NEW
│       └── evaluate.py            ✨ NEW (重写为markdown输出)
├── results/week09/
│   ├── clean_marketing.csv        (自动生成)
│   └── diagnostics_report.md      (自动生成，包含完整诊断)
└── docs/week09/
    └── report.md                  ✨ THIS FILE
```

### 运行指令
```bash
# 步骤1: 生成清洁数据
python students/20_zyf/src/week09/data_prep.py \
    --input homework/week09/data/dirty_marketing.csv \
    --output students/20_zyf/results/week09/clean_marketing.csv

# 步骤2: 运行诊断和交叉验证（自动调用步骤1）
python students/20_zyf/src/week09/evaluate.py

# 输出文件:
# - students/20_zyf/results/week09/diagnostics_report.md
# - students/20_zyf/results/week09/clean_marketing.csv
```

---

## 📚 参考资源

### VIF参考
- $VIF_j = \frac{1}{1-R_j^2}$ 其中 $R_j^2$ 来自 $X_j \sim X_{\text{rest}}$ 的回归
- 一般规则：VIF > 5-10表示问题，VIF > 10为严重

### 数据泄漏参考
- Leakage发生在: 预处理统计量在整个数据集上计算
- 解决方案: 所有fit()操作只在训练集上执行

### 虚拟变量陷阱参考
- K个类别 → K-1个虚拟变量（drop_first=True）
- 否则列线性相关，$(X^TX)$ 行列式为0，不可逆

--
