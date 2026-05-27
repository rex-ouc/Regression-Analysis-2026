# 🏆 里程碑项目2：管道与无泄露泛化报告

**学生**: 20_zyf  
**日期**: 2026-05-13  
**课程**: 回归分析 2026

## 🎯 背景与目标

本次作业聚焦于数据泄露问题，通过实现自定义转换器和无泄露交叉验证流水线，处理真实业务数据中的缺失值和异常值。目标是构建工业级流水线，确保模型评估的纯洁性和可靠性。

## 📝 阶段一：扩充算法工具箱

### Task 1: 评估指标库 (`utils/metrics.py`)

实现了三个核心评估指标：
- `calculate_rmse(y_true, y_pred)`: 计算均方根误差
- `calculate_mae(y_true, y_pred)`: 计算平均绝对误差
- `calculate_mape(y_true, y_pred)`: 计算平均绝对百分比误差，处理分母为0的异常情况

### Task 2: 打造转换器 API (`utils/transformers.py`)

编写了 `CustomStandardScaler` 类，遵循Transformer接口：
- `fit(X)`: 计算并保存均值和标准差
- `transform(X)`: 使用保存参数进行标准化
- `fit_transform(X)`: 合并fit和transform

## 💻 阶段二：真实业务试炼与交叉验证

### Task 3: 危险的诱惑 —— 制造数据泄露

在 `bad_cross_validation()` 中：
- 对全量数据进行全局预处理（标准化和缺失值填充）
- 使用5折交叉验证
- 平均RMSE: 655.57

### Task 4: 坚不可摧的护城河 —— 手撸防泄漏流水线

在 `good_cross_validation()` 中：
- 在每个折叠内部单独fit Scaler（仅用训练数据）
- 使用训练参数transform验证数据
- 平均RMSE: 649.40

### 思考题：对比Task 3和Task 4的RMSE

Task 3的RMSE看似更好（655.57 vs 649.40），但这是致命的假象。因为全局预处理泄露了测试集信息，导致评估过于乐观。Task 4的更高误差反映真实性能，避免了生产中的灾难性失败。

## 📦 阶段三：自动化 I/O 与汇报

### Task 5: 自动化制品管理

- 单一入口: `uv run src/milestone2/main.py`
- 动态清理results目录
- 输出: `evaluation_comparison.md` 和可选图表

### 输出物料

评估指标比较见 `results/evaluation_comparison.md`：
- RMSE差异: +6.17 (错误CV vs 正确CV)
- MAE差异: +5.52
- MAPE差异: +0.84%

## 🎤 课堂答辩

### CTO 视角 (技术复盘)

`CustomStandardScaler` 结构：
- fit: 保存 `self.mean_` 和 `self.std_`
- transform: `(X - mean) / std`
- 在 `good_cross_validation` 中，每个折叠重新fit，确保数据隔离。

### CMO 视角 (业务解读)

基于MAE $647.92，模型上线后每日广告预算预测误差约$648。应报告Task 4的“差成绩”而非Task 3的“好成绩”，因为后者不可靠，会导致生产失败和信任损失。

## 🚨 验收红线

- 无绝对路径，使用相对路径
- 无二次泄露，确保transform仅用训练参数

**结论**: 正确的数据处理比复杂算法更重要，无泄露评估是模型可靠性的基石。