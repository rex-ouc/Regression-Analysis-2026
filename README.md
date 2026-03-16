# Regression-Analysis-2026

回归分析课程仓库，用于发布课程资料、布置作业、收集学生提交，并统一管理课程实践内容。

## 仓库用途

这个仓库主要承担四个功能：

- 发布课程作业与实践要求；
- 提供统一的学生目录结构；
- 收集学生通过 Pull Request 提交的课程内容；
- 保存课程参考资料与实验报告。

当前仓库更偏向教学组织与提交管理，不是一个单独的算法实现项目。

## 目录结构

```text
.
├── README.md
├── books/                  # 课程参考书与阅读材料
├── homework/               # 每周作业说明
└── students/               # 学生个人目录与提交内容
	├── template/           # 推荐模板
	└── <your_id>/          # 学生个人目录
```

### 目录说明

- `books/`：放置课程参考书、讲义或补充阅读材料。
- `homework/`：按周维护作业要求，例如 `week01.md`、`week02.md`。
- `students/template/`：提供建议的目录组织方式，便于统一提交格式。
- `students/<your_id>/`：每位同学在自己的目录中完成介绍、代码、报告和实验结果。

## 建议环境

建议使用以下工具完成课程实践：

- Git 与 GitHub；
- Terminal（macOS）或 WSL（Windows）；
- `uv` 进行 Python 环境与依赖管理；
- Jupyter Notebook 或 VS Code。

如果你需要完成 Python 作业，建议至少准备：

- Python 3.11 或更高版本；
- `numpy`；
- `scikit-learn`；
- `statsmodels`；
- `jupyter`。

一个可参考的初始化流程：

```bash
uv venv
source .venv/bin/activate
uv pip install numpy scikit-learn statsmodels jupyter
```

## 课程作业流程

### 第一步：Fork 仓库

1. 在 GitHub 上 Fork 本课程仓库到你自己的账号。
2. 将你 Fork 后的仓库克隆到本地。
3. 在本地完成作业后，推送到自己的远端仓库。
4. 向课程仓库发起 Pull Request。

### 第二步：创建个人目录

请在 `students/` 下使用自己的标识创建目录，例如：

```text
students/ZMY/
```

建议参考 `students/template/` 的结构组织内容。

### 第三步：完成每周作业

每周作业说明位于 `homework/` 目录。

- 第一周：完成环境配置、Fork 仓库，并提交自我介绍。
- 第二周：完成一元回归分析实验，包括手工推导/计算与工具库比较。

后续作业会继续按周补充到 `homework/` 目录。

## 学生目录建议

建议每位同学的目录按下面方式组织：

```text
students/<your_id>/
├── introduction.txt        # 自我介绍
├── docs/                   # 报告、说明文档
└── week02_simple_regression/
	├── main.py             # 代码实现（示例）
	├── notebook.ipynb      # 可选：notebook 版本
	└── result.md           # 可选：实验结论
```

不要求所有文件都必须出现，但建议结构清晰、命名稳定。

## 提交规范

请尽量遵守以下约定：

- 一个 PR 对应一次明确的作业提交；
- 提交说明写清楚周次和内容；
- 不修改其他同学的目录；
- 不随意改动 `homework/` 中的作业要求；
- 如果补交或修改作业，请在 PR 描述中注明原因。

推荐的提交信息示例：

```text
week02: submit simple regression assignment
```

## 第二周作业提示

根据 `homework/week02.md`，第二周建议至少完成以下内容：

- 生成一组满足简单线性回归模型的数据；
- 手工计算参数估计值；
- 比较手工结果、`sklearn` 与 `statsmodels` 的结果；
- 说明估计值与真实值之间的偏差；
- 选做：撰写实验报告。

## 对课程维护者的建议

如果后续要继续完善这个仓库，建议优先补充：

- 每周作业对应的最小示例代码；
- 统一的报告模板；
- 环境依赖文件，例如 `pyproject.toml`；
- PR 模板或作业提交检查清单。

## 说明

本仓库默认假设学生以 GitHub PR 的形式提交作业。如果课程安排有变化，请以课程通知为准。