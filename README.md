# Regression-Analysis-2026

回归分析课程仓库，用于统一发布课程资料、布置作业、收集学生提交，并规范课程实践的提交方式。

## README 导航

- [仓库用途](#仓库用途)
- [目录组织建议](#目录组织建议)
- [Git 协同工作流](#git-协同工作流)
- [Python 与 uv 环境管理](#python-与-uv-环境管理)
- [使用 uv 增加 Jupyter Notebook](#使用-uv-增加-jupyter-notebook)
- [Jupyter Notebook 使用指南](#jupyter-notebook-使用指南)
- [每周作业提交建议](#每周作业提交建议)

## 仓库用途

这个仓库主要用于：

1. 发布课程资料；
2. 布置与管理每周作业；
3. 收集学生代码与报告；
4. 统一课程实践中的 Git 与 Python 环境使用方式。

## 目录组织建议

请参考课程提供的模板目录组织自己的作业内容。

建议你仿照 `template/` 的结构，在个人目录中使用如下命名方式：

- `<学号最后2位>_<姓名拼音小写>`

这样便于教师和助教统一检查、汇总和批改。

---

## Git 协同工作流

下面是一套推荐的 Git 操作流程，用于从课程主仓库同步更新、完成作业并提交 PR。

### 0. 初始化本地仓库

先克隆你自己的 fork 仓库，再添加课程主仓库为 `upstream`：

```/dev/null/bash#L1-8
git clone <fork_repo_addr>
cd <repo_name>

git remote add upstream git@github.com:rex-ouc/Regression-Analysis-2026.git

git remote -v
```

说明：

- `origin` 指向你自己的 fork 仓库；
- `upstream` 指向课程主仓库；
- `git remote -v` 用于检查远程仓库是否配置正确。

### 1. 从主仓库同步更新

开始新一周作业前，先同步主仓库内容到本地：

```/dev/null/bash#L1-5
git switch main
git fetch upstream
git diff main upstream/main
git merge upstream/main
```

说明：

- `git switch main`：切换到本地主分支；
- `git fetch upstream`：获取主仓库更新，但暂不合并；
- `git diff main upstream/main`：查看本地 `main` 与主仓库 `main` 的差异；
- `git merge upstream/main`：将主仓库更新合并到本地 `main`。

### 2. 为本周作业创建分支

建议每周作业使用独立分支完成，例如：

```/dev/null/bash#L1-1
git branch week<n>-hw
```

你也可以直接创建并切换过去：

```/dev/null/bash#L1-1
git switch -c week<n>-hw
```

这样可以把不同周次的修改隔离开，避免直接在 `main` 上开发。

### 3. 完成作业并提交到 fork 仓库

完成代码与报告后，执行：

```/dev/null/bash#L1-4
git add .
git commit -m "..."
git push origin week<n>-hw
```

然后在 GitHub 上基于该分支发起 PR。

### 4. PR 合并后清理分支

确认 PR 已经通过并合并后，再删除本地分支：

```/dev/null/bash#L1-1
git branch -d week<n>-hw
```

请注意：只有在确认不再需要该分支时再删除。

### 5. 可视化工具建议

推荐你在 VS Code 中安装 `Git Graph` 插件，用于更直观地查看分支、提交历史和合并关系。

---

## Python 与 uv 环境管理

课程实践推荐使用 `uv` 管理 Python 项目、虚拟环境与依赖。

### 1. 初始化项目

进入你自己的作业目录后，执行：

```/dev/null/bash#L1-4
cd <your_dir>
uv init
uv venv
uv add numpy statsmodels scikit-learn
```

说明：

- `uv init`：初始化 Python 项目；
- `uv venv`：创建虚拟环境；
- `uv add ...`：安装项目依赖，并写入项目配置。

如果你需要激活虚拟环境，通常可以使用：

```/dev/null/bash#L1-2
source .venv/bin/activate
python --version
```

---

## 使用 uv 增加 Jupyter Notebook

如果你希望在课程作业中使用 Notebook 进行实验、可视化或推导记录，建议也通过 `uv` 安装相关依赖。

### 1. 安装 Jupyter Notebook

在项目目录下执行：

```/dev/null/bash#L1-1
uv add notebook
```

如果你还没有安装常用的数据分析依赖，也可以一起安装：

```/dev/null/bash#L1-1
uv add notebook numpy pandas matplotlib seaborn statsmodels scikit-learn
```

### 2. 启动 Notebook

推荐使用以下方式启动：

```/dev/null/bash#L1-1
uv run jupyter notebook
```

这会使用当前项目的虚拟环境启动 Jupyter Notebook，避免系统 Python 与项目环境不一致。

### 3. 首次启动后会发生什么

启动成功后，终端通常会输出一个本地访问地址，例如：

```/dev/null/text#L1-1
http://localhost:8888/tree?token=...
```

你可以：

- 直接点击终端中的链接；
- 或复制到浏览器中打开。

---

## Jupyter Notebook 使用指南

下面是一套适合课程作业的基础使用流程。

### 1. 新建 Notebook

在浏览器打开 Jupyter 页面后：

1. 进入你的项目目录；
2. 点击右上角 `New`；
3. 选择 `Python 3`（或当前项目对应的内核）；
4. 创建一个新的 `.ipynb` 文件。

建议把 Notebook 文件放在有明确含义的位置，例如：

- `homework/week01/notebooks/`：按周次管理，存放实验记录。

### 2. 基本单元操作

Notebook 由一个个单元（cell）组成，常用操作包括：

- `Shift + Enter`：运行当前单元并跳到下一个单元；
- `Ctrl + Enter`：运行当前单元但停留在当前单元；
- `Alt + Enter`：运行当前单元并在下方插入新单元。

常见单元类型：

- `Code`：编写和运行 Python 代码；
- `Markdown`：编写说明文字、公式和结论。

建议你把“代码、输出、解释”写在相邻单元中，方便展示分析过程。

### 3. 在 Notebook 中导入项目依赖

只要你是通过 `uv run jupyter notebook` 启动的，通常就可以直接使用项目环境中的依赖，例如：

```/dev/null/python#L1-4
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
```

如果出现缺少依赖的报错，返回终端补充安装即可，例如：

```/dev/null/bash#L1-1
uv add pandas matplotlib seaborn
```

### 4. 保存与导出

Notebook 会自动保存，但你仍然应当养成手动保存的习惯。

可使用页面菜单：

- `File -> Save and Checkpoint`：手动保存；
- `File -> Download as`：导出为 HTML、Markdown 等格式。

如果课程要求最终提交 Markdown 报告，建议你：

- 用 Notebook 做实验与过程记录；
- 再整理为结构更清晰的 `.md` 报告提交。

### 5. 关闭 Notebook

使用完成后，建议回到启动 Notebook 的终端，按：

```/dev/null/text#L1-1
Ctrl + C
```

然后根据提示确认关闭服务。


---

## 每周作业提交建议

### 提交内容

建议每周作业至少包含：

- Python 源代码；
- Markdown 格式报告；
- 清晰的程序运行入口，例如 `main.py`。

### 提交规范

- 优先提交可复现的代码与说明；
- 报告建议使用 `.md` 格式；
- 若使用 Notebook，建议同时整理出可阅读的结论性报告，而不只提交实验过程。

### 代码质量工具

推荐使用 `ruff` 进行代码格式化与静态检查：

```/dev/null/bash#L1-2
uvx ruff format src
uvx ruff check src
```


## 补充建议

- 每周开始前先同步主仓库；
- 每次作业使用独立分支；
- 环境依赖统一交给 `uv` 管理；
- Notebook 适合做探索性分析，但最终提交材料应尽量结构清晰、便于批改；
- 保持目录命名统一，有助于课程资料管理与自动化检查。
