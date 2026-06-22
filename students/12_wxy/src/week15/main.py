import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent  
sys.path.insert(0, str(project_root))

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 导入原有工具 + 上面新增二分类工具
from src.utils.transformers import (
    StandardScaler,
    generate_synthetic_binary_taskA,
    generate_highdim_collinear_binary_taskD
)
from src.utils.models import (
    CustomOLS, train_logistic_base, cv_tune_logistic
)
from src.utils.metrics import calc_binary_metrics, scan_all_thresholds
from src.utils.diagnostics import (
    plot_ols_vs_logistic_single_feature,
    plot_mse_logloss_curve,
    plot_threshold_metric_tradeoff,
    plot_l1_l2_coef_compare
)

# 文件夹初始化
os.makedirs("src/week15/data", exist_ok=True)
os.makedirs("src/week15/results", exist_ok=True)

def task_a():
    print("==== Task A：模拟二分类数据 OLS vs 逻辑回归 ====")
    df, true_beta = generate_synthetic_binary_taskA(n_samples=500, n_feats=5)
    df.to_csv("src/week15/data/synthetic_binary.csv", index=False)
    X = df.drop(["y", "true_prob"], axis=1)
    y = df["y"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # 1. OLS 自实现 CustomOLS
    ols = CustomOLS(fit_intercept=True, alpha=0.0)
    ols.fit(Xtr, y_train)
    ols_pred = np.array(ols.predict(Xte))  

# 2. 逻辑回归
    lr_base = train_logistic_base(Xtr, y_train, penalty="l2")
    lr_prob = np.array(lr_base.predict_proba(Xte)[:,1])  


    # 绘图：单特征对比图
    feat0_test = X_test["feat_0"].values
    plot_ols_vs_logistic_single_feature(
    x_arr=feat0_test,
    y_arr=np.array(y_test),          
    ols_pred_arr=np.array(ols_pred), 
    lr_prob_arr=np.array(lr_prob),   
    save_path="src/week15/results/ols_lr_compare.png"
)
    # 指标计算
    ols_clamp_prob = np.clip(ols_pred, 0, 1)
    met_ols = calc_binary_metrics(y_test, ols_clamp_prob, threshold=0.5)
    met_lr = calc_binary_metrics(y_test, lr_prob, threshold=0.5)

    # 生成 synthetic_report.md
    md = "# Task A 模拟二分类数据实验报告\n\n"
    md += f"- 样本量：{len(df)}，特征数量：5\n"
    md += f"- 真实线性系数 β = {true_beta.tolist()}\n"
    md += "- DGP流程：线性得分η = Xβ → Sigmoid映射概率p → 伯努利分布采样0/1标签y\n\n"
    md += "## OLS 与逻辑回归测试集指标对比\n"
    compare_df = pd.DataFrame([met_ols, met_lr], index=["CustomOLS", "LogisticRegression"])
    md += compare_df.to_markdown()
    md += """
## 核心问题回答
1. OLS缺陷：预测输出无边界，会小于0或大于1，无法解释为概率；损失MSE基于连续正态假设，不匹配0-1离散标签。
2. 逻辑回归经Sigmoid压缩至[0,1]区间，数学上等价伯努利分布的事件发生概率。
3. 二者本质差异不是能否划分类别，而是输出值是否具备严谨概率统计学含义。
"""
    with open("src/week15/results/synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    return Xte, y_test, lr_prob

def task_b():
    print("==== Task B：伯努利分布与 LogLoss 损失 ====")
    plot_mse_logloss_curve(save_path="src/week15/results/loss_compare.png")
    md = "# Task B 伯努利似然与对数损失\n\n"
    md += r"""
### 1. 伯努利分布
$$ Y \sim Bernoulli(p) $$
Y仅可取0/1，p代表y=1的发生概率，完美匹配二分类标签分布。

### 2. 单样本似然函数
$$ L(p;y) = p^y (1-p)^{1-y} $$
y=1时保留p项；y=0保留(1-p)项，衡量当前概率p生成观测标签y的可能性。

### 3. 负对数似然 LogLoss
$$ \ell(p,y) = -y\log p - (1-y)\log(1-p) $$
对似然取对数并取负，最大化似然等价最小化该损失。

## 实验结论
1. 模型高度自信预测错误时，LogLoss惩罚远大于MSE，约束模型不能输出极端错误概率。
2. LogLoss不是人为定义，是伯努利分布极大似然估计的直接数学推导结果。
3. 若输出解释为概率，LogLoss衡量概率分布拟合程度，MSE无概率分布理论支撑。
"""
    with open("src/week15/results/threshold_report.md", "w", encoding="utf-8") as f:
        f.write(md)

def task_c(X_test, y_test, lr_prob):
    print("==== Task C：阈值扫描与分类指标权衡 ====")
    base_metrics = calc_binary_metrics(y_test, lr_prob, threshold=0.5)
    scan_df = scan_all_thresholds(y_test, lr_prob, 0.1, 0.9, 0.1)
    plot_threshold_metric_tradeoff(scan_df, save_path="src/week15/results/threshold_trade.png")

    md = "\n\n## Task C 阈值权衡实验\n## 0.5阈值基础指标\n"
    md += pd.DataFrame([base_metrics]).to_markdown(index=False)
    md += "\n## 全阈值扫描结果（节选）\n"
    md += scan_df[["Threshold","Accuracy","Precision","Recall","F1"]].to_markdown(index=False)
    md += """
## 业务场景：疾病初筛
业务优先关注 Recall（召回率）。
原因：漏诊(FN)会延误患者治疗，代价极高；宁可增加复检样本，也不能遗漏患病个体。
推荐选择偏低阈值0.2~0.3，牺牲精确率换取更高召回。

## 图表说明
横轴：分类判定阈值；纵轴：各分类指标取值。
阈值升高 → Precision上升、Recall下降，二者天然存在权衡；样本不均衡场景仅看Accuracy会严重误导判断。
"""
    with open("src/week15/results/threshold_report.md", "a", encoding="utf-8") as f:
        f.write(md)

def task_d():
    print("==== Task D：L1 vs L2 正则逻辑回归 ====")
    df = generate_highdim_collinear_binary_taskD(n_samples=600, n_feats=22)
    X = df.drop("y", axis=1)
    y = df["y"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # 交叉验证调参 L1 / L2
    l1_model, l1_best = cv_tune_logistic(Xtr, y_train, penalty="l1")
    l2_model, l2_best = cv_tune_logistic(Xtr, y_train, penalty="l2")

    l1_prob = l1_model.predict_proba(Xte)[:,1]
    l2_prob = l2_model.predict_proba(Xte)[:,1]
    met_l1 = calc_binary_metrics(y_test, l1_prob)
    met_l2 = calc_binary_metrics(y_test, l2_prob)

    # 系数稀疏度统计
    l1_coef = l1_model.coef_[0]
    l2_coef = l2_model.coef_[0]
    non_zero_l1 = np.sum(np.abs(l1_coef) > 1e-6)
    non_zero_l2 = np.sum(np.abs(l2_coef) > 1e-6)

    plot_l1_l2_coef_compare(l1_coef, l2_coef, save_path="src/week15/results/l1_l2_coef.png")

    # 输出报告
    md = "# Task D L1/L2正则逻辑回归对比\n\n"
    res_table = pd.DataFrame([met_l1, met_l2], index=["L1正则","L2正则"])
    res_table["非零特征数量"] = [non_zero_l1, non_zero_l2]
    md += res_table.to_markdown()
    md += f"\n最优正则超参：L1 C={l1_best['C']}, L2 C={l2_best['C']}\n\n"
    md += """
## 对比问题回答
1. 预测性能（AUC/Accuracy）差距通常很小，整体表现接近。
2. L1正则模型更稀疏，大量特征系数压缩至0。
3. 需要精简变量、筛选关键特征时选择L1正则。
4. 业务追求预测稳定、不需要变量筛选，优先L2，L2平滑缩小所有权重无突变。
"""
    with open("src/week15/results/regularization_report.md", "w", encoding="utf-8") as f:
        f.write(md)

def task_summary():
    print("==== Task F：综合总结 ====")
    summary_text = """# Week15 逻辑回归二分类综合总结
1. 逻辑回归不是简单线性回归加Sigmoid：OLS基于正态连续值MSE损失；逻辑回归基于伯努利0/1分布，损失来自极大似然估计，建模底层完全不同。
2. 完整逻辑链：Sigmoid将线性得分映射至0~1概率 → 伯努利分布描述二分类标签生成 → 最大化似然推导出LogLoss损失。
3. 不能仅依靠Accuracy：样本不平衡时准确率具备欺骗性，必须结合Precision/Recall/F1/AUC与业务代价权衡阈值。
4. L1适合高维场景特征筛选、降噪；L2适合稳定预测、防止系数剧烈波动。
5. 逻辑回归优势：输出可解释概率、系数直观反映特征正负影响、训练稳定，是工业界分类任务强基线模型。
"""
    with open("src/week15/results/summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

if __name__ == "__main__":
    X_test_out, y_test_out, lr_prob_out = task_a()
    task_b()
    task_c(X_test_out, y_test_out, lr_prob_out)
    task_d()
    task_summary()
    print("全部实验完成，图片与Markdown报告输出至 src/week15/results/")
