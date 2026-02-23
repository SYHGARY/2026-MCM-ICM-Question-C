import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 核心数据（含95%置信区间上下限，基于模型SE计算）
labels = ["Judges Score", "Vote (linear)", "Vote (softmax)"]
coef = [-1.5507, -0.5020, -1.1503]
# 置信区间计算：coef ± 1.96*SE（95%置信水平）
ci_lower = [-1.7978, -0.6080, -1.5996]  # 下界 = coef - 1.96*SE
ci_upper = [-1.3036, -0.3961, -0.7010]  # 上界 = coef + 1.96*SE
colors = ["steelblue", "salmon", "lightcoral"]

# 设置图表样式（符合MCM O奖规范）
plt.rcParams.update({"font.size": 11, "font.family": "Arial", "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(6, 4.5))

# 绘制带置信区间的柱状图
bars = ax.bar(labels, coef, color=colors, edgecolor="black", linewidth=0.8, alpha=0.8)
ax.errorbar(labels, coef, yerr=[np.array(coef)-np.array(ci_lower), np.array(ci_upper)-np.array(coef)],
            fmt="none", color="black", capsize=5, capthick=0.8, elinewidth=0.8)

# 基准线与标签设置
ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax.set_ylabel("Effect of Age (per 1 SD)", fontsize=12, fontweight="bold")
ax.set_title("Age Effect Comparison: Judges Score vs Vote Mechanisms", 
             fontsize=13, fontweight="bold", pad=15)

# 数值标注（系数+显著性）
for i, (c, lower, upper) in enumerate(zip(coef, ci_lower, ci_upper)):
    ax.text(i, c - 0.15, f"Coef: {c:.2f}", ha="center", va="top", fontsize=10)
    ax.text(i, c + 0.05 if c > 0 else c - 0.05, f"[{lower:.2f}, {upper:.2f}]", 
            ha="center", va="bottom" if c > 0 else "top", fontsize=9, color="darkred")
    ax.text(i, c - 0.3, "p<0.0001", ha="center", va="top", fontsize=9, style="italic")

# 图表优化
ax.set_ylim(-2.2, 0.5)  # 预留足够标注空间
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 调整布局
plt.tight_layout()
plt.savefig("age_effect_comparison_with_ci.png", dpi=300, bbox_inches="tight")
plt.show()
