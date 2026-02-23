import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 全局设置（符合学术图表规范）
plt.rcParams.update({
    "font.size": 11,
    "font.family": "Arial",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# 1. 读取基础方差数据（JudgesScore + 线性投票）
df_var = pd.read_csv("variance_comparison.csv")
labels = ["Celebrity", "Partner", "Residual"]
x = np.arange(len(labels))
width = 0.28  # 适配三组数据的柱宽

# 提取基础数据
judge_vars = df_var[df_var["Model"]=="JudgesScore"][["Celebrity_Var", "Partner_Var", "Residual_Var"]].values.flatten()
fan_linear_vars = df_var[df_var["Model"]=="VoteLogitShare"][["Celebrity_Var", "Partner_Var", "Residual_Var"]].values.flatten()

# 补充softmax投票方差数据（来自补充信息）
fan_softmax_vars = np.array([1.148764e-08, 18.818528, 11.877892])

# 2. 创建图表（支持三组对比）
fig, ax = plt.subplots(figsize=(9, 5))

# 绘制三组柱状图
bars1 = ax.bar(x - width, judge_vars, width, label="Judges Score", color="#2E86AB", alpha=0.8, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x, fan_linear_vars, width, label="Fan Votes (Linear)", color="#A23B72", alpha=0.8, edgecolor="black", linewidth=0.5)
bars3 = ax.bar(x + width, fan_softmax_vars, width, label="Fan Votes (Softmax)", color="#F18F01", alpha=0.8, edgecolor="black", linewidth=0.5)

# 3. 数值标签（智能适配极小值/大值）
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # 极小值用科学计数法，其他按精度显示
        if height < 1e-5:
            label = f"{height:.1e}"
            y_pos = height + 0.1  # 避免与轴重叠
        elif height < 1:
            label = f"{height:.3f}"
            y_pos = height + 0.05
        else:
            label = f"{height:.2f}"
            y_pos = height + 0.2
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label, ha="center", va="bottom", fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# 4. 图表标注与布局
ax.set_ylabel("Variance Component", fontweight="bold")
ax.set_title("Variance Decomposition: Judges Scores vs Fan Votes (Multi-Method)", fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis="y")

# 调整y轴范围（适配softmax的大值）
max_var = max(fan_softmax_vars[1], judge_vars[2], fan_softmax_vars[2]) * 1.1
ax.set_ylim(0, max_var)

# 图例（位置优化）
ax.legend(loc="upper right", frameon=True, framealpha=0.9)

# 添加核心结论注释
ax.text(0.02, 0.98, 
        "Key Finding: Partner impacts votes more; Celebrity impacts scores more", 
        transform=ax.transAxes, fontsize=10, 
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.7),
        verticalalignment="top")

# 5. 保存与显示
plt.tight_layout()
plt.savefig("variance_decomposition_improved.png", dpi=300, bbox_inches="tight")
plt.show()

# 可选：单独生成基础版对比图（仅Judges + 线性投票）
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
bars1_basic = ax2.bar(x - width/2, judge_vars, width, label="Judges Score", color="#2E86AB", alpha=0.8, edgecolor="black", linewidth=0.5)
bars2_basic = ax2.bar(x + width/2, fan_linear_vars, width, label="Fan Votes (Linear)", color="#A23B72", alpha=0.8, edgecolor="black", linewidth=0.5)

add_value_labels(bars1_basic)
add_value_labels(bars2_basic)

ax2.set_ylabel("Variance Component", fontweight="bold")
ax2.set_title("Variance Decomposition: Judges Scores vs Fan Votes", fontweight="bold", pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.grid(axis="y")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("variance_decomposition_basic.png", dpi=300, bbox_inches="tight")
plt.show()
