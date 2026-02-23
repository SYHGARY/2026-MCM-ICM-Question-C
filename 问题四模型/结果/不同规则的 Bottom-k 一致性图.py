import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 全局设置（兼容低版本matplotlib，移除所有不兼容参数）
plt.rcParams.update({
    "font.size": 11,
    "font.family": "Arial",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.unicode_minus": False  # 解决负号显示问题
})

# 1. 加载补充数据（精确一致性数值+原始次数）
labels = ["Bottom-1", "Bottom-2", "Bottom-3"]
base_consistency = [62.50, 88.64, 97.73]  # 基础评分一致性
adj_consistency = [62.88, 89.39, 97.35]   # 调整后评分一致性
x = np.arange(len(labels))
width = 0.35  # 柱宽适配

# 2. 创建图表
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制柱状图（添加边框增强区分度）
bars1 = ax.bar(x - width/2, base_consistency, width, 
               label="Combined Score", color="#2E86AB", alpha=0.8, 
               edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, adj_consistency, width, 
               label="Stability-Adjusted Score", color="#2ca02c", alpha=0.8,
               edgecolor="black", linewidth=0.5)

# 3. 数值标签（显示百分比+原始次数，匹配补充数据）
def add_value_labels(bars):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # 对应补充数据中的原始次数（base/adjusted分别对应不同n）
        counts = [165, 234, 258] if bars == bars1 else [166, 236, 257]
        label = f"{height:.2f}%\n(n={counts[i]})"
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, 
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

add_value_labels(bars1)
add_value_labels(bars2)

# 4. 图表标注优化（移除所有不兼容参数）
ax.set_ylabel("Elimination Consistency (%)", fontweight="bold")
ax.set_title("Elimination Consistency: Base vs Stability-Adjusted Scoring Rules", 
             fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)

# 简化grid设置（确保兼容所有版本）
ax.grid(axis="y")

# 调整y轴范围（聚焦有效区间，增强对比）
ax.set_ylim(60, 100)
ax.set_yticks(np.arange(60, 101, 5))

# 修复图例错误：移除linewidth参数（低版本matplotlib不支持）
ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="black")

# 添加核心补充信息注释（保留关键数据说明）
info_text = (
    f"Fan-vote Merge Coverage: 100.00%\n"
    f"Total Elimination Weeks: 264\n"
    f"Adjusted Score Improves Bottom-1/2 Consistency"
)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
        verticalalignment="top")

# 5. 保存高清图片（适配论文提交）
plt.tight_layout()
plt.savefig("elimination_consistency_final.png", 
            dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

# 6. 消除原数据处理的FutureWarning（可选使用）
def fix_groupby_warning(df, group_cols, func):
    """修复DataFrameGroupBy.apply的FutureWarning"""
    return df.groupby(group_cols, group_keys=False).apply(
        lambda g: func(g.drop(columns=group_cols)), include_groups=False
    )

# 原代码line163替换示例（若需处理原始数据）
# 替换：.apply(lambda g: pd.Series({"pred_bottomk": bottomk_names(g)}))
# 为：fix_groupby_warning(df, group_cols=["season", "week"], 
#                        func=lambda g: pd.Series({"pred_bottomk": bottomk_names(g)}))
