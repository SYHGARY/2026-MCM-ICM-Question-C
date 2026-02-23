import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====================== 全局学术样式配置（统一版） ======================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 11
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True

# ====================== 统一配色方案 ======================
color_rank_percent = '#1f77b4'       # 深蓝色（Rank vs Percent）
color_rank_js = '#ff7f0e'          # 橙色（Rank + Judges Save）
color_percent_js = '#2ca02c'       # 绿色（Percent + Judges Save）

# ====================== 数据准备（总体平均值，与输出一致） ======================
labels = ["Rank vs Percent", "Rank + Judges Save", "Percent + Judges Save"]
overall_values = [0.436, 0.144, 0.425]  # 43.6% / 14.4% / 42.5%
colors = [color_rank_percent, color_rank_js, color_percent_js]

# ====================== 创建独立画布（统一尺寸基准） ======================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# ====================== 核心绘图 ======================
# 绘制柱状图（统一柱子样式）
bars = ax.bar(labels, overall_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1, zorder=3, width=0.6)

# 标注数值（统一百分比+详细说明样式）
for bar, value in zip(bars, overall_values):
    height = bar.get_height()
    # 主数值标注
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    # 补充说明（小字体，统一样式）
    bar_label = labels[list(bars).index(bar)]
    if bar_label == "Rank vs Percent":
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                'Rule Switch', ha='center', va='top', fontsize=9, style='italic', color='white')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.02,
                'With Judges Save', ha='center', va='top', fontsize=9, style='italic', color='white')

# ====================== 格式优化（统一版） ======================
ax.set_ylabel('Average Fraction of Weeks with Changed Outcomes', fontweight='bold', fontsize=13)
ax.set_title('Overall Impact of Rule Changes on Elimination Outcomes',
              fontsize=16, fontweight='bold', pad=25, y=0.98)

# 调整坐标轴（统一样式）
ax.set_ylim(0, 0.55)
ax.set_yticks(np.arange(0, 0.51, 0.1))
ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 0.51, 0.1)], fontsize=10)

# 网格与图例（统一样式）
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='y')
ax.legend(labels, loc='upper right', fontsize=10)

# 布局调整（统一边距）
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

# ====================== 保存（统一分辨率和格式） ======================
plt.savefig('MCM_Counterfactual_Overall_Impact.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('MCM_Counterfactual_Overall_Impact.eps', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
