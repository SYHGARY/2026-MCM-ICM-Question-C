import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ====================== 全局学术样式配置 ======================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 11

# ====================== 数据读取与预处理 ======================
df = pd.read_csv("optimized_fan_votes_uncertainty.csv")
std_col = [c for c in df.columns if "std" in c.lower()][0]
mean_col = [c for c in df.columns if "mean" in c.lower()][0]

# 周度统计（复用核心数据，保证一致性）
weekly_stats = df.groupby("week").agg({
    std_col: ["mean", "std", "count"],
    mean_col: "mean"
}).round(3)
weekly_stats.columns = ["avg_std", "std_std", "sample_size", "avg_mean"]
weekly = weekly_stats.reset_index()

# 计算变异系数（核心关联指标）
weekly["cv"] = (weekly["avg_std"] / weekly["avg_mean"] * 100).round(2)

# ====================== 可视化配置 ======================
# 学术级渐变配色
primary_color = '#1f77b4'
secondary_color = '#ff7f0e'
tertiary_color = '#2ca02c'
accent_color = '#d62728'
cmap = LinearSegmentedColormap.from_list("mcm_cmap", [primary_color, secondary_color, tertiary_color], N=256)

# 创建独立画布
fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# ====================== 双轴绘图 ======================
# 左轴：散点图（票数vs不确定性）
scatter = ax.scatter(weekly["avg_mean"], weekly["avg_std"], c=weekly["week"], cmap=cmap, s=120, 
                     alpha=0.8, edgecolors='white', linewidth=1.5, label='Weekly Uncertainty')

# 右轴：变异系数趋势线（稳定性指标）
ax_twin = ax.twinx()
cv_line = ax_twin.plot(weekly["avg_mean"], weekly["cv"], color=accent_color, linewidth=3, 
                       linestyle='-.', marker='D', markersize=6, markerfacecolor='white', 
                       markeredgecolor=accent_color, markeredgewidth=1.5, label='Coefficient of Variation')

# ====================== 格式优化 ======================
# 左轴配置
ax.set_xlabel('Average Inferred Fan Votes', fontweight='bold', fontsize=13)
ax.set_ylabel('Average Uncertainty (Std)', fontweight='bold', fontsize=13, color=primary_color)
ax.tick_params(axis='y', labelcolor=primary_color)
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)

# 右轴配置
ax_twin.set_ylabel('Coefficient of Variation (%)', fontweight='bold', fontsize=13, color=accent_color)
ax_twin.tick_params(axis='y', labelcolor=accent_color)

# 标题与图例
ax.set_title('Correlation Between Fan Vote Magnitude, Uncertainty, and Stability', 
             fontsize=15, fontweight='bold', pad=25, y=0.98)

# 合并双轴图例
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, fancybox=True, shadow=True)

# 颜色条（关联周数）
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Competition Week', fontweight='bold', fontsize=11)
cbar.ax.tick_params(direction='in')

# ====================== 保存 ======================
plt.tight_layout()
plt.savefig('MCM_Uncertainty_Correlation_Analysis.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('MCM_Uncertainty_Correlation_Analysis.eps', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
