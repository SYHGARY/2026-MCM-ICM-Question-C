import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ====================== 全局学术样式配置（统一论文风格） ======================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 11
plt.rcParams['legend.framealpha'] = 0.95

# ====================== 数据读取与预处理 ======================
df = pd.read_csv("optimized_fan_votes_uncertainty.csv")
std_col = [c for c in df.columns if "std" in c.lower()][0]
mean_col = [c for c in df.columns if "mean" in c.lower()][0]

# 周度统计（保持核心数据逻辑）
weekly_stats = df.groupby("week").agg({
    std_col: ["mean", "std"],
    mean_col: "mean"
}).round(2)
weekly_stats.columns = ["avg_std", "std_std", "avg_mean"]
weekly = weekly_stats.reset_index()

# ====================== 可视化配置（统一学术级配色） ======================
# 统一学术配色体系
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors[:3], N=100)

# 创建独立画布（论文单图黄金尺寸）
fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# ====================== 散点+折线融合绘制 ======================
# 1. 核心散点图（关联票数与不确定性，颜色映射周数）
scatter = ax.scatter(weekly["avg_mean"], weekly["avg_std"], 
                     c=weekly["week"], cmap=cmap, s=100,  # 放大散点更醒目
                     alpha=0.8, edgecolors='white', linewidth=1.5, 
                     label='Weekly Data Point')

# 2. 趋势折线（拟合散点趋势，体现整体规律）
# 用多项式拟合生成平滑折线（避免折线过于生硬）
z = np.polyfit(weekly["avg_mean"], weekly["avg_std"], 2)  # 二次拟合更贴合数据趋势
p = np.poly1d(z)
# 生成拟合线的x轴数据（均匀分布，保证折线平滑）
x_fit = np.linspace(weekly["avg_mean"].min(), weekly["avg_mean"].max(), 100)
y_fit = p(x_fit)

# 绘制趋势折线（与散点配色呼应，加粗突出）
ax.plot(x_fit, y_fit, color=colors[2], linewidth=3, linestyle='-', 
        marker=None, label='Trend Line (2nd Order Fit)', alpha=0.8)

# ====================== 格式优化（统一学术风格） ======================
ax.set_xlabel('Average Fan Votes (Mean)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Uncertainty (Std)', fontsize=12, fontweight='bold')
ax.set_title('Correlation Between Average Votes and Uncertainty', 
             fontsize=14, fontweight='bold', pad=20)

# 图例（统一风格：带阴影+高透明度）
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, framealpha=0.95, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# 颜色条（统一样式）
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Week', fontsize=10, fontweight='bold')
cbar.ax.tick_params(direction='in')

# ====================== 高分辨率保存 ======================
plt.tight_layout()
plt.savefig('MCM_Uncertainty_Correlation_Scatter_Line.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('MCM_Uncertainty_Correlation_Scatter_Line.eps', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
