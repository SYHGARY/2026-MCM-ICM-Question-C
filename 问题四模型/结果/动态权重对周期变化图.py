import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====================== 全局设置（紧凑专业）======================
plt.rcParams.update({
    "font.size": 9.5,
    "font.family": "Arial",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.framealpha": 0.9,
    "legend.edgecolor": "black",
    "legend.fancybox": False
})

# ====================== 高对比度配色（突出曲线变化）======================
COLOR_JUDGE = "#1E40AF"       # 裁判权重：深蓝（加深，增强对比）
COLOR_FAN = "#DC2626"         # 粉丝权重：深红（加深，增强对比）
COLOR_CONTESTANTS = "#3F7F93" # 选手数量：青绿（保持柔和，不抢戏）
COLOR_INHERIT = "#F59E0B"     # 权重继承：亮橙（醒目，突出节点）
COLOR_GRID = "#D1D5DB"        # 网格色：浅灰

# ====================== 数据加载与预处理 ======================
df = pd.read_csv("weights_by_season_week.csv")
season = 32  # 可修改为1-34任意赛季
plot_mode = "single_season"  # "single_season"或"all_season_avg"

# 数据筛选与处理
if plot_mode == "single_season":
    sub = df[df["season"] == season].copy()
    if len(sub) == 0:
        raise ValueError(f"赛季{season}无数据，请选择1-34之间的赛季")
    sub = sub.sort_values("week").reset_index(drop=True)
    weeks = sub["week"].values
    w_judge = sub["w_judge"].values * 100
    w_fan = sub["w_fan"].values * 100
    n_contestants = sub["n_contestants"].values
    used_inherit = sub["used_inherit"].values
    title_suffix = f"Season {season}"
elif plot_mode == "all_season_avg":
    sub = df.groupby("week").agg({
        "w_judge": "mean",
        "w_fan": "mean",
        "n_contestants": "mean"
    }).reset_index()
    weeks = sub["week"].values
    w_judge = sub["w_judge"].values * 100
    w_fan = sub["w_fan"].values * 100
    n_contestants = sub["n_contestants"].values
    used_inherit = np.zeros(len(weeks), dtype=bool)
    title_suffix = "All Seasons (Average)"

# ====================== 主图绘制（缩小尺寸+增强变化可见性）======================
# 缩小尺寸为8×5英寸（1.6:1紧凑比例），适配论文排版
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# 1. 核心曲线（加粗线条+高对比度配色，突出变化）
line1 = ax1.plot(weeks, w_judge, color=COLOR_JUDGE, linewidth=3.5,
                 marker='o', markersize=5, markerfacecolor='white',
                 markeredgecolor=COLOR_JUDGE, markeredgewidth=2, label="Judge Score Weight")
line2 = ax1.plot(weeks, w_fan, color=COLOR_FAN, linewidth=3.5,
                 marker='s', markersize=5, markerfacecolor='white',
                 markeredgecolor=COLOR_FAN, markeredgewidth=2, label="Fan Vote Weight")

# 2. 取消面积填充（避免模糊曲线变化，让波动更清晰）
# 移除fill_between，专注突出曲线本身的变化趋势

# 3. 权重继承节点（适度放大，强化关键变化点）
if plot_mode == "single_season":
    inherit_weeks = weeks[used_inherit]
    inherit_judge = w_judge[used_inherit]
    inherit_fan = w_fan[used_inherit]
    ax1.scatter(inherit_weeks, inherit_judge, color=COLOR_INHERIT, s=120,
                marker='*', edgecolor="black", linewidth=1, zorder=5, label="Weight Inheritance")
    ax1.scatter(inherit_weeks, inherit_fan, color=COLOR_INHERIT, s=120,
                marker='*', edgecolor="black", linewidth=1, zorder=5)

# 4. 选手数量曲线（弱化处理，避免干扰主曲线）
line3 = ax2.plot(weeks, n_contestants, color=COLOR_CONTESTANTS, linewidth=2,
                 linestyle='-.', marker='d', markersize=4, markerfacecolor='white',
                 markeredgecolor=COLOR_CONTESTANTS, markeredgewidth=1, label="Number of Contestants")

# ====================== 关键优化（缩小+突出变化）======================
# 轴标签与范围（优化刻度，让变化更易读）
ax1.set_xlabel("Competition Week", fontweight="bold", fontsize=10.5)
ax1.set_ylabel("Weight (%)", fontweight="bold", fontsize=10.5)
# 权重范围调整为20-80（聚焦变化区间，放大波动视觉效果）
ax1.set_ylim(20, 80)
ax1.set_yticks(np.arange(20, 81, 10))  # 加密刻度，凸显小幅度变化
ax1.grid(axis="y", color=COLOR_GRID)
ax1.set_xticks(weeks)
ax1.set_xticklabels([f"Week {w}" for w in weeks], rotation=30, ha="right")

# 次轴优化（弱化显示，不干扰主曲线）
ax2.set_ylabel("Number of Contestants", fontweight="bold", fontsize=10, color=COLOR_CONTESTANTS)
ax2.tick_params(axis='y', labelcolor=COLOR_CONTESTANTS, labelsize=9)
ax2.set_ylim(0, max(n_contestants) * 1.15)

# 图例（紧凑布局，不占用过多空间）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
           ncol=1, handletextpad=0.5, columnspacing=0.6, fontsize=9,
           frameon=True, edgecolor="black", bbox_to_anchor=(0.97, 0.97))

# 标题（简洁紧凑，适配小尺寸图表）
fig.suptitle(f"Dynamic Weight Changes - {title_suffix}",
             fontweight="bold", fontsize=11.5, y=0.96, x=0.5)

# 注释框（精简内容，紧凑显示）
inherit_count = len(inherit_weeks) if plot_mode == "single_season" else 0
info_text = (
    f"Coverage: 100.00% | Consistency: 97.73%\n"
    f"Judge: {w_judge.min():.1f}%-{w_judge.max():.1f}%\n"
    f"Fan: {w_fan.min():.1f}%-{w_fan.max():.1f}%\n"
    f"Inheritance: {inherit_count} weeks"
)
ax1.text(0.02, 0.02, info_text, transform=ax1.transAxes, fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.85, edgecolor="black"),
         verticalalignment="bottom", family="Arial", fontweight="bold")

# 布局调整（紧凑边距，适配小尺寸）
plt.tight_layout()
plt.subplots_adjust(top=0.91, bottom=0.16, left=0.12, right=0.88)

# ====================== 输出保存 ======================
save_path = f"dynamic_weights_compact_{title_suffix.replace(' ', '_')}.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.show()

print(f"紧凑优化版图表已保存至：{save_path}")

# 输出配套统计表格
if plot_mode == "single_season":
    weight_stats = sub[["week", "w_judge", "w_fan", "n_contestants", "used_inherit"]].copy()
    weight_stats[["w_judge", "w_fan"]] = weight_stats[["w_judge", "w_fan"]] * 100
    weight_stats.columns = ["Week", "Judge_Weight(%)", "Fan_Weight(%)", "N_Contestants", "Weight_Inheritance"]
    stats_path = f"q4_outputs_v3/weight_stats_compact_season_{season}.csv"
    weight_stats.to_csv(stats_path, index=False)
    print(f"配套统计表格已保存至：{stats_path}")
