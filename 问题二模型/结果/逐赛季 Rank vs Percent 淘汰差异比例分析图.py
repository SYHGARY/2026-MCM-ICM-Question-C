import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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

# ====================== 数据准备（整合核心差异指标） ======================
# 读取数据
df_summary = pd.read_csv("2_season_summary.csv")
df_weekly = pd.read_csv("2_weekly_elimination_compare.csv")

# 提取关键指标（按赛季排序）
seasons = df_summary["season"].values
elimination_difference_rate = df_summary["elimination_difference_rate"].values  # 淘汰差异率
champion_changed = df_summary["champion_changed"].values  # 冠军是否变动
champion_change_rate = sum(champion_changed) / len(seasons)

# 按赛制分组（1-2季：rank；3-27季：percent；28-34季：rank_special）
season_groups = []
for s in seasons:
    if s in [1, 2]:
        season_groups.append('rank_early')
    elif 3 <= s <= 27:
        season_groups.append('percent_main')
    else:
        season_groups.append('rank_special')

# 计算每组的平均差异率（用于背景标注）
group_avg = {
    'rank_early': np.mean(elimination_difference_rate[[s in [1,2] for s in seasons]]),
    'percent_main': np.mean(elimination_difference_rate[[3<=s<=27 for s in seasons]]),
    'rank_special': np.mean(elimination_difference_rate[[28<=s<=34 for s in seasons]])
}

# ====================== 统一配色方案 ======================
colors = {
    'rank_early': '#2E86AB',          # 蓝色（早期rank赛制）
    'percent_main': '#A23B72',       # 紫色（主体percent赛制）
    'rank_special': '#F18F01',       # 橙色（特殊rank赛制）
    'champion_change': '#C73E1D',    # 红色（冠军变动）
    'high_difference': '#8E44AD'     # 深紫色（高差异率≥0.6）
}

# 创建独立画布（统一尺寸基准）
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# ====================== 核心绘图：逐赛季淘汰差异率柱状图 ======================
# 绘制基础柱状图（按赛制着色，添加细黑框）
bars = []
for i, (season, rate, group) in enumerate(zip(seasons, elimination_difference_rate, season_groups)):
    bar = ax.bar(season, rate, width=0.7, color=colors[group], alpha=0.7,
                 edgecolor='black', linewidth=0.8, zorder=3)  # 关键修改：edgecolor改为black，linewidth微调为0.8
    bars.append(bar)

# 叠加冠军变动标注（统一红色边框+星号样式）
champion_label_added = False
for i, (season, rate, changed) in enumerate(zip(seasons, elimination_difference_rate, champion_changed)):
    if changed:
        ax.bar(season, rate, width=0.7, color='none', edgecolor=colors['champion_change'], linewidth=3, zorder=4)
        if not champion_label_added:
            ax.scatter(season, rate+0.02, color=colors['champion_change'], marker='*', s=120, zorder=5,
                       edgecolors='white', linewidth=1.5, label='Champion Changed')
            champion_label_added = True
        else:
            ax.scatter(season, rate+0.02, color=colors['champion_change'], marker='*', s=120, zorder=5,
                       edgecolors='white', linewidth=1.5)

# 标注高差异率赛季（≥0.6，统一标注样式）
high_diff_mask = elimination_difference_rate >= 0.6
high_diff_seasons = seasons[high_diff_mask]
high_diff_rates = elimination_difference_rate[high_diff_mask]
for season, rate in zip(high_diff_seasons, high_diff_rates):
    ax.annotate(f'{rate:.1%}', xy=(season, rate+0.03), ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=colors['high_difference'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors['high_difference'], alpha=0.8))

# ====================== 辅助元素：赛制分组线+平均差异率线 ======================
# 赛制分组垂直分隔线（统一样式）
ax.axvline(x=2.5, color='gray', linestyle=':', linewidth=2, alpha=0.8)
ax.axvline(x=27.5, color='gray', linestyle=':', linewidth=2, alpha=0.8)

# 分组平均差异率水平线（统一虚线样式）
ax.axhline(y=group_avg['rank_early'], color=colors['rank_early'], linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'Early Rank Avg: {group_avg["rank_early"]:.1%}')
ax.axhline(y=group_avg['percent_main'], color=colors['percent_main'], linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'Main Percent Avg: {group_avg["percent_main"]:.1%}')
ax.axhline(y=group_avg['rank_special'], color=colors['rank_special'], linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'Special Rank Avg: {group_avg["rank_special"]:.1%}')

# 赛制分组文字标注（统一样式）
ax.text(1.5, 0.95, 'Early Rank\n(Seasons 1-2)', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
ax.text(15, 0.95, 'Main Percent\n(Seasons 3-27)', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.5))
ax.text(31, 0.95, 'Special Rank\n(Seasons 28-34)', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))

# ====================== 格式优化（统一版） ======================
ax.set_xlabel('Season', fontweight='bold', fontsize=13)
ax.set_ylabel('Elimination Difference Rate (Rank vs Percent)', fontweight='bold', fontsize=13)
ax.set_title('Elimination Outcome Difference Between Rank and Percent Combination Rules',
             fontsize=16, fontweight='bold', pad=25, y=0.98)

# 坐标轴范围与刻度（统一样式）
ax.set_xlim(seasons.min()-0.8, seasons.max()+0.8)
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.01, 0.1)])
ax.set_xticks(seasons[::2])
ax.set_xticklabels([f'{int(s)}' for s in seasons[::2]], rotation=45, ha='right')

# 网格与图例（统一样式）
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, axis='y')
ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=9)

# ====================== 保存（统一分辨率和格式） ======================
plt.tight_layout()
plt.savefig('MCM_Elimination_Difference_Rate_Analysis.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('MCM_Elimination_Difference_Rate_Analysis.eps', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
