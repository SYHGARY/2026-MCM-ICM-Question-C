import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====================== 全局学术样式配置（适配论文版） ======================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 14  # 基础字体放大
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True

# ====================== 统一配色方案 ======================
color_rank = '#1f77b4'       # 深蓝色（Rank规则）
color_percent = '#ff7f0e'    # 橙色（Percent规则）
color_champion = '#d62728'   # 红色（冠军变动）
color_rank_area = '#e8f4f8'  # 浅蓝色（Rank优势区域）
color_percent_area = '#fff3cd'# 浅橙色（Percent优势区域）
color_equal_area = '#f0f0f0' # 浅灰色（两者相等区域）

# ====================== 数据准备 ======================
df_summary = pd.read_csv("2_season_summary.csv")
seasons = df_summary["season"].values
rank_spearman = df_summary["rank_spearman"].values
percent_spearman = df_summary["percent_spearman"].values
champion_changed = df_summary["champion_changed"].values
more_fan_friendly = df_summary["more_fan_friendly"].values

# 计算统计信息
prefer_rank = sum(1 for x in more_fan_friendly if x == "Rank Combination")
prefer_percent = sum(1 for x in more_fan_friendly if x == "Percent Combination")
prefer_equal = sum(1 for x in more_fan_friendly if x == "Equal")
champion_change_count = sum(champion_changed)

# ====================== 创建画布（缩小尺寸适配论文） ======================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # 原14x7 → 10x6，更适配论文版面

# ====================== 核心绘图 ======================
# 绘制趋势线（线条/标记适度放大）
ax.plot(seasons, rank_spearman, color=color_rank, linewidth=4, marker='o', markersize=7,
        markerfacecolor='white', markeredgecolor=color_rank, markeredgewidth=2, label='Rank Combination')
ax.plot(seasons, percent_spearman, color=color_percent, linewidth=4, marker='s', markersize=7,
        markerfacecolor='white', markeredgecolor=color_percent, markeredgewidth=2, label='Percent Combination')

# 标注冠军变动赛季（标记放大）
change_seasons = seasons[champion_changed]
change_rank = rank_spearman[champion_changed]
change_percent = percent_spearman[champion_changed]
ax.scatter(change_seasons, change_rank+0.01, color=color_champion, marker='*', s=150, zorder=5,
           edgecolors='white', linewidth=2, label=f'Champion Changed ({champion_change_count} seasons)')
ax.scatter(change_seasons, change_percent+0.01, color=color_champion, marker='*', s=150, zorder=5,
           edgecolors='white', linewidth=2)

# 背景填充（保持原有逻辑）
for i, season in enumerate(seasons):
    y_min = min(rank_spearman[i], percent_spearman[i]) - 0.005
    y_max = max(rank_spearman[i], percent_spearman[i]) + 0.005
    if more_fan_friendly[i] == "Rank Combination":
        ax.fill_between([season-0.35, season+0.35], y_min, y_max, alpha=0.4, color=color_rank_area)
    elif more_fan_friendly[i] == "Percent Combination":
        ax.fill_between([season-0.35, season+0.35], y_min, y_max, alpha=0.4, color=color_percent_area)
    else:
        ax.fill_between([season-0.35, season+0.35], y_min, y_max, alpha=0.4, color=color_equal_area)

# 统计文本框（字体放大，适配小画布）
stats_text = f"""Rule Preference Summary:
• Prefer Percent: {prefer_percent} seasons ({prefer_percent/len(seasons)*100:.1f}%)
• Prefer Rank: {prefer_rank} seasons ({prefer_rank/len(seasons)*100:.1f}%)
• Equal: {prefer_equal} seasons ({prefer_equal/len(seasons)*100:.1f}%)
• Champion Changed: {champion_change_count} seasons"""

ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

# ====================== 格式优化（字体全面放大） ======================
ax.set_xlabel('Season', fontweight='bold', fontsize=15)
ax.set_ylabel('Spearman Correlation with Fan Preference', fontweight='bold', fontsize=15)
ax.set_title('Fan Satisfaction Comparison: Rank vs Percent Combination Rules',
              fontsize=18, fontweight='bold', pad=20, y=0.98)

# 坐标轴调整（刻度字体放大）
ax.set_xlim(seasons.min()-1, seasons.max()+1)
ax.set_ylim(0.78, 1.01)
ax.set_xticks(seasons[::1])
ax.set_xticklabels([f'{int(s)}' for s in seasons[::1]], rotation=45, ha='right', fontsize=12)
ax.set_yticks(np.arange(0.8, 1.01, 0.02))
ax.set_yticklabels([f'{x:.2f}' for x in np.arange(0.8, 1.01, 0.02)], fontsize=12)

# 网格与图例（图例字体放大）
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='y')
ax.legend(loc='lower left', fontsize=12)

# 布局调整（适配小画布）
plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.18)

# ====================== 保存（保持高分辨率） ======================
plt.savefig('MCM_Fan_Satisfaction_Thesis_Version.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('MCM_Fan_Satisfaction_Thesis_Version.eps', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
