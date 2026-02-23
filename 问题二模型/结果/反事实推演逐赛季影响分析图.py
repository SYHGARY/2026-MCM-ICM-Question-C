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
color_controversy = '#d62728'      # 红色（争议选手赛季）

# ====================== 数据准备 ======================
# 读取数据
df_season = pd.read_csv("2_反事实推演_season_summary.csv")
df_controversy = pd.read_csv("2_反事实推演_controversy_celebrity_summary.csv")

# 提取逐赛季指标
seasons = df_season["season"].values
rank_percent_diff = df_season["rank_vs_percent_diff_rate"].values
rank_js_change = df_season["rank_js_change_rate"].values
percent_js_change = df_season["percent_js_change_rate"].values

# 争议选手所在赛季（去重）
controversy_seasons = df_controversy["season"].unique()

# 总体平均值（用于参考线）
overall_rank_percent = 0.436
overall_rank_js = 0.144
overall_percent_js = 0.425

# ====================== 创建独立画布（统一尺寸基准） ======================
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# ====================== 核心绘图 ======================
# 绘制逐赛季趋势线（统一线条和标记样式）
ax.plot(seasons, rank_percent_diff, color=color_rank_percent, linewidth=3.5, marker='o', markersize=5,
         markerfacecolor='white', markeredgecolor=color_rank_percent, markeredgewidth=2, label='Rank vs Percent')
ax.plot(seasons, rank_js_change, color=color_rank_js, linewidth=3.5, marker='s', markersize=5,
         markerfacecolor='white', markeredgecolor=color_rank_js, markeredgewidth=2, label='Rank + Judges Save')
ax.plot(seasons, percent_js_change, color=color_percent_js, linewidth=3.5, marker='^', markersize=5,
         markerfacecolor='white', markeredgecolor=color_percent_js, markeredgewidth=2, label='Percent + Judges Save')

# 标注争议选手赛季（统一填充样式）
for season in controversy_seasons:
    ax.fill_between([season-0.4, season+0.4], 0, 1.0, alpha=0.2, color=color_controversy,
                     label='Controversial Celebrity' if season == controversy_seasons[0] else "")

# 绘制总体平均参考线（统一虚线样式）
ax.axhline(y=overall_rank_percent, color=color_rank_percent, linestyle='--', linewidth=2.5, alpha=0.7,
             label=f'Overall Avg: {overall_rank_percent:.1%}')
ax.axhline(y=overall_rank_js, color=color_rank_js, linestyle='--', linewidth=2.5, alpha=0.7,
             label=f'Overall Avg: {overall_rank_js:.1%}')
ax.axhline(y=overall_percent_js, color=color_percent_js, linestyle='--', linewidth=2.5, alpha=0.7,
             label=f'Overall Avg: {overall_percent_js:.1%}')

# 标注高影响赛季（统一标注样式）
high_impact_mask = rank_percent_diff >= 0.8
high_impact_seasons = seasons[high_impact_mask]
high_impact_values = rank_percent_diff[high_impact_mask]
for season, value in zip(high_impact_seasons, high_impact_values):
    ax.annotate(f'{value:.1%}', xy=(season, value+0.03), ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color_rank_percent,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color_rank_percent, alpha=0.8))

# ====================== 格式优化（统一版） ======================
ax.set_xlabel('Season', fontweight='bold', fontsize=13)
ax.set_ylabel('Fraction of Weeks with Changed Outcomes', fontweight='bold', fontsize=13)
ax.set_title('Season-by-Season Impact of Rule Changes on Elimination Outcomes',
              fontsize=16, fontweight='bold', pad=25, y=0.98)

# 坐标轴调整（统一刻度样式）
ax.set_xlim(seasons.min()-1.5, seasons.max()+1.5)
ax.set_ylim(0, 1.05)
ax.set_xticks(seasons[::1])
ax.set_xticklabels([f'{int(s)}' for s in seasons[::1]], rotation=45, ha='right', fontsize=9)
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.set_yticklabels([f'{x:.0%}' for x in np.arange(0, 1.01, 0.1)], fontsize=9)

# 网格与图例（统一样式）
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='y')
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.18),
           fontsize=9)

# 布局调整（统一边距）
plt.tight_layout()
plt.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.22)

# ====================== 保存（统一分辨率和格式） ======================
plt.savefig('MCM_Counterfactual_Seasonal_Impact.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('MCM_Counterfactual_Seasonal_Impact.eps', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
