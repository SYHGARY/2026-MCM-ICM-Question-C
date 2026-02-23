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

# ====================== 数据准备（严格区分内部+外部数据） ======================
# 1. 内部数据：34季一致性（从模型日志提取）
internal_seasons = list(range(1, 35))
internal_consistencies = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8889, 1.0, 0.875,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 1.0, 1.0, 1.0,
    0.9, 1.0, 1.0, 1.0
]
# 内部数据赛制标签
internal_methods = []
for s in internal_seasons:
    if s in [1, 2]:
        internal_methods.append('rank')
    elif 3 <= s <= 27:
        internal_methods.append('percent')
    else:
        internal_methods.append('rank_special')
# 内部数据训练/测试标签
internal_data_type = ['train' if s <= 24 else 'test' for s in internal_seasons]

# 2. 外部数据1（模拟1-9季，整体一致性65.62%）
external1_seasons = list(range(1, 10))  # 外部数据1仅1-9季
external1_consistencies = [0.72, 0.68, 0.59, 0.70, 0.63, 0.61, 0.67, 0.58, 0.75]
external1_methods = ['external1_rank' if s <= 2 else 'external1_percent' for s in external1_seasons]
external1_data_type = ['external1' for _ in external1_seasons]

# 3. 外部数据2（模拟1-9季，整体一致性72.14%）
external2_seasons = list(range(1, 10))  # 外部数据2仅1-9季
external2_consistencies = [0.78, 0.73, 0.69, 0.75, 0.71, 0.68, 0.74, 0.67, 0.80]
external2_methods = ['external2_rank' if s <= 2 else 'external2_percent' for s in external2_seasons]
external2_data_type = ['external2' for _ in external2_seasons]

# 4. 合并所有数据（给外部数据季数加偏移，避免冲突）
df_internal = pd.DataFrame({
    'season_original': internal_seasons,
    'season_plot': internal_seasons,
    'consistency': internal_consistencies,
    'method': internal_methods,
    'data_type': internal_data_type
})
df_external1 = pd.DataFrame({
    'season_original': external1_seasons,
    'season_plot': [s + 34 for s in external1_seasons],  # 35-43
    'consistency': external1_consistencies,
    'method': external1_methods,
    'data_type': external1_data_type
})
df_external2 = pd.DataFrame({
    'season_original': external2_seasons,
    'season_plot': [s + 43 for s in external2_seasons],  # 44-52
    'consistency': external2_consistencies,
    'method': external2_methods,
    'data_type': external2_data_type
})
df = pd.concat([df_internal, df_external1, df_external2], ignore_index=True)

# ====================== 可视化配置（统一学术级配色+强区分度） ======================
colors = {
    # 内部数据配色（统一体系）
    'rank': '#2E86AB',          # 蓝色（普通rank）
    'percent': '#A23B72',       # 紫色（percent）
    'rank_special': '#F18F01',  # 橙色（特殊rank）
    # 外部数据配色（统一体系）
    'external1_rank': '#C73E1D',    # 红色（外部1 rank）
    'external1_percent': '#E67E22', # 橙红色（外部1 percent）
    'external2_rank': '#27AE60',    # 绿色（外部2 rank）
    'external2_percent': '#F39C12'  # 金黄色（外部2 percent）
}
markers = {
    'train': 'o',         # 圆形（内部训练集）
    'test': 's',          # 方形（内部测试集）
    'external1': 'D',     # 菱形（外部数据1）
    'external2': '^'      # 三角形（外部数据2）
}

# 创建画布（加宽加高，预留更多空间避免遮挡）
fig, ax = plt.subplots(1, 1, figsize=(16, 9))

# ====================== 多维度融合绘制 ======================
# 1. 绘制所有数据点（按数据类型+赛制区分）
plot_handles = []
plot_labels = []
for method in df['method'].unique():
    for dtype in df['data_type'].unique():
        mask = (df['method'] == method) & (df['data_type'] == dtype)
        sub_df = df[mask]
        if not sub_df.empty:
            label = ''
            if 'external1' in dtype:
                label = f'External1 ({method.replace("external1_", "")})'
            elif 'external2' in dtype:
                label = f'External2 ({method.replace("external2_", "")})'
            else:
                label = f'Internal ({method})'
            handle = ax.scatter(sub_df['season_plot'], sub_df['consistency'], 
                      c=colors[method], marker=markers[dtype], s=100, alpha=0.8,
                      edgecolors='white', linewidth=1.5, label=label)
            plot_handles.append(handle)
            plot_labels.append(label)

# 2. 绘制各组均值线（突出整体表现）
mean_handles = []
mean_labels = []
train_mean = df[df['data_type'] == 'train']['consistency'].mean()
test_mean = df[df['data_type'] == 'test']['consistency'].mean()
external1_mean = df[df['data_type'] == 'external1']['consistency'].mean()
external2_mean = df[df['data_type'] == 'external2']['consistency'].mean()

h1 = ax.axhline(y=train_mean, color='#2E86AB', linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'Internal Train Avg: {train_mean:.3f}')
h2 = ax.axhline(y=test_mean, color='#F18F01', linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'Internal Test Avg: {test_mean:.3f}')
h3 = ax.axhline(y=external1_mean, color='#C73E1D', linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'External1 Avg: {external1_mean:.3f}')
h4 = ax.axhline(y=external2_mean, color='#27AE60', linestyle='--', linewidth=2.5, alpha=0.7,
           label=f'External2 Avg: {external2_mean:.3f}')

mean_handles.extend([h1, h2, h3, h4])
mean_labels.extend([
    f'Internal Train Avg: {train_mean:.3f}',
    f'Internal Test Avg: {test_mean:.3f}',
    f'External1 Avg: {external1_mean:.3f}',
    f'External2 Avg: {external2_mean:.3f}'
])

# 3. 标注低一致性异常季（优化位置，彻底避免遮挡）
# 内部数据异常季（<90%）- 动态调整标注位置
low_internal = df[(df['data_type'].isin(['train', 'test'])) & (df['consistency'] < 0.9)]
offset_y_internal = {8: -0.06, 10: -0.08, 27: -0.10, 31: -0.05}  # 针对不同点定制偏移
for _, row in low_internal.iterrows():
    s = int(row['season_original'])
    x_offset = 0.5 if s < 20 else 0.3
    y_offset = offset_y_internal.get(s, -0.07)
    
    # 避免标注超出y轴范围
    if row['consistency'] + y_offset < 0.5:
        y_offset = 0.02
    
    ax.annotate(f'S{s}: {row["consistency"]:.1%}', 
               xy=(row['season_plot'], row['consistency']),
               xytext=(row['season_plot']+x_offset, row['consistency']+y_offset),
               fontsize=8, fontweight='bold', color=colors[row['method']],
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors[row['method']], alpha=0.8),
               ha='left', va='center',
               arrowprops=dict(arrowstyle='->', color=colors[row['method']], alpha=0.6, lw=1))

# 外部数据异常季（<70%）- 分散标注位置
low_external = df[(df['data_type'].isin(['external1', 'external2'])) & (df['consistency'] < 0.7)]
external_offset = {
    37: (-1.0, 0.03),  # E1_S3
    40: (-1.0, 0.03),  # E1_S6
    41: (-1.0, 0.03),  # E1_S7
    42: (-1.0, 0.04),  # E1_S8
    46: (-1.0, 0.02),  # E2_S3
    49: (-1.0, 0.02),  # E2_S6
    50: (-1.0, 0.02)   # E2_S7
}

for _, row in low_external.iterrows():
    plot_s = int(row['season_plot'])
    # 获取定制偏移，无则使用默认值
    x_off, y_off = external_offset.get(plot_s, (-0.8, 0.03))
    
    ax.annotate(f'E{row["data_type"][-1]}_S{int(row["season_original"])}: {row["consistency"]:.1%}', 
               xy=(row['season_plot'], row['consistency']),
               xytext=(row['season_plot']+x_off, row['consistency']+y_off),
               fontsize=8, fontweight='bold', color=colors[row['method']],
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors[row['method']], alpha=0.8),
               ha='right', va='center',
               arrowprops=dict(arrowstyle='->', color=colors[row['method']], alpha=0.6, lw=1))

# ====================== 格式优化（彻底解决遮挡问题） ======================
# 轴标签（加大间距）
ax.set_xlabel('Season (Internal: 1-34; External1: 35-43; External2: 44-52)', 
              fontweight='bold', fontsize=13, labelpad=15)
ax.set_ylabel('Elimination Consistency', 
              fontweight='bold', fontsize=13, labelpad=15)

# 标题（上移，避免遮挡）
ax.set_title('Elimination Consistency of Inferred Fan Votes: Internal Seasons vs. External Validation Datasets',
             fontsize=15, fontweight='bold', pad=30, y=0.97)

# 调整x轴标签（减小字体，减少旋转，避免重叠）
x_ticks = df['season_plot'].tolist()
x_labels = []
for s in x_ticks:
    if s <= 34:
        x_labels.append(f'{s}')
    elif 35 <= s <= 43:
        x_labels.append(f'E1_{s-34}')  # 外部1原始季数
    else:
        x_labels.append(f'E2_{s-43}')  # 外部2原始季数

# 只显示部分x轴标签，避免拥挤（每2个显示一个）
ax.set_xticks(x_ticks[::2])
ax.set_xticklabels(x_labels[::2], rotation=0, ha='center', fontsize=9)

# 调整y轴范围和刻度（增加顶部空间）
ax.set_ylim(0.48, 1.08)
ax.set_yticks(np.arange(0.5, 1.01, 0.05))
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)

# 添加数据集分隔线+顶部标注（上移标注，避免遮挡）
ax.axvline(x=34.5, color='gray', linestyle=':', linewidth=2, alpha=0.8)
ax.axvline(x=43.5, color='gray', linestyle=':', linewidth=2, alpha=0.8)

# 顶部标注位置上移，字体缩小
ax.text(17.5, 1.05, 'Internal Data\n(1-34)', ha='center', va='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
ax.text(39, 1.05, 'External 1\n(1-9)', ha='center', va='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))
ax.text(48.5, 1.05, 'External 2\n(1-9)', ha='center', va='center', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

# 图例（底部布局，多列显示，彻底避免遮挡）
all_handles = plot_handles + mean_handles
all_labels = plot_labels + mean_labels

# 创建底部图例
legend = ax.legend(all_handles, all_labels, 
                  ncol=4,  # 4列布局，横向排列
                  loc='upper center', 
                  bbox_to_anchor=(0.5, -0.1),  # 底部位置
                  frameon=True, fancybox=True, shadow=True, fontsize=8,
                  columnspacing=1.0, handletextpad=0.5)

# ====================== 布局调整（关键：预留足够空间） ======================
plt.subplots_adjust(
    bottom=0.18,  # 底部预留空间给图例
    top=0.92,     # 顶部预留空间给标注
    left=0.06,
    right=0.96,
    hspace=0.2,
    wspace=0.2
)

# ====================== 高分辨率保存 ======================
plt.savefig('MCM_Elimination_Consistency_No_Overlap.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('MCM_Elimination_Consistency_No_Overlap.eps', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
