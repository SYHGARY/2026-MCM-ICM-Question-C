import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取CSV文件
df = pd.read_csv('stability_kfold_weights.csv')

# 筛选出有数据（非空）的行
df_valid = df.dropna(subset=['w_judge_mean', 'w_fan_mean'])

# 设置图形大小和风格

# 创建分组柱状图
x = np.arange(len(df_valid))
width = 0.35


# 按赛季分组绘制子图
seasons = df_valid['season'].unique()
num_seasons = len(seasons)
fig, axes = plt.subplots(nrows=(num_seasons+4)//5, ncols=5, figsize=(20, 15))
axes = axes.flatten()

for i, season in enumerate(seasons):
    if i >= len(axes):
        break
    ax = axes[i]
    season_data = df_valid[df_valid['season'] == season]
    
    x_local = np.arange(len(season_data))
    ax.bar(x_local - width/2, season_data['w_judge_mean'], width, color='skyblue', label='Judge')
    ax.bar(x_local + width/2, season_data['w_fan_mean'], width, color='yellow', label='Fan')
    
    ax.set_title(f'Season {season}', fontsize=10)
    ax.set_xticks(x_local)
    ax.set_xticklabels([f'W{w}' for w in season_data['week']], rotation=45, fontsize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 隐藏多余的子图
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('K-Fold Weights by Season', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig('1.png', dpi=300, bbox_inches='tight')
