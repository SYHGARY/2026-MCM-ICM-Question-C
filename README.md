# Decoding the Dance: Inverse Optimization for Hidden Fan Votes and Fairer Scoring Mechanisms in Dancing with the Stars

&emsp;![1](https://img.shields.io/badge/language-python-blue.svg)
![2](https://img.shields.io/badge/license-NonCommercial--NoticeRequired-red)

&emsp;This repository contains the code and supplementary materials for the paper "Decoding the Dance: An Inverse Optimization Approach to Estimating Hidden Fan Votes and Designing Fairer Scoring Mechanisms in Dancing with the Stars" , submitted to the 2026 Mathematical Contest in Modeling (MCM).

---

## 📖 Overview
&emsp;Dancing with the Stars (DWTS) combines judges' scores with undisclosed fan votes to determine eliminations, often sparking controversy when popular but weak contestants advance. Since fan vote totals are never released, evaluating the fairness of the scoring system is a fundamental inference challenge.\
&emsp;**We develop a unified, data-driven framework to:**\
&emsp;1. Reconstruct latent fan votes from elimination results using inverse optimization with temporal smoothness.\
&emsp;2. Compare aggregation rules (Rank vs. Percent) via counterfactual analysis.\
&emsp;3. Quantify heterogeneous influences on judges and fans using crossed random-effects mixed models.\
&emsp;4. Design a fairer, dynamic scoring mechanism with entropy-based weighting and stability penalties.\
&emsp;5. Our approach achieves 98.28% consistency with historical eliminations across 34 seasons and provides actionable insights for improving fairness in DWTS and similar competition systems.

## 🔍 Key Contributions
&emsp;· Inverse optimization framework for estimating hidden vote shares from elimination-consistency constraints.\
&emsp;· Counterfactual simulations showing that switching between Rank and Percent rules alters elimination outcomes in 43.6% of weeks.\
&emsp;· Mixed-effects models revealing that judges prioritize technical performance and age, while fans are strongly influenced by professional partner effects.\
&emsp;· Novel dynamic weighting mechanism (entropy-based + stability penalty) that improves Bottom-2 consistency to 89.39% while reducing controversial eliminations.

## 🧠 Methodology
&emsp;**We tackle four sequential research tasks:**\
&emsp;**Task 1: Fan Vote Estimation (Inverse Optimization)**\
&emsp;· Formulated as a constrained optimization problem: weekly fan vote shares are latent variables constrained by official elimination rules.\
&emsp;· Added regularization: (i) minimum deviation from a judge-score prior, (ii) temporal smoothness.\
&emsp;· Solved via convex optimization (Percent seasons) or feasibility sampling (Rank seasons).\
&emsp;· Uncertainty quantified via Monte Carlo ensemble (50 iterations, 2% jitter).

&emsp;**Task 2: Rule Comparison (Counterfactual Analysis)**\
&emsp;· Fixed inferred fan votes, simulated elimination outcomes under Rank and Percent rules.\
&emsp;· Measured fan satisfaction using Spearman correlation between final placement and average fan rank.\
&emsp;· Evaluated impact of Judge Save mechanism on controversial cases.

&emsp;**Task 3: Factor Analysis (Crossed Random-Effects Models)**\
&emsp;· Judge Score Model: Linear mixed model with celebrity/partner random effects, fixed effects (age, industry, week, season).\
&emsp;· Fan Vote Logit-Share Model: Logit-transformed vote shares with same random-effects structure.\
&emsp;· Variance decomposition to separate drivers of technical scoring vs. popular voting.

&emsp;**Task 4: Dynamic Scoring Mechanism**\
&emsp;· Entropy Weight Method: Weekly weights for judges and fans based on information dispersion.\
&emsp;· Stability Penalty: Penalizes contestants with high rank volatility over a rolling window.\
&emsp;· Resulting rule adapts weights dynamically, reducing short-term noise.

## 📊 Results Summary
| Metric | Value |
|--------|-------|
| Consistency of reconstructed fan votes (overall) | **98.28%** |
| Consistency on test seasons | **96.5%** |
| External validation (similar show)	| **~65–72%** |
| Weeks where Rank vs. Percent rules differ	| **43.6%** |
| Weeks changed by Judge Save (under Rank)	| **14.4%** |
| Weeks changed by Judge Save (under Percent)	| **42.5%** |
| Judge score: age coefficient	| **–1.55 (p < 0.001)** |
| Fan vote: age coefficient	| **–0.50 (p < 0.01)** |
| Partner effect variance (fan model)	| **~40% of total random variance** |
| Proposed rule: Bottom-2 consistency	| **89.39%** |

&emsp;Detailed results and figures are available in the paper.

## 🚀 Getting Started
Prerequisites\
Python 3.8+\
Required packages: numpy, pandas, scipy, statsmodels, matplotlib, seaborn, scikit-learn



# 解码舞蹈：基于逆优化的隐藏粉丝投票估计与《与星共舞》公平评分机制设计
&emsp;本仓库包含论文 《解码舞蹈：基于逆优化的隐藏粉丝投票估计与〈与星共舞〉公平评分机制设计》的代码和补充材料，该论文提交至2026年美国大学生数学建模竞赛（MCM）。

## 📖 概述
&emsp;《与星共舞》（DWTS）将评委打分与未公开的粉丝投票相结合决定淘汰结果，常因技术弱但人气高的选手晋级而引发争议。由于粉丝投票总数从未公布，评估评分系统的公平性是一个根本性的推断挑战。\
&emsp;**我们开发了一个 统一的数据驱动框架，旨在：**\
&emsp;1、利用逆优化和时序平滑性，从淘汰结果中 重建潜在粉丝投票。\
&emsp;2、通过反事实分析 比较两种投票聚合规则（排名组合 vs 百分比组合）。\
&emsp;3、使用交叉随机效应混合模型 量化评委与粉丝的异质性影响因素。\
&emsp;4、设计一个基于熵权法和稳定性惩罚的 更公平的动态评分机制。\
&emsp;5、我们的方法在34季的历史淘汰结果中达到了 98.28% 的一致性，并为改善DWTS及类似竞赛系统的公平性提供了可行见解。

## 🔍 主要贡献
&emsp;提出了 逆优化框架，通过淘汰一致性约束估计隐藏投票份额。\
&emsp;· 反事实模拟 显示，在排名规则和百分比规则之间切换会改变 43.6% 的周次 淘汰结果。\
&emsp;· 混合效应模型 揭示评委更关注技术表现和年龄，而粉丝投票受职业搭档效应强烈影响。\
&emsp;· 设计了 新颖的动态加权机制（熵权法+稳定性惩罚），将倒数两名淘汰区的一致性提升至 89.39%，同时减少争议淘汰。

## 🧠 方法
&emsp;**我们依次处理四个研究任务：**\
&emsp;**任务1：粉丝投票估计（逆优化）**\
&emsp;构建约束优化问题：每周粉丝投票份额为受官方淘汰规则约束的潜变量。\
&emsp;添加正则项：(i) 与评委分数先验的最小偏差，(ii) 时序平滑性。\
&emsp;通过凸优化（百分比规则季）或可行性采样（排名规则季）求解。\
&emsp;通过蒙特卡洛集成（50次迭代，2%抖动）量化不确定性。

&emsp;**任务2：规则比较（反事实分析）**\
&emsp;固定推断出的粉丝投票，模拟在排名规则和百分比规则下的淘汰结果。\
&emsp;使用斯皮尔曼相关系数（最终排名与平均粉丝排名的相关性）衡量粉丝满意度。\
&emsp;评估评委拯救机制对争议案例的影响。

&emsp;**任务3：因素分析（交叉随机效应模型）**\
&emsp;评委打分模型： 线性混合模型，包含名人/搭档随机效应，固定效应（年龄、行业、周次、季次）。\
&emsp;粉丝投票对数份额模型： 对数转换后的投票份额，采用相同随机效应结构。\
&emsp;方差分解以分离技术评分与大众投票的驱动因素。

&emsp;**任务4：动态评分机制**\
&emsp;熵权法： 根据每周数据离散度动态分配评委和粉丝的权重。\
&emsp;稳定性惩罚： 惩罚滚动窗口内排名波动大的选手。\
&emsp;最终规则动态调整权重，减少短期噪声。

## 📊 结果摘要
| 指标 | 数值| 
| ----- | ----- | 
|重建粉丝投票一致性（总体）	      |                    98.28%|
|测试季一致性	                 |                       96.5%|
|外部验证（类似节目）|	~65–72%|
|排名规则与百分比规则结果不同的周次比例	|43.6%|
|评委拯救机制改变结果的周次（排名规则下）|	14.4%|
|评委拯救机制改变结果的周次（百分比规则下）|	42.5%|
|评委打分：年龄系数	|–1.55 (p < 0.001)|
|粉丝投票：年龄系数	|–0.50 (p < 0.01)|
|搭档效应方差（粉丝模型）	|约占随机效应总方差的 40%|
|新规则下倒数两名淘汰区一致性|	89.39%|

&emsp;详细结果和图表请参阅 论文。

## 🚀 快速开始
环境要求\
Python 3.8+\
依赖包：numpy, pandas, scipy, statsmodels, matplotlib, seaborn, scikit-learn
