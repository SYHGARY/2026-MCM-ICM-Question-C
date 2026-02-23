"""
DWTS Modeling System - MCM-Ready Implementation (Enhanced, Style-Aligned, FIXED)
Author: MCM Team
Date: 2026-01-31

Fixes:
1) Partner EB histogram empty -> dropna + empty-message fallback
2) Age effect subplot missing -> comparison_df fallback to model fixed_effects; otherwise show message
3) Keep academic plot style aligned with "uncertainty plot" style
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ====================== 全局学术样式配置（对齐“不确定性图表.py”） ======================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 11


# ==================== 0. UTILITIES ====================

def safe_mkdir(path: str):
    if path and (not os.path.exists(path)):
        os.makedirs(path, exist_ok=True)

def robust_zscore(x: pd.Series):
    mu = x.mean()
    sd = x.std(ddof=0)
    if (sd is None) or (sd <= 0) or (not np.isfinite(sd)):
        sd = 1.0
    return (x - mu) / sd

def format_p(p):
    try:
        if p < 1e-4:
            return "<1e-4"
        return f"{p:.4g}"
    except Exception:
        return str(p)


# ==================== 1. DATA INPUT AND TARGET STRUCTURE ====================

class DataPreprocessor:
    """
    输出：
      - df_common: 清洗后的基础表
      - df_score : 满足 Model 1 必需字段的子集
      - df_vote  : 满足 Model 2 必需字段的子集
    """

    def __init__(self, fan_votes_file, additional_data_file, out_dir="outputs"):
        self.fan_votes_file = fan_votes_file
        self.additional_data_file = additional_data_file
        self.out_dir = out_dir
        safe_mkdir(out_dir)

        self.df_base = None
        self.df_common = None
        self.df_score = None
        self.df_vote = None
        self.missing_report = None

    def load_and_merge_data(self):
        print("Step 1.1: Loading data...")

        df_votes = pd.read_csv(self.fan_votes_file)
        print(f"Fan votes data shape: {df_votes.shape}")

        # Align columns
        df_votes = df_votes.rename(columns={
            "fan_votes": "fan_vote_est",
            "judge_score": "judges_score"
        })

        print("Loading additional data...")
        df_additional = pd.read_csv(self.additional_data_file, encoding="utf-8")
        df_additional.columns = df_additional.columns.str.strip()

        if "celebrity_name" in df_additional.columns and "celebrity" not in df_additional.columns:
            df_additional = df_additional.rename(columns={"celebrity_name": "celebrity"})

        print(f"Additional data shape: {df_additional.shape}")

        print("Merging data on (celebrity, season)...")
        df_base = pd.merge(
            df_votes,
            df_additional,
            on=["celebrity", "season"],
            how="left"
        )

        print(f"Merged data shape: {df_base.shape}")
        self.df_base = df_base
        return self.df_base

    def clean_and_validate(self):
        print("\nStep 2.1: Cleaning and validating data...")

        df = self.df_base.copy()

        categorical_cols = [
            "celebrity",
            "ballroom_partner",
            "celebrity_industry",
            "celebrity_homecountry/region",
        ]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({"nan": np.nan, "": np.nan, "None": np.nan})

        for col in ["week", "judges_score", "fan_vote_est", "celebrity_age_during_season", "season"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        initial_rows = len(df)

        # week validation
        if "week" in df.columns:
            invalid_week = (df["week"] <= 0) | df["week"].isna()
            if invalid_week.any():
                print(f"  Found {invalid_week.sum()} invalid week rows (dropping)")
                df = df.loc[~invalid_week].copy()

        # age validation
        if "celebrity_age_during_season" in df.columns:
            invalid_age = (
                (df["celebrity_age_during_season"] < 10) |
                (df["celebrity_age_during_season"] > 100) |
                df["celebrity_age_during_season"].isna()
            )
            if invalid_age.any():
                print(f"  Found {invalid_age.sum()} invalid age rows (dropping)")
                df = df.loc[~invalid_age].copy()

        # judges_score validation
        if "judges_score" in df.columns:
            neg = (df["judges_score"] < 0) & (~df["judges_score"].isna())
            if neg.any():
                print(f"  Found {neg.sum()} negative judges_score rows (dropping)")
                df = df.loc[~neg].copy()

            valid_scores = df["judges_score"].dropna()
            if len(valid_scores) > 50:
                q1 = valid_scores.quantile(0.25)
                q3 = valid_scores.quantile(0.75)
                iqr = q3 - q1
                ub = valid_scores.quantile(0.999) + 3 * iqr
                extreme_hi = (df["judges_score"] > ub) & (~df["judges_score"].isna())
                if extreme_hi.any():
                    print(f"  Found {extreme_hi.sum()} extreme high judges_score rows above {ub:.2f} (dropping)")
                    df = df.loc[~extreme_hi].copy()

        print(f"  Removed {initial_rows - len(df)} rows due to validation")
        print(f"  Cleaned data shape: {df.shape}")

        self.df_common = df
        return self.df_common

    @staticmethod
    def _missing_report(df):
        missing_stats = pd.DataFrame({
            "Field": df.columns,
            "Missing_Count": df.isnull().sum(),
            "Missing_Rate": df.isnull().mean() * 100
        }).sort_values("Missing_Rate", ascending=False)
        return missing_stats

    def build_model_datasets(self):
        print("\nStep 2.2: Building model-specific datasets (split dropna)...")

        df = self.df_common.copy()
        self.missing_report = self._missing_report(df)
        miss_path = os.path.join(self.out_dir, "missing_value_report.csv")
        self.missing_report.to_csv(miss_path, index=False)
        print(f"  Missing report saved to: {miss_path}")

        common_required = [
            "celebrity", "season", "week",
            "ballroom_partner",
            "celebrity_industry",
            "celebrity_age_during_season"
        ]
        model1_required = common_required + ["judges_score"]
        model2_required = common_required + ["fan_vote_est", "celebrity_homecountry/region"]

        # Model 1 dataset
        df_score = df.copy()
        for col in model1_required:
            if col not in df_score.columns:
                df_score[col] = np.nan
        df_score = df_score.dropna(subset=model1_required).copy()
        if "judges_score" in df_score.columns:
            if (df_score["judges_score"] == 0).mean() > 0.01:
                df_score = df_score.loc[df_score["judges_score"] > 0].copy()

        # Model 2 dataset
        df_vote = df.copy()
        for col in model2_required:
            if col not in df_vote.columns:
                df_vote[col] = np.nan
        df_vote = df_vote.dropna(subset=model2_required).copy()
        df_vote = df_vote.loc[df_vote["fan_vote_est"] > 0].copy()

        print(f"  df_score rows (Model 1): {len(df_score)}")
        print(f"  df_vote rows  (Model 2): {len(df_vote)}")

        self.df_score = df_score
        self.df_vote = df_vote

        return self.df_score, self.df_vote, self.missing_report


# ==================== 2. FEATURE ENGINEERING ====================

class FeatureEngineer:
    """
    - age_z
    - week_cat, season_cat
    - industry consolidation
    - is_us robust
    """

    def __init__(self, df, out_dir="outputs"):
        self.df = df.copy()
        self.out_dir = out_dir
        safe_mkdir(out_dir)
        self.df_features = None

    @staticmethod
    def _normalize_country(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        s2 = (
            s.replace(".", "")
             .replace(",", "")
             .replace("(", "")
             .replace(")", "")
             .replace("-", " ")
        )
        s2 = " ".join(s2.split())
        return s2.upper()

    def create_derived_features(self, industry_min_count=None):
        print("\n" + "=" * 60)
        print("Step 2.3: Feature Engineering")
        print("=" * 60)

        df = self.df.copy()

        df["age_z"] = robust_zscore(df["celebrity_age_during_season"])

        df["week_cat"] = pd.Categorical(df["week"])
        df["season_cat"] = pd.Categorical(df["season"])

        df["celebrity_industry"] = df["celebrity_industry"].astype(str).str.strip()
        counts = df["celebrity_industry"].value_counts(dropna=True)

        if industry_min_count is None:
            industry_min_count = max(10, int(0.01 * len(df)))
        keep = set(counts[counts >= industry_min_count].index.tolist())
        df["celebrity_industry_consolidated"] = df["celebrity_industry"].apply(lambda x: x if x in keep else "Other")

        us_aliases = {
            "UNITED STATES",
            "UNITED STATES OF AMERICA",
            "USA",
            "US",
            "U S",
            "U S A",
        }
        if "celebrity_homecountry/region" in df.columns:
            df["country_norm"] = df["celebrity_homecountry/region"].apply(self._normalize_country)
            df["is_us"] = df["country_norm"].apply(lambda s: 1 if (isinstance(s, str) and s in us_aliases) else 0)
        else:
            df["country_norm"] = np.nan
            df["is_us"] = 0

        self.df_features = df

        out_path = os.path.join(self.out_dir, "processed_features.csv")
        df.to_csv(out_path, index=False)
        print(f"Feature engineering complete. Shape: {df.shape}")
        print(f"  Features saved to: {out_path}")
        print(f"  industry_min_count used: {industry_min_count}")

        return self.df_features


# ==================== 3. MIXEDLM BASE (AUTO FALLBACK) ====================

class MixedLMRunner:
    def __init__(self, df, y_col, full_formula_rhs, simple_formula_rhs,
                 group_col="celebrity", partner_col="ballroom_partner",
                 maxiter=2000, method="lbfgs"):
        self.df = df.copy()
        self.y_col = y_col
        self.full_formula_rhs = full_formula_rhs
        self.simple_formula_rhs = simple_formula_rhs
        self.group_col = group_col
        self.partner_col = partner_col
        self.maxiter = maxiter
        self.method = method

        self.results = None
        self.used_spec = None

    def _fit_once(self, rhs: str):
        formula = f"{self.y_col} ~ {rhs}"
        vc = {"partner": f"0 + C({self.partner_col})"}
        md = smf.mixedlm(formula, self.df, groups=self.df[self.group_col],
                         re_formula="1", vc_formula=vc)
        res = md.fit(method=self.method, maxiter=self.maxiter)
        return res

    def fit(self):
        try:
            res = self._fit_once(self.full_formula_rhs)
            if getattr(res, "converged", False):
                self.results = res
                self.used_spec = "FULL"
                return res
            else:
                print("  FULL spec did not converge, falling back to SIMPLE...")
        except Exception as e:
            print(f"  FULL spec fitting error: {e}")
            print("  Falling back to SIMPLE...")

        try:
            res2 = self._fit_once(self.simple_formula_rhs)
            self.results = res2
            self.used_spec = "SIMPLE"
            return res2
        except Exception as e2:
            print(f"  SIMPLE spec fitting error: {e2}")
            self.results = None
            self.used_spec = None
            return None


def extract_mixedlm_outputs(res, df_used, y_col, out_prefix, out_dir="outputs"):
    safe_mkdir(out_dir)
    if res is None:
        return None

    fe = pd.DataFrame({
        "Coefficient": res.fe_params,
        "Std_Error": res.bse_fe,
        "Z_Value": res.tvalues,
        "P_Value": res.pvalues
    })

    try:
        ci = res.conf_int()
        fe["CI_Lower"] = ci.iloc[:, 0]
        fe["CI_Upper"] = ci.iloc[:, 1]
    except Exception:
        fe["CI_Lower"] = fe["Coefficient"] - 1.96 * fe["Std_Error"]
        fe["CI_Upper"] = fe["Coefficient"] + 1.96 * fe["Std_Error"]

    var_celebrity = float(res.cov_re.iloc[0, 0]) if hasattr(res, "cov_re") else np.nan
    var_partner = float(res.vcomp[0]) if hasattr(res, "vcomp") and len(res.vcomp) > 0 else np.nan
    var_resid = float(res.scale) if hasattr(res, "scale") else np.nan

    var_comp = pd.Series({
        "Celebrity_Var": var_celebrity,
        "Partner_Var": var_partner,
        "Residual_Var": var_resid
    })

    df2 = df_used.copy()
    df2["fitted"] = res.fittedvalues
    df2["residual"] = df2[y_col] - df2["fitted"]

    g = df2.groupby("ballroom_partner")["residual"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "Raw_Mean", "std": "Raw_SD", "count": "N"}
    )

    vg = var_partner if np.isfinite(var_partner) else 0.0
    ve = var_resid if np.isfinite(var_resid) else 1.0

    denom = (vg + (ve / g["N"].clip(lower=1)))
    g["Shrink_Factor"] = np.where(denom > 0, vg / denom, np.nan)
    g["Effect_EB"] = g["Raw_Mean"] * g["Shrink_Factor"]

    partner_effects = g.sort_values("Effect_EB", ascending=False)

    diagnostics = pd.Series({
        "AIC": getattr(res, "aic", np.nan),
        "BIC": getattr(res, "bic", np.nan),
        "LLF": getattr(res, "llf", np.nan),
        "Residual_Mean": df2["residual"].mean(),
        "Residual_SD": df2["residual"].std(),
        "Converged": getattr(res, "converged", np.nan),
    })

    fe.to_csv(os.path.join(out_dir, f"{out_prefix}_fixed_effects.csv"))
    partner_effects.to_csv(os.path.join(out_dir, f"{out_prefix}_partner_effects.csv"))
    var_comp.to_csv(os.path.join(out_dir, f"{out_prefix}_variance_components.csv"))
    diagnostics.to_csv(os.path.join(out_dir, f"{out_prefix}_diagnostics.csv"))

    return {
        "fixed_effects": fe,
        "partner_effects": partner_effects,
        "variance_components": var_comp,
        "diagnostics": diagnostics,
        "model_summary": str(res.summary()),
    }


# ==================== 4. MODEL 1 ====================

class JudgesScoreLMM:
    def __init__(self, df_features, out_dir="outputs"):
        self.df = df_features.copy()
        self.out_dir = out_dir
        safe_mkdir(out_dir)
        self.used_spec = None
        self.results = None

    def prepare_data(self):
        required = [
            "celebrity", "ballroom_partner", "judges_score",
            "age_z", "celebrity_industry_consolidated",
            "week_cat", "season_cat"
        ]
        df = self.df[required].dropna().copy()
        df["celebrity"] = df["celebrity"].astype(str)
        df["ballroom_partner"] = df["ballroom_partner"].astype(str)

        print("\n" + "=" * 60)
        print("Model 1: Judges Score MixedLM - Data Preparation")
        print("=" * 60)
        print(f"  Sample size: {len(df)}")
        return df

    def fit(self, df):
        print("\nFitting Model 1 MixedLM with auto-robust fallback...")

        full_rhs = "age_z + C(celebrity_industry_consolidated) + C(week_cat) + C(season_cat)"
        simple_rhs = "age_z + C(celebrity_industry_consolidated) + C(week_cat)"

        runner = MixedLMRunner(
            df=df,
            y_col="judges_score",
            full_formula_rhs=full_rhs,
            simple_formula_rhs=simple_rhs,
            group_col="celebrity",
            partner_col="ballroom_partner"
        )
        res = runner.fit()
        self.used_spec = runner.used_spec

        if res is None:
            print("  Model 1 failed to fit.")
            self.results = None
            return None

        self.results = extract_mixedlm_outputs(
            res=res,
            df_used=df,
            y_col="judges_score",
            out_prefix="model1",
            out_dir=self.out_dir
        )
        return self.results


# ==================== 5. MODEL 2 ====================

class FanVoteMixedModel:
    def __init__(self, df_features, out_dir="outputs"):
        self.df = df_features.copy()
        self.out_dir = out_dir
        safe_mkdir(out_dir)
        self.used_spec = None
        self.results = None
        self.share_method_used = None

    @staticmethod
    def _share_linear(group, col="fan_vote_est"):
        x = group[col].values.astype(float)
        s = np.nansum(x)
        if (s is None) or (s <= 0) or (not np.isfinite(s)):
            group["vote_share"] = np.nan
        else:
            group["vote_share"] = x / s
        return group

    @staticmethod
    def _share_softmax(group, col="fan_vote_est"):
        x = group[col].values.astype(float)
        x = x - np.nanmax(x)
        ex = np.exp(x)
        denom = np.nansum(ex)
        if (denom is None) or (denom <= 0) or (not np.isfinite(denom)):
            group["vote_share"] = np.nan
        else:
            group["vote_share"] = ex / denom
        return group

    def prepare_vote_data(self, share_method="linear", eps=1e-4):
        print("\n" + "=" * 60)
        print(f"Model 2: Vote Logit-Share MixedLM - Data Preparation (share_method={share_method})")
        print("=" * 60)

        df = self.df.copy()

        if share_method == "softmax":
            df = df.groupby(["season", "week"], group_keys=False).apply(self._share_softmax).reset_index(drop=True)
        else:
            df = df.groupby(["season", "week"], group_keys=False).apply(self._share_linear).reset_index(drop=True)

        df["vote_beta"] = df["vote_share"].clip(eps, 1 - eps)
        df["vote_logit"] = np.log(df["vote_beta"] / (1 - df["vote_beta"]))

        required = [
            "celebrity", "ballroom_partner",
            "vote_logit",
            "age_z", "celebrity_industry_consolidated",
            "is_us",
            "week_cat", "season_cat"
        ]
        df_vote = df[required].dropna().copy()
        df_vote["celebrity"] = df_vote["celebrity"].astype(str)
        df_vote["ballroom_partner"] = df_vote["ballroom_partner"].astype(str)

        self.share_method_used = share_method
        return df_vote

    def fit(self, df_vote, out_prefix="model2"):
        print("\nFitting Model 2 MixedLM (auto-robust fallback)...")

        full_rhs = "age_z + C(celebrity_industry_consolidated) + is_us + C(week_cat) + C(season_cat)"
        simple_rhs = "age_z + C(celebrity_industry_consolidated) + is_us + C(week_cat)"

        runner = MixedLMRunner(
            df=df_vote,
            y_col="vote_logit",
            full_formula_rhs=full_rhs,
            simple_formula_rhs=simple_rhs,
            group_col="celebrity",
            partner_col="ballroom_partner"
        )
        res = runner.fit()
        self.used_spec = runner.used_spec

        if res is None:
            print("  Model 2 failed to fit.")
            self.results = None
            return None

        self.results = extract_mixedlm_outputs(
            res=res,
            df_used=df_vote,
            y_col="vote_logit",
            out_prefix=out_prefix,
            out_dir=self.out_dir
        )
        return self.results


# ==================== 6. MODEL COMPARISON ====================

class ModelComparator:
    def __init__(self, out_dir="outputs"):
        self.out_dir = out_dir
        safe_mkdir(out_dir)

    @staticmethod
    def _pick_coef(fe_df, name):
        if fe_df is None or name not in fe_df.index:
            return (np.nan, np.nan, np.nan)
        return (
            float(fe_df.loc[name, "Coefficient"]),
            float(fe_df.loc[name, "Std_Error"]),
            float(fe_df.loc[name, "P_Value"]),
        )

    def compare_two_models(self, m1_results, m2_results, out_name="model_comparison_table.csv"):
        # 永远返回一个 df（哪怕是 NaN），避免 Age 图“消失”
        rows = []
        fe1 = m1_results["fixed_effects"] if m1_results else None
        fe2 = m2_results["fixed_effects"] if m2_results else None

        for var in ["Intercept", "age_z"]:
            c1, se1, p1 = self._pick_coef(fe1, var)
            c2, se2, p2 = self._pick_coef(fe2, var)
            rows.append({
                "Variable": var,
                "Model1_Coef": c1, "Model1_SE": se1, "Model1_P": p1,
                "Model2_Coef": c2, "Model2_SE": se2, "Model2_P": p2,
                "Difference": c2 - c1
            })

        df = pd.DataFrame(rows)
        out_path = os.path.join(self.out_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"\nComparison table saved to: {out_path}")
        return df

    def compare_variance(self, m1_results, m2_results, out_name="variance_comparison.csv"):
        if m1_results is None or m2_results is None:
            return None

        v1 = m1_results["variance_components"]
        v2 = m2_results["variance_components"]
        out = pd.DataFrame([{
            "Model": "JudgesScore",
            "Celebrity_Var": v1.get("Celebrity_Var", np.nan),
            "Partner_Var": v1.get("Partner_Var", np.nan),
            "Residual_Var": v1.get("Residual_Var", np.nan)
        }, {
            "Model": "VoteLogitShare",
            "Celebrity_Var": v2.get("Celebrity_Var", np.nan),
            "Partner_Var": v2.get("Partner_Var", np.nan),
            "Residual_Var": v2.get("Residual_Var", np.nan)
        }])
        out_path = os.path.join(self.out_dir, out_name)
        out.to_csv(out_path, index=False)
        print(f"Variance comparison saved to: {out_path}")
        return out


# ==================== 7. VISUALIZATION (FIXED) ====================

class VisualizationModule:
    @staticmethod
    def create_plots(model1_results, model2_results, comparison_df=None, out_dir="outputs",
                     fig_name="model_results_visualization.png"):
        safe_mkdir(out_dir)

        primary_color   = '#1f77b4'  # blue
        secondary_color = '#ff7f0e'  # orange
        accent_color    = '#d62728'  # red

        def stylize(ax):
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
            ax.tick_params(direction='in', width=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

        # -------- (1) Age effect --------
        ax1 = fig.add_subplot(gs[0, 0])
        stylize(ax1)

        def pick_age_from_results(mres):
            if (mres is None) or ("fixed_effects" not in mres):
                return (np.nan, np.nan)
            fe = mres["fixed_effects"]
            if "age_z" not in fe.index:
                return (np.nan, np.nan)
            return (float(fe.loc["age_z", "Coefficient"]), float(fe.loc["age_z", "Std_Error"]))

        m1_c, m1_se = (np.nan, np.nan)
        m2_c, m2_se = (np.nan, np.nan)

        if (comparison_df is not None) and (len(comparison_df) > 0) and ("age_z" in comparison_df["Variable"].values):
            r = comparison_df.loc[comparison_df["Variable"] == "age_z"].iloc[0]
            m1_c, m1_se = r.get("Model1_Coef", np.nan), r.get("Model1_SE", np.nan)
            m2_c, m2_se = r.get("Model2_Coef", np.nan), r.get("Model2_SE", np.nan)

        if (not np.isfinite(m1_c)) or (not np.isfinite(m1_se)):
            m1_c, m1_se = pick_age_from_results(model1_results)
        if (not np.isfinite(m2_c)) or (not np.isfinite(m2_se)):
            m2_c, m2_se = pick_age_from_results(model2_results)

        if np.isfinite(m1_c) or np.isfinite(m2_c):
            ax1.bar(
                ["JudgesScore", "VoteLogitShare"],
                [m1_c, m2_c],
                yerr=[m1_se if np.isfinite(m1_se) else 0.0, m2_se if np.isfinite(m2_se) else 0.0],
                capsize=6,
                alpha=0.85,
                color=[primary_color, accent_color],
                edgecolor="white",
                linewidth=1.2
            )
            ax1.axhline(0, linestyle="--", alpha=0.5, linewidth=1.0, color="black")
        else:
            ax1.text(0.5, 0.5, "Age effect not available\n(model dropped / not estimated).",
                     ha="center", va="center", transform=ax1.transAxes, fontweight="bold")

        ax1.set_title("Age Effect (coef ± SE)", fontweight="bold", fontsize=13)
        ax1.set_ylabel("Coefficient", fontweight="bold")

        # -------- (2) Partner EB histogram (FIX) --------
        ax2 = fig.add_subplot(gs[0, 1])
        stylize(ax2)

        pe1 = np.array([])
        pe2 = np.array([])

        if model1_results and ("partner_effects" in model1_results):
            s1 = model1_results["partner_effects"].get("Effect_EB", pd.Series(dtype=float))
            pe1 = pd.Series(s1).dropna().values

        if model2_results and ("partner_effects" in model2_results):
            s2 = model2_results["partner_effects"].get("Effect_EB", pd.Series(dtype=float))
            pe2 = pd.Series(s2).dropna().values

        if len(pe1) > 0:
            ax2.hist(pe1, bins=20, alpha=0.6, density=True,
                     color=primary_color, edgecolor="white", linewidth=1.0,
                     label="JudgesScore")
        if len(pe2) > 0:
            ax2.hist(pe2, bins=20, alpha=0.6, density=True,
                     color=secondary_color, edgecolor="white", linewidth=1.0,
                     label="VoteLogitShare")

        ax2.set_title("Partner EB Effects Distribution", fontweight="bold", fontsize=13)
        ax2.set_xlabel("EB Effect", fontweight="bold")
        ax2.set_ylabel("Density", fontweight="bold")

        if (len(pe1) == 0) and (len(pe2) == 0):
            ax2.text(0.5, 0.5, "No valid EB effects (all NaN or empty).",
                     ha="center", va="center", transform=ax2.transAxes, fontweight="bold")
        else:
            ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

        # -------- (3) Variance components --------
        ax3 = fig.add_subplot(gs[1, :])
        stylize(ax3)

        if model1_results and model2_results:
            v1 = model1_results["variance_components"]
            v2 = model2_results["variance_components"]
            cats = ["Celebrity_Var", "Partner_Var", "Residual_Var"]
            x = np.arange(len(cats))
            w = 0.36

            ax3.bar(x - w/2, [v1.get(c, np.nan) for c in cats], w,
                    alpha=0.85, color=primary_color, edgecolor="white", linewidth=1.2,
                    label="JudgesScore")
            ax3.bar(x + w/2, [v2.get(c, np.nan) for c in cats], w,
                    alpha=0.85, color=accent_color, edgecolor="white", linewidth=1.2,
                    label="VoteLogitShare")

            ax3.set_xticks(x)
            ax3.set_xticklabels(cats, fontweight="bold")
            ax3.set_title("Variance Components Comparison", fontweight="bold", fontsize=13)
            ax3.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
        else:
            ax3.text(0.5, 0.5, "Variance components not available.",
                     ha="center", va="center", transform=ax3.transAxes, fontweight="bold")
            ax3.set_title("Variance Components Comparison", fontweight="bold", fontsize=13)

        plt.suptitle("DWTS Modeling Results (MCM-Ready Enhanced)", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_path = os.path.join(out_dir, fig_name)
        plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Visualization saved to: {out_path}")
        plt.show()


# ==================== 8. SENSITIVITY ====================

def run_model2_share_sensitivity(fe_vote, out_dir="outputs"):
    rows = []
    for method in ["linear", "softmax"]:
        m2 = FanVoteMixedModel(fe_vote, out_dir=out_dir)
        df_vote = m2.prepare_vote_data(share_method=method)
        res = m2.fit(df_vote, out_prefix=f"model2_{method}")

        if res is None:
            rows.append({"share_method": method, "SpecUsed": m2.used_spec, "Converged": np.nan, "AIC": np.nan})
            continue

        diag = res["diagnostics"]
        rows.append({
            "share_method": method,
            "SpecUsed": m2.used_spec,
            "Converged": diag.get("Converged", np.nan),
            "AIC": diag.get("AIC", np.nan),
        })

    df_sens = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "model2_sensitivity_share_method.csv")
    df_sens.to_csv(out_path, index=False)
    print(f"\n[Sensitivity] share_method comparison saved to: {out_path}")
    return df_sens


# ==================== 9. MAIN ====================

def main():
    print("=" * 80)
    print("DWTS MODELING SYSTEM - MCM-READY ENHANCED IMPLEMENTATION (STYLE-ALIGNED, FIXED)")
    print("=" * 80)

    fan_votes_file = "optimized_fan_votes.csv"
    additional_data_file = "2026_MCM_Problem_C_Data.csv"
    out_dir = "outputs"
    safe_mkdir(out_dir)

    try:
        pre = DataPreprocessor(fan_votes_file, additional_data_file, out_dir=out_dir)
        _ = pre.load_and_merge_data()
        _ = pre.clean_and_validate()
        df_score_raw, df_vote_raw, _ = pre.build_model_datasets()

        fe_score = FeatureEngineer(df_score_raw, out_dir=out_dir).create_derived_features(industry_min_count=None)
        fe_vote = FeatureEngineer(df_vote_raw, out_dir=out_dir).create_derived_features(industry_min_count=None)

        # Model 1
        m1 = JudgesScoreLMM(fe_score, out_dir=out_dir)
        df_score = m1.prepare_data()
        model1_results = m1.fit(df_score)

        # Model 2 main
        m2_main = FanVoteMixedModel(fe_vote, out_dir=out_dir)
        df_vote_main = m2_main.prepare_vote_data(share_method="linear")
        model2_results = m2_main.fit(df_vote_main, out_prefix="model2_main_linear")

        # Sensitivity
        _ = run_model2_share_sensitivity(fe_vote, out_dir=out_dir)

        # Compare
        comp = ModelComparator(out_dir=out_dir)
        comparison_df = comp.compare_two_models(model1_results, model2_results, out_name="model_comparison_table.csv")
        _ = comp.compare_variance(model1_results, model2_results, out_name="variance_comparison.csv")

        # Visualization
        VisualizationModule.create_plots(model1_results, model2_results, comparison_df, out_dir=out_dir)

        return True

    except Exception as e:
        print(f"\nError in execution: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", 20)
    main()
