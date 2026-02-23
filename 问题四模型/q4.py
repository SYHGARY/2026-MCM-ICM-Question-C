# q4_full_solution_v3.py
# ------------------------------------------------------------
# Problem 4: Fairer fairness system
# Core: Entropy dynamic weights + small-sample inheritance + EMA smoothing
# Deepening (v3): Add "stability penalty" for elimination ranking only
#   adjusted_score = combined_score - lambda_penalty * rank_volatility
#
# Inputs:
#   /mnt/data/2026_MCM_Problem_C_Data.csv
#   /mnt/data/optimized_fan_votes.csv
#
# Outputs (out_dir):
#   - combined_scores_long.csv
#   - weights_by_season_week.csv
#   - predicted_vs_actual_elimination_base.csv
#   - predicted_vs_actual_elimination_adjusted.csv
#   - elimination_consistency_by_season.csv
#   - stability_kfold_weights.csv
# ------------------------------------------------------------

import os
import re
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ----------------------------
# Utilities
# ----------------------------
def safe_minmax(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    denom = x.max() - x.min()
    if pd.isna(denom) or abs(denom) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.min()) / (denom + 1e-12)


def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_eliminated_week(results_str: str) -> Optional[int]:
    if pd.isna(results_str):
        return None
    m = re.search(r"Eliminated\s+Week\s+(\d+)", str(results_str), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


# ----------------------------
# Entropy weights
# ----------------------------
def entropy_weights(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    X = df[cols].copy()
    for c in cols:
        X[c] = safe_minmax(X[c])

    col_sums = X.sum(axis=0).replace(0.0, np.nan)
    P = X.div(col_sums, axis=1).fillna(0.0)

    m = len(P)
    if m <= 1:
        return {c: 1.0 / len(cols) for c in cols}

    k = 1.0 / np.log(m)
    eps = 1e-12
    E = -k * (P * np.log(P + eps)).sum(axis=0)

    d = 1.0 - E
    if float(d.sum()) <= 1e-12 or np.isnan(d.sum()):
        return {c: 1.0 / len(cols) for c in cols}

    w = (d / d.sum()).to_dict()
    return {c: float(w[c]) for c in cols}


def compute_scores_with_weights(g: pd.DataFrame, w: Dict[str, float]) -> pd.DataFrame:
    out = g.copy()
    out["judge_norm"] = safe_minmax(out["judge_total"])
    out["fan_norm"] = safe_minmax(out["fan_vote_est"])
    out["w_judge"] = float(w["judge_total"])
    out["w_fan"] = float(w["fan_vote_est"])
    out["combined_score"] = out["w_judge"] * out["judge_norm"] + out["w_fan"] * out["fan_norm"]
    out["combined_rank"] = out["combined_score"].rank(ascending=False, method="min").astype(int)
    return out


# ----------------------------
# Data reshaping
# ----------------------------
def melt_judges_wide_to_long(judges_df: pd.DataFrame) -> pd.DataFrame:
    df = judges_df.copy()
    pattern = re.compile(r"^week(\d+)_judge(\d+)_score$", flags=re.IGNORECASE)
    score_cols = [c for c in df.columns if pattern.match(c)]
    if not score_cols:
        raise ValueError("No columns like weekX_judgeY_score found in judges CSV.")

    long = df.melt(
        id_vars=["season", "celebrity_name", "results", "placement"],
        value_vars=score_cols,
        var_name="week_judge",
        value_name="judge_score"
    )
    long["week"] = long["week_judge"].str.extract(r"week(\d+)_", expand=False).astype(int)
    long["judge_score"] = pd.to_numeric(long["judge_score"], errors="coerce")

    agg = (
        long.groupby(["season", "celebrity_name", "results", "placement", "week"], as_index=False)["judge_score"]
        .sum()
        .rename(columns={"judge_score": "judge_total"})
    )

    has_score = (
        long.groupby(["season", "celebrity_name", "week"])["judge_score"]
        .apply(lambda s: np.any(~pd.isna(s)))
        .reset_index()
        .rename(columns={"judge_score": "has_score"})
    )
    agg = agg.merge(has_score, on=["season", "celebrity_name", "week"], how="left")
    agg.loc[agg["has_score"] == False, "judge_total"] = np.nan

    agg["eliminated_week_actual"] = agg["results"].apply(extract_eliminated_week)
    agg["celebrity_key"] = agg["celebrity_name"].apply(normalize_name)
    return agg


def prep_fan_votes(fan_df: pd.DataFrame) -> pd.DataFrame:
    df = fan_df.copy()
    required = {"season", "week", "celebrity", "fan_vote_est"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fan votes CSV missing columns: {missing}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype(int)
    df["celebrity_key"] = df["celebrity"].apply(normalize_name)
    return df[["season", "week", "celebrity", "celebrity_key", "fan_vote_est"]].copy()


# ----------------------------
# Evaluation
# ----------------------------
def predict_elimination(scored_long: pd.DataFrame, score_col: str, bottom_k: int = 1) -> pd.DataFrame:
    df = scored_long.copy()
    df = df[(df["judge_total"].notna()) & (df["judge_total"] > 0)].copy()

    def bottomk_names(g: pd.DataFrame) -> List[str]:
        g2 = g.sort_values(score_col, ascending=True)
        return list(g2["celebrity_name"].head(bottom_k).values)

    pred = (
        df.groupby(["season", "week"], as_index=False)
        .apply(lambda g: pd.Series({"pred_bottomk": bottomk_names(g)}))
        .reset_index(drop=True)
    )

    actual = df[df["eliminated_week_actual"].notna()].copy()
    actual = actual[actual["eliminated_week_actual"] == actual["week"]]
    actual_grouped = (
        actual.groupby(["season", "week"])["celebrity_name"]
        .apply(lambda s: sorted(list(s.unique())))
        .reset_index()
        .rename(columns={"celebrity_name": "actual_eliminated_list"})
    )

    out = pred.merge(actual_grouped, on=["season", "week"], how="left")
    out["match"] = out.apply(
        lambda r: (
            isinstance(r["actual_eliminated_list"], list)
            and any(p in r["actual_eliminated_list"] for p in (r["pred_bottomk"] or []))
        ),
        axis=1
    )
    out["bottom_k"] = bottom_k
    out["score_col"] = score_col
    return out


def consistency_by_season(pred_vs_actual: pd.DataFrame) -> pd.DataFrame:
    tmp = pred_vs_actual.copy()
    tmp = tmp[tmp["actual_eliminated_list"].notna()].copy()
    summary = (
        tmp.groupby(["season", "bottom_k", "score_col"])["match"]
        .agg(weeks_with_elimination="count", matched="sum")
        .reset_index()
    )
    summary["consistency"] = summary["matched"] / summary["weeks_with_elimination"].replace(0, np.nan)
    return summary


def kfold_weight_stability(scored_long: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    df = scored_long.copy()
    df = df[(df["judge_total"].notna()) & (df["judge_total"] > 0)].copy()

    rows = []
    for (season, week), g in df.groupby(["season", "week"]):
        if len(g) < max(6, n_splits * 2):
            rows.append({
                "season": season, "week": week, "n": len(g),
                "w_judge_mean": np.nan, "w_judge_std": np.nan,
                "w_fan_mean": np.nan, "w_fan_std": np.nan
            })
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        ws = []
        for train_idx, _ in kf.split(g):
            train = g.iloc[train_idx]
            w = entropy_weights(train, ["judge_total", "fan_vote_est"])
            ws.append([w["judge_total"], w["fan_vote_est"]])

        ws = np.array(ws)
        rows.append({
            "season": season, "week": week, "n": len(g),
            "w_judge_mean": ws[:, 0].mean(), "w_judge_std": ws[:, 0].std(ddof=1),
            "w_fan_mean": ws[:, 1].mean(), "w_fan_std": ws[:, 1].std(ddof=1),
        })
    return pd.DataFrame(rows)


def sensitivity_analysis(scored_long: pd.DataFrame, delta: float = 0.05) -> pd.DataFrame:
    df = scored_long.copy()
    df = df[(df["judge_total"].notna()) & (df["judge_total"] > 0)].copy()

    base = df["combined_score"].values
    wj = df["w_judge"].values
    wf = df["w_fan"].values

    wf2 = np.clip(wf + delta, 0.0, 1.0)
    wj2 = np.clip(wj - delta, 0.0, 1.0)
    s = wf2 + wj2
    s[s == 0] = 1.0
    wf2 /= s
    wj2 /= s

    perturbed = wj2 * df["judge_norm"].values + wf2 * df["fan_norm"].values
    df["abs_change"] = np.abs(perturbed - base)

    out = (
        df.groupby(["season", "week"], as_index=False)["abs_change"]
        .mean()
        .rename(columns={"abs_change": f"mean_abs_change_delta_{delta}"})
    )
    return out


# ----------------------------
# Main pipeline
# ----------------------------
@dataclass
class Paths:
    judges_csv: str
    fan_csv: str
    out_dir: str


def run_pipeline(paths: Paths, alpha: float, min_n: int, kfold: int, delta: float,
                 lambda_penalty: float, vol_window: int) -> None:
    judges_df = pd.read_csv(paths.judges_csv)
    fan_df = pd.read_csv(paths.fan_csv)

    judges_long = melt_judges_wide_to_long(judges_df)
    fan_long = prep_fan_votes(fan_df)

    merged = judges_long.merge(
        fan_long[["season", "week", "celebrity_key", "fan_vote_est"]],
        on=["season", "week", "celebrity_key"],
        how="left"
    )

    # Coverage only on active weeks
    active = merged[(merged["judge_total"].notna()) & (merged["judge_total"] > 0)].copy()
    coverage_active = active["fan_vote_est"].notna().mean()
    print(f"[INFO] Fan-vote merge coverage (active weeks only): {coverage_active:.2%}")
    if coverage_active < 0.95:
        unmatched = active[active["fan_vote_est"].isna()][["season", "week", "celebrity_name"]].head(20)
        print("[WARN] Missing fan_vote_est among active weeks (likely name mismatch). Examples:")
        print(unmatched.to_string(index=False))

    data = merged.dropna(subset=["judge_total", "fan_vote_est"]).copy()

    scored_parts = []
    weights_rows = []

    for season, season_df in data.groupby("season"):
        prev_w = None

        for week, g in season_df.groupby("week"):
            g_valid = g[g["judge_total"] > 0].copy()
            if len(g_valid) < 2:
                continue

            if len(g_valid) < min_n and prev_w is not None:
                w_use = prev_w
                used_inherit = True
            else:
                w_raw = entropy_weights(g_valid, ["judge_total", "fan_vote_est"])
                if prev_w is None:
                    w_use = w_raw
                else:
                    w_use = {
                        "judge_total": alpha * w_raw["judge_total"] + (1 - alpha) * prev_w["judge_total"],
                        "fan_vote_est": alpha * w_raw["fan_vote_est"] + (1 - alpha) * prev_w["fan_vote_est"],
                    }
                used_inherit = False

            scored_g = compute_scores_with_weights(g_valid, w_use)
            scored_parts.append(scored_g)

            weights_rows.append({
                "season": int(season),
                "week": int(week),
                "w_judge": float(w_use["judge_total"]),
                "w_fan": float(w_use["fan_vote_est"]),
                "n_contestants": int(len(g_valid)),
                "used_inherit": used_inherit,
            })

            prev_w = w_use

    if not scored_parts:
        raise RuntimeError("No scored data produced. Check merge keys / input files.")

    scored_long = pd.concat(scored_parts, ignore_index=True)
    weights_by_week = pd.DataFrame(weights_rows)

    # ----------------------------
    # Deepening: stability penalty (volatility)
    # We compute rolling std of combined_rank within each season-celebrity.
    # If vol_window <= 0 -> use std over all available weeks.
    # ----------------------------
    scored_long = scored_long.sort_values(["season", "celebrity_name", "week"]).copy()

    if vol_window and vol_window > 1:
        scored_long["rank_volatility"] = (
            scored_long.groupby(["season", "celebrity_name"])["combined_rank"]
            .transform(lambda s: s.rolling(vol_window, min_periods=2).std())
            .fillna(0.0)
        )
    else:
        scored_long["rank_volatility"] = (
            scored_long.groupby(["season", "celebrity_name"])["combined_rank"]
            .transform(lambda s: float(np.nanstd(s, ddof=1)) if len(s) > 1 else 0.0)
            .fillna(0.0)
        )

    scored_long["adjusted_score"] = scored_long["combined_score"] - lambda_penalty * scored_long["rank_volatility"]
    scored_long["adjusted_rank"] = scored_long["adjusted_score"].rank(ascending=False, method="min").astype(int)

    # Sensitivity + stability
    stability = kfold_weight_stability(scored_long, n_splits=kfold)
    sens = sensitivity_analysis(scored_long, delta=delta)
    weights_plus = weights_by_week.merge(sens, on=["season", "week"], how="left")

    # Evaluate elimination consistency for both base and adjusted, bottom-k = 1..3
    preds = []
    for score_col in ["combined_score", "adjusted_score"]:
        for k in [1, 2, 3]:
            preds.append(predict_elimination(scored_long, score_col=score_col, bottom_k=k))
    pred_all = pd.concat(preds, ignore_index=True)
    consistency = consistency_by_season(pred_all)

    # Save outputs
    os.makedirs(paths.out_dir, exist_ok=True)

    scored_path = os.path.join(paths.out_dir, "combined_scores_long.csv")
    weights_path = os.path.join(paths.out_dir, "weights_by_season_week.csv")
    base_pred_path = os.path.join(paths.out_dir, "predicted_vs_actual_elimination_base.csv")
    adj_pred_path = os.path.join(paths.out_dir, "predicted_vs_actual_elimination_adjusted.csv")
    cons_path = os.path.join(paths.out_dir, "elimination_consistency_by_season.csv")
    stab_path = os.path.join(paths.out_dir, "stability_kfold_weights.csv")

    scored_long.to_csv(scored_path, index=False, encoding="utf-8-sig")
    weights_plus.to_csv(weights_path, index=False, encoding="utf-8-sig")
    pred_all[pred_all["score_col"] == "combined_score"].to_csv(base_pred_path, index=False, encoding="utf-8-sig")
    pred_all[pred_all["score_col"] == "adjusted_score"].to_csv(adj_pred_path, index=False, encoding="utf-8-sig")
    consistency.to_csv(cons_path, index=False, encoding="utf-8-sig")
    stability.to_csv(stab_path, index=False, encoding="utf-8-sig")

    # Print summary
    print("\n[RESULT] Output files written to:", paths.out_dir)
    print(" -", scored_path)
    print(" -", weights_path)
    print(" -", base_pred_path)
    print(" -", adj_pred_path)
    print(" -", cons_path)
    print(" -", stab_path)

    # Overall match rates
    tmp = pred_all[pred_all["actual_eliminated_list"].notna()].copy()
    if len(tmp) == 0:
        print("\n[RESULT] No elimination weeks detected (check parsing of results).")
    else:
        print("\n[RESULT] Overall elimination consistency (weeks with elimination):")
        for score_col in ["combined_score", "adjusted_score"]:
            for k in [1, 2, 3]:
                t = tmp[(tmp["score_col"] == score_col) & (tmp["bottom_k"] == k)]
                print(f"   {score_col:14s} bottom-{k}: {t['match'].sum()} / {len(t)} = {t['match'].mean():.2%}")

    print("\n[RESULT] Weights head:")
    print(weights_plus.sort_values(["season", "week"]).head(12).to_string(index=False))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Problem 4 full pipeline v3 (entropy + smoothing + stability penalty).")
    p.add_argument("--judges_csv", type=str, default="2026_MCM_Problem_C_Data.csv")
    p.add_argument("--fan_csv", type=str, default="optimized_fan_votes.csv")
    p.add_argument("--out_dir", type=str, default="q4_outputs_v3")

    # dynamic weight controls
    p.add_argument("--alpha", type=float, default=0.6, help="EMA smoothing factor (0~1)")
    p.add_argument("--min_n", type=int, default=5, help="If n contestants < min_n, inherit previous week's weights")

    # robustness controls
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--delta", type=float, default=0.05)

    # deepening: stability penalty
    p.add_argument("--lambda_penalty", type=float, default=0.03,
                   help="Penalty strength for rank volatility in elimination ranking only")
    p.add_argument("--vol_window", type=int, default=4,
                   help="Rolling window for volatility (>=2). If <=1, use season-wide std.")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_pipeline(
        Paths(args.judges_csv, args.fan_csv, args.out_dir),
        alpha=args.alpha,
        min_n=args.min_n,
        kfold=args.kfold,
        delta=args.delta,
        lambda_penalty=args.lambda_penalty,
        vol_window=args.vol_window
    )

