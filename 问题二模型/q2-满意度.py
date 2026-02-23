# -*- coding: utf-8 -*-
"""
MCM/ICM 2026 Problem C — Question 2
Optimized, judge-style-friendly implementation to compare:
  (1) Rank-combination method
  (2) Percent-combination method

Key fixes vs. the original script:
1) Matches the show structure better: supports weeks with 0 or multiple eliminations,
   inferred from the official `results` field in 2026_MCM_Problem_C_Data.csv.
2) Final week is treated as a ranking week (no elimination), per problem statement appendix.
3) "Fan-friendliness" metric uses Spearman correlation SIGN (not abs), so "more aligned with fans"
   means higher positive correlation between final placement (1=best) and avg fan rank (1=best).
4) Adds CLI, robust IO, and exports reproducible tables.

Inputs (default filenames match the problem statement / your pipeline):
- 2026_MCM_Problem_C_Data.csv  (judges + results)
- optimized_fan_votes_v1.csv   (your estimated fan votes with columns: season, week, celebrity, fan_votes)

Outputs (CSV):
- 2_season_summary.csv
- 2_weekly_elimination_compare.csv
- 2_final_ranking_compare.csv
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# -----------------------------
# Utilities
# -----------------------------
_RESULTS_ELIM_RE = re.compile(r"Eliminated\s*Week\s*(\d+)", re.IGNORECASE)


def _safe_str(x) -> str:
    return str(x).strip() if pd.notna(x) else ""


def _extract_elimination_week(results_str: str) -> Optional[int]:
    """
    Parse judge_data['results'] like:
      - "Eliminated Week 2"
      - "1st Place"
      - "Runner up"
    Returns elimination week (int) or None.
    """
    if not results_str:
        return None
    m = _RESULTS_ELIM_RE.search(results_str)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class MethodOutput:
    eliminated: List[str]                 # eliminated contestants this week (may be empty)
    details: Dict[str, Dict[str, float]]  # per-contestant scores/ranks
    tie_info: str                         # short message


class EliminationComparatorQ2:
    """
    Compare rank vs percent combination across seasons.
    This class assumes fan votes are *estimated* and provided as a table.
    """

    def __init__(self, judge_data_path: str, fan_votes_path: str):
        self.judge_data = pd.read_csv(judge_data_path, encoding="utf-8")
        self.fan_votes = pd.read_csv(fan_votes_path)

        self._standardize_fan_votes_columns()
        self._preprocess()

        # results[season] = {"simulation": ..., "metrics": ...}
        self.results: Dict[int, dict] = {}

    # -----------------------------
    # 0) Cleaning / preprocessing
    # -----------------------------
    def _standardize_fan_votes_columns(self):
        if "celebrity_name" in self.fan_votes.columns:
            self.fan_votes = self.fan_votes.rename(columns={"celebrity_name": "celebrity"})
        elif "celebrity" not in self.fan_votes.columns:
            for col in ["star", "contestant", "name"]:
                if col in self.fan_votes.columns:
                    self.fan_votes = self.fan_votes.rename(columns={col: "celebrity"})
                    break

        required = ["season", "week", "celebrity", "fan_votes"]
        missing = [c for c in required if c not in self.fan_votes.columns]
        if missing:
            raise ValueError(f"fan_votes 缺少必要列: {missing}. 需要至少包含: {required}")

    def _preprocess(self):
        # fan votes
        self.fan_votes["celebrity"] = self.fan_votes["celebrity"].astype(str).str.strip()
        self.fan_votes["season"] = pd.to_numeric(self.fan_votes["season"], errors="coerce").astype(int)
        self.fan_votes["week"] = pd.to_numeric(self.fan_votes["week"], errors="coerce").astype(int)
        self.fan_votes["fan_votes"] = pd.to_numeric(self.fan_votes["fan_votes"], errors="coerce").fillna(0.0)

        # judge data
        if "season" not in self.judge_data.columns or "celebrity_name" not in self.judge_data.columns:
            raise ValueError("judge_data 必须包含 season 与 celebrity_name 列。")

        self.judge_data["season"] = pd.to_numeric(self.judge_data["season"], errors="coerce").astype(int)
        self.judge_data["celebrity_name"] = self.judge_data["celebrity_name"].astype(str).str.strip()

        # parse elimination weeks from 'results' (official)
        if "results" in self.judge_data.columns:
            self.judge_data["elim_week_official"] = self.judge_data["results"].astype(str).apply(_extract_elimination_week)
        else:
            self.judge_data["elim_week_official"] = None  # fallback

        self.seasons = sorted(self.fan_votes["season"].unique().tolist())

    # -----------------------------
    # 1) Accessors
    # -----------------------------
    def get_season_weeks(self, season: int) -> List[int]:
        weeks = sorted(self.fan_votes[self.fan_votes["season"] == season]["week"].unique().tolist())
        return weeks

    def get_weekly_judge_scores(self, season: int, week: int) -> Dict[str, float]:
        """
        Compute total judge score in a given (season, week) by summing week{week}_judge{k}_score (k=1..4).
        Returns {celebrity: total_score}. Missing / non-positive scores treated as 0.
        """
        sdf = self.judge_data[self.judge_data["season"] == season]
        if sdf.empty:
            return {}

        judge_cols = [f"week{week}_judge{k}_score" for k in range(1, 5)]
        scores: Dict[str, float] = {}

        for _, row in sdf.iterrows():
            celeb = _safe_str(row.get("celebrity_name"))
            total = 0.0
            valid = 0
            for col in judge_cols:
                if col in sdf.columns and pd.notna(row.get(col)):
                    try:
                        v = float(row.get(col))
                    except Exception:
                        continue
                    if v > 0:
                        total += v
                        valid += 1
            scores[celeb] = total if valid > 0 else 0.0
        return scores

    def get_fan_votes_for_week(self, season: int, week: int) -> Dict[str, float]:
        wdf = self.fan_votes[(self.fan_votes["season"] == season) & (self.fan_votes["week"] == week)]
        if wdf.empty:
            return {}
        return dict(zip(wdf["celebrity"], wdf["fan_votes"]))

    def get_active_contestants_for_week(self, season: int, week: int) -> List[str]:
        """
        Primary definition of 'active' contestants: those appearing in fan_votes table at (season, week).
        Fallback: judge score > 0.
        """
        wdf = self.fan_votes[(self.fan_votes["season"] == season) & (self.fan_votes["week"] == week)]
        if not wdf.empty:
            return wdf["celebrity"].tolist()

        j = self.get_weekly_judge_scores(season, week)
        return [c for c, sc in j.items() if sc > 0]

    def get_official_elimination_count_by_week(self, season: int, final_week: int) -> Dict[int, int]:
        """
        Infer how many eliminations happened in each week from the official 'results' field in judge_data.
        Problem statement notes: some weeks have no elimination; others may have >1. (See notes)
        Final week is treated as ranking-only: 0 eliminations.
        """
        sdf = self.judge_data[self.judge_data["season"] == season]
        counts: Dict[int, int] = {}
        if "elim_week_official" in sdf.columns:
            for w, g in sdf.dropna(subset=["elim_week_official"]).groupby("elim_week_official"):
                try:
                    w_int = int(w)
                except Exception:
                    continue
                counts[w_int] = int(len(g))
        # enforce final week = 0 eliminations (ranking week)
        counts[final_week] = 0
        return counts

    # -----------------------------
    # 2) Combination methods
    # -----------------------------
    @staticmethod
    def _rank_method(judge_scores: Dict[str, float], fan_votes: Dict[str, float], active: List[str]) -> MethodOutput:
        df = pd.DataFrame({
            "celebrity": active,
            "judge_score": [float(judge_scores.get(c, 0.0)) for c in active],
            "fan_votes": [float(fan_votes.get(c, 0.0)) for c in active],
        })

        # Official idea: combine by ranks (Appendix)
        # Use "min" to make ties explicit; can be changed to "average" if desired.
        df["judge_rank"] = df["judge_score"].rank(method="min", ascending=False)
        df["fan_rank"] = df["fan_votes"].rank(method="min", ascending=False)
        df["total_rank"] = df["judge_rank"] + df["fan_rank"]

        # Worst = largest total_rank (if tie: lower judge_score, then lower fan_votes)
        df = df.sort_values(["total_rank", "judge_score", "fan_votes"], ascending=[False, True, True])
        # Provide a deterministic tie note
        worst_total = df["total_rank"].iloc[0]
        tied = df[df["total_rank"] == worst_total]
        tie_info = "No tie"
        if len(tied) > 1:
            # check if judge_score breaks tie
            min_j = tied["judge_score"].min()
            tied2 = tied[tied["judge_score"] == min_j]
            tie_info = "Tie on total_rank; broken by judge score" if len(tied2) == 1 else "Tie on total_rank & judge score; broken by fan votes"

        details = df.set_index("celebrity")[["judge_score", "fan_votes", "judge_rank", "fan_rank", "total_rank"]].to_dict("index")
        return MethodOutput(eliminated=[], details=details, tie_info=tie_info)

    @staticmethod
    def _percent_method(judge_scores: Dict[str, float], fan_votes: Dict[str, float], active: List[str], alpha: float) -> MethodOutput:
        judge = {c: float(judge_scores.get(c, 0.0)) for c in active}
        fan = {c: float(fan_votes.get(c, 0.0)) for c in active}

        # Avoid division by zero
        eps = 1e-12
        judge_total = sum(judge.values())
        fan_total = sum(fan.values())
        if judge_total <= 0:
            judge = {c: judge[c] + eps for c in active}
            judge_total = sum(judge.values())
        if fan_total <= 0:
            fan = {c: fan[c] + eps for c in active}
            fan_total = sum(fan.values())

        judge_p = {c: judge[c] / judge_total for c in active}
        fan_p = {c: fan[c] / fan_total for c in active}
        combined = {c: alpha * judge_p[c] + (1 - alpha) * fan_p[c] for c in active}

        df = pd.DataFrame({
            "celebrity": active,
            "judge_score": [judge[c] for c in active],
            "fan_votes": [fan[c] for c in active],
            "judge_percent": [judge_p[c] for c in active],
            "fan_percent": [fan_p[c] for c in active],
            "combined_score": [combined[c] for c in active],
        })

        # Worst = smallest combined_score (tie-break similar)
        df = df.sort_values(["combined_score", "judge_score", "fan_votes"], ascending=[True, True, True])
        worst_score = df["combined_score"].iloc[0]
        tied = df[df["combined_score"] == worst_score]
        tie_info = "No tie"
        if len(tied) > 1:
            min_j = tied["judge_score"].min()
            tied2 = tied[tied["judge_score"] == min_j]
            tie_info = "Tie on combined; broken by judge score" if len(tied2) == 1 else "Tie on combined & judge score; broken by fan votes"

        details = df.set_index("celebrity")[["judge_score", "fan_votes", "judge_percent", "fan_percent", "combined_score"]].to_dict("index")
        return MethodOutput(eliminated=[], details=details, tie_info=tie_info)

    # -----------------------------
    # 3) Season simulation (supports 0 / multi eliminations, and finals ranking)
    # -----------------------------
    def simulate_season(self, season: int, alpha: float = 0.5) -> dict:
        weeks = self.get_season_weeks(season)
        if not weeks:
            return {}

        final_week = max(weeks)
        elim_counts = self.get_official_elimination_count_by_week(season, final_week=final_week)

        sim = {
            "rank": {
                "active": set(),
                "eliminations": {},            # week -> list[str]
                "trajectory": [],              # in order of elimination (each element is a name)
                "weekly_details": {},           # week -> dict
                "final_ranking": [],
                "champion": None,
            },
            "percent": {
                "active": set(),
                "eliminations": {},
                "trajectory": [],
                "weekly_details": {},
                "final_ranking": [],
                "champion": None,
            },
        }

        # Initialize active set at first week
        first_week = min(weeks)
        initial_active = self.get_active_contestants_for_week(season, first_week)
        sim["rank"]["active"] = set(initial_active)
        sim["percent"]["active"] = set(initial_active)

        for w in weeks:
            judge_scores = self.get_weekly_judge_scores(season, w)
            fan_votes = self.get_fan_votes_for_week(season, w)

            # How many eliminations in this week?
            k = int(elim_counts.get(w, 1))  # if missing, fallback to 1 elimination
            if w == final_week:
                k = 0  # finals: ranking only, per problem statement

            # -------- Rank weekly compute
            rank_active = list(sim["rank"]["active"])
            r_out = self._rank_method(judge_scores, fan_votes, rank_active) if rank_active else MethodOutput([], {}, "No active contestants")

            eliminated_r: List[str] = []
            if k > 0 and len(rank_active) > 0:
                # pick k worst by ordering already sorted in details keys? safer to rebuild ordering from details
                # Rebuild a DF from details for deterministic ordering
                rdf = pd.DataFrame([
                    {"celebrity": c, **vals} for c, vals in r_out.details.items()
                ])
                if not rdf.empty:
                    rdf = rdf.sort_values(["total_rank", "judge_score", "fan_votes"], ascending=[False, True, True])
                    eliminated_r = rdf["celebrity"].head(min(k, len(rdf) - 1 if len(rdf) > 1 else 1)).tolist()

            for name in eliminated_r:
                if name in sim["rank"]["active"]:
                    sim["rank"]["active"].remove(name)
                    sim["rank"]["trajectory"].append(name)

            sim["rank"]["eliminations"][w] = eliminated_r
            sim["rank"]["weekly_details"][w] = {
                "eliminated": eliminated_r,
                "details": r_out.details,
                "tie_info": r_out.tie_info,
                "active_count": len(rank_active),
                "fan_votes": fan_votes,
                "k_eliminations_official": k,
                "is_final_week": (w == final_week),
            }

            # -------- Percent weekly compute
            percent_active = list(sim["percent"]["active"])
            p_out = self._percent_method(judge_scores, fan_votes, percent_active, alpha=alpha) if percent_active else MethodOutput([], {}, "No active contestants")

            eliminated_p: List[str] = []
            if k > 0 and len(percent_active) > 0:
                pdf = pd.DataFrame([
                    {"celebrity": c, **vals} for c, vals in p_out.details.items()
                ])
                if not pdf.empty:
                    pdf = pdf.sort_values(["combined_score", "judge_score", "fan_votes"], ascending=[True, True, True])
                    eliminated_p = pdf["celebrity"].head(min(k, len(pdf) - 1 if len(pdf) > 1 else 1)).tolist()

            for name in eliminated_p:
                if name in sim["percent"]["active"]:
                    sim["percent"]["active"].remove(name)
                    sim["percent"]["trajectory"].append(name)

            sim["percent"]["eliminations"][w] = eliminated_p
            sim["percent"]["weekly_details"][w] = {
                "eliminated": eliminated_p,
                "details": p_out.details,
                "tie_info": p_out.tie_info,
                "active_count": len(percent_active),
                "fan_votes": fan_votes,
                "k_eliminations_official": k,
                "is_final_week": (w == final_week),
            }

        # Final week ranking among finalists
        finalists_rank = list(sim["rank"]["active"])
        finalists_percent = list(sim["percent"]["active"])

        # Rank final ordering: smaller total_rank is better, but our details for final_week contain all actives
        r_final_details = sim["rank"]["weekly_details"].get(final_week, {}).get("details", {})
        if r_final_details:
            rdf = pd.DataFrame([{"celebrity": c, **vals} for c, vals in r_final_details.items()])
            if not rdf.empty and "total_rank" in rdf.columns:
                rdf = rdf.sort_values(["total_rank", "judge_score", "fan_votes"], ascending=[True, False, False])
                finalists_rank = rdf["celebrity"].tolist()

        p_final_details = sim["percent"]["weekly_details"].get(final_week, {}).get("details", {})
        if p_final_details:
            pdf = pd.DataFrame([{"celebrity": c, **vals} for c, vals in p_final_details.items()])
            if not pdf.empty and "combined_score" in pdf.columns:
                pdf = pdf.sort_values(["combined_score", "judge_score", "fan_votes"], ascending=[False, False, False])
                finalists_percent = pdf["celebrity"].tolist()

        # Build a full season ranking: finalists (best->worst) then reverse elimination trajectory
        sim["rank"]["final_ranking"] = finalists_rank + list(reversed(sim["rank"]["trajectory"]))
        sim["percent"]["final_ranking"] = finalists_percent + list(reversed(sim["percent"]["trajectory"]))

        sim["rank"]["champion"] = sim["rank"]["final_ranking"][0] if sim["rank"]["final_ranking"] else None
        sim["percent"]["champion"] = sim["percent"]["final_ranking"][0] if sim["percent"]["final_ranking"] else None

        return sim

    # -----------------------------
    # 4) Metrics for Q2
    # -----------------------------
    @staticmethod
    def _avg_fan_rank_from_weekly_details(weekly_details: Dict[int, dict]) -> Dict[str, float]:
        """
        Average fan-rank across weeks for each contestant.
        Rank definition: 1=highest votes, larger=lower votes.
        """
        ranks: Dict[str, List[float]] = {}
        for _, info in weekly_details.items():
            fan_votes = info.get("fan_votes", {})
            if not fan_votes:
                continue
            s = pd.Series(fan_votes, dtype=float).rank(method="min", ascending=False)
            for c, r in s.items():
                ranks.setdefault(c, []).append(float(r))
        return {c: float(np.mean(v)) for c, v in ranks.items() if len(v) > 0}

    @staticmethod
    def _spearman_alignment(final_ranking: List[str], avg_fan_rank: Dict[str, float]) -> Tuple[float, float, int]:
        """
        Spearman correlation between:
          - final_position (1=best)
          - avg_fan_rank (1=best)
        Higher +corr => more aligned with fans.
        Returns: (corr, p_value, n_used)
        """
        rows = []
        for pos, c in enumerate(final_ranking, start=1):
            if c in avg_fan_rank:
                rows.append((pos, avg_fan_rank[c]))
        if len(rows) < 2:
            return 0.0, 1.0, len(rows)
        df = pd.DataFrame(rows, columns=["final_position", "avg_fan_rank"])
        corr, p = spearmanr(df["final_position"], df["avg_fan_rank"])
        corr = float(corr) if pd.notna(corr) else 0.0
        p = float(p) if pd.notna(p) else 1.0
        return corr, p, len(rows)

    def calculate_q2_metrics(self, season_sim: dict) -> Dict[str, object]:
        """
        - elimination_difference_rate: fraction of weeks where eliminated sets differ
        - champion_changed: whether champion differs
        - rank_spearman / percent_spearman: fan alignment (higher => more fan-friendly)
        """
        metrics: Dict[str, object] = {}

        # elimination difference rate
        r_elim = season_sim["rank"]["eliminations"]
        p_elim = season_sim["percent"]["eliminations"]
        weeks = sorted(set(r_elim.keys()) | set(p_elim.keys()))
        if weeks:
            diff = 0
            for w in weeks:
                if set(r_elim.get(w, [])) != set(p_elim.get(w, [])):
                    diff += 1
            metrics["elimination_difference_rate"] = diff / len(weeks)
        else:
            metrics["elimination_difference_rate"] = 0.0

        metrics["champion_changed"] = (season_sim["rank"].get("champion") != season_sim["percent"].get("champion"))

        # fan alignment
        rank_avg = self._avg_fan_rank_from_weekly_details(season_sim["rank"]["weekly_details"])
        percent_avg = self._avg_fan_rank_from_weekly_details(season_sim["percent"]["weekly_details"])

        r_corr, r_p, r_n = self._spearman_alignment(season_sim["rank"]["final_ranking"], rank_avg)
        p_corr, p_p, p_n = self._spearman_alignment(season_sim["percent"]["final_ranking"], percent_avg)

        metrics["rank_spearman"] = r_corr
        metrics["rank_p_value"] = r_p
        metrics["rank_n"] = r_n

        metrics["percent_spearman"] = p_corr
        metrics["percent_p_value"] = p_p
        metrics["percent_n"] = p_n

        if r_corr > p_corr:
            metrics["more_fan_friendly"] = "Rank Combination"
        elif p_corr > r_corr:
            metrics["more_fan_friendly"] = "Percent Combination"
        else:
            metrics["more_fan_friendly"] = "Equal"

        return metrics

    # -----------------------------
    # 5) Run + export
    # -----------------------------
    def run(self, seasons: Optional[List[int]] = None, alpha: float = 0.5) -> Dict[int, dict]:
        if seasons is None:
            seasons = self.seasons

        self.results = {}
        for s in seasons:
            s = int(s)
            try:
                sim = self.simulate_season(s, alpha=alpha)
                if not sim:
                    continue
                metrics = self.calculate_q2_metrics(sim)
                self.results[s] = {"simulation": sim, "metrics": metrics}
            except Exception as e:
                print(f"[WARN] season={s} failed: {e}")

        return self.results

    def export_tables(self, out_dir: str, alpha: float):
        if not self.results:
            raise RuntimeError("No results. Run .run() first.")

        os.makedirs(out_dir, exist_ok=True)

        season_rows = []
        weekly_rows = []
        final_rows = []

        for season, obj in sorted(self.results.items()):
            sim = obj["simulation"]
            m = obj["metrics"]

            season_rows.append({
                "season": season,
                "alpha_percent_method": alpha,
                "elimination_difference_rate": m.get("elimination_difference_rate"),
                "champion_changed": m.get("champion_changed"),
                "rank_spearman": m.get("rank_spearman"),
                "percent_spearman": m.get("percent_spearman"),
                "rank_p_value": m.get("rank_p_value"),
                "percent_p_value": m.get("percent_p_value"),
                "rank_n": m.get("rank_n"),
                "percent_n": m.get("percent_n"),
                "more_fan_friendly": m.get("more_fan_friendly"),
                "rank_champion": sim["rank"].get("champion"),
                "percent_champion": sim["percent"].get("champion"),
            })

            weeks = sorted(set(sim["rank"]["weekly_details"].keys()) | set(sim["percent"]["weekly_details"].keys()))
            for w in weeks:
                rinfo = sim["rank"]["weekly_details"].get(w, {})
                pinfo = sim["percent"]["weekly_details"].get(w, {})
                weekly_rows.append({
                    "season": season,
                    "week": w,
                    "rank_eliminated": ";".join(rinfo.get("eliminated", []) or []),
                    "percent_eliminated": ";".join(pinfo.get("eliminated", []) or []),
                    "same_elimination_set": (set(rinfo.get("eliminated", []) or []) == set(pinfo.get("eliminated", []) or [])),
                    "k_eliminations_official": rinfo.get("k_eliminations_official"),
                    "is_final_week": rinfo.get("is_final_week"),
                    "rank_tie_info": rinfo.get("tie_info"),
                    "percent_tie_info": pinfo.get("tie_info"),
                    "rank_active_count": rinfo.get("active_count"),
                    "percent_active_count": pinfo.get("active_count"),
                })

            rrank = sim["rank"].get("final_ranking", [])
            prank = sim["percent"].get("final_ranking", [])
            maxlen = max(len(rrank), len(prank))
            for pos in range(1, maxlen + 1):
                final_rows.append({
                    "season": season,
                    "final_position": pos,
                    "rank_method_contestant": (rrank[pos - 1] if pos - 1 < len(rrank) else None),
                    "percent_method_contestant": (prank[pos - 1] if pos - 1 < len(prank) else None),
                })

        season_df = pd.DataFrame(season_rows).sort_values("season")
        weekly_df = pd.DataFrame(weekly_rows).sort_values(["season", "week"])
        final_df = pd.DataFrame(final_rows).sort_values(["season", "final_position"])

        paths = {
            "season_summary": os.path.join(out_dir, "2_season_summary.csv"),
            "weekly_elimination_compare": os.path.join(out_dir, "2_weekly_elimination_compare.csv"),
            "final_ranking_compare": os.path.join(out_dir, "2_final_ranking_compare.csv"),
        }
        season_df.to_csv(paths["season_summary"], index=False, encoding="utf-8-sig")
        weekly_df.to_csv(paths["weekly_elimination_compare"], index=False, encoding="utf-8-sig")
        final_df.to_csv(paths["final_ranking_compare"], index=False, encoding="utf-8-sig")

        return paths


def main():
    parser = argparse.ArgumentParser(description="Q2: compare Rank vs Percent combination across seasons.")
    parser.add_argument("--judge", default="2026_MCM_Problem_C_Data.csv", help="Path to 2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--fan", default="optimized_fan_votes.csv", help="Path to estimated fan votes csv")
    parser.add_argument("--out", default=".", help="Output directory for CSV tables")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha used in Percent method (judge weight).")
    parser.add_argument("--season_from", type=int, default=1)
    parser.add_argument("--season_to", type=int, default=34)
    args = parser.parse_args()

    comp = EliminationComparatorQ2(judge_data_path=args.judge, fan_votes_path=args.fan)
    comp.run(seasons=list(range(args.season_from, args.season_to + 1)), alpha=args.alpha)
    paths = comp.export_tables(out_dir=args.out, alpha=args.alpha)

    print("\n✅ Exported CSV tables:")
    for k, v in paths.items():
        print(f" - {k}: {v}")
        
        # ---- CLI 文字输出：满意度（粉丝一致性）概览 ----
    print("\n📌 满意度（Spearman：越大越符合粉丝）逐季概览：")
    for season in sorted(comp.results.keys()):
        metrics = comp.results[season]["metrics"]
        rank_r = metrics.get("rank_spearman", float("nan"))
        percent_r = metrics.get("percent_spearman", float("nan"))
        better = metrics.get("more_fan_friendly", "N/A")

        print(
            f"Season {season:>2}: "
            f"Rank={rank_r:+.3f}, "
            f"Percent={percent_r:+.3f}  --> "
            f"更符合粉丝：{better}"
        )

    # （可选）再给一个总览：统计哪个方法赢得多
    winners = [comp.results[s]["metrics"].get("more_fan_friendly", "") for s in comp.results]
    rank_win = sum(1 for w in winners if w == "Rank")
    percent_win = sum(1 for w in winners if w == "Percent")
    tie = len(winners) - rank_win - percent_win



if __name__ == "__main__":
    main()
