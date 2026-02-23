# -*- coding: utf-8 -*-
"""
MCM/ICM 2026 Problem C — Q2 (Controversy) Optimized v2
=====================================================

Target questions (Q2):
1) For "controversial" celebrities (judges vs fans disagree), would using Rank vs Percent
   combination lead to the same outcomes for each contestant?
2) What is the impact if an additional rule is added: identify the bottom two using the combined
   judge+fan method, then judges choose which of those two to eliminate? Should it be adopted?

What this script does (tables only, for downstream processing):
- Runs counterfactual season simulations under four schemes:
    A) Rank
    B) Percent (alpha configurable)
    C) Rank + Judges Save (bottom two by Rank; judges eliminate lower judge_score in bottom two)
    D) Percent + Judges Save (bottom two by Percent; judges eliminate lower judge_score in bottom two)
- Supports weeks with 0 or multiple eliminations inferred from official 'results' in judge_data
  (counts how many contestants have "Eliminated Week X"). Final week treated as ranking week (0 eliminations).
- Exports CSV tables:
    1) 2_2_weekly_methods_compare.csv
    2) 2_2_season_summary.csv
    3) 2_2_controversy_celebrity_summary.csv  (optional; default includes famous controversy examples)

Determinism / reproducibility:
- All tie breaks are deterministic; no randomness.

Inputs (defaults match typical pipeline):
- 2026_MCM_Problem_C_Data.csv
- optimized_fan_votes.csv  (estimated fan votes; must have: season, week, celebrity, fan_votes)

Run:
  python q2-2_optimized_v2.py --judge 2026_MCM_Problem_C_Data.csv --fan optimized_fan_votes.csv --out . --alpha 0.5

Notes on "Judges Save":
- The real show uses judge votes among bottom-two. Since per-judge preference is unobserved in our
  model, we use a reproducible proxy: eliminate the one with LOWER total judge_score in that week
  (tie -> lower fan_votes).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


_RESULTS_ELIM_RE = re.compile(r"Eliminated\s*Week\s*(\d+)", re.IGNORECASE)


def _safe_str(x) -> str:
    return str(x).strip() if pd.notna(x) else ""


def _extract_elimination_week(results_str: str) -> Optional[int]:
    if not results_str:
        return None
    m = _RESULTS_ELIM_RE.search(str(results_str))
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class WeeklyOutcome:
    eliminated: List[str]   # eliminated contestants this week (can be empty)
    bottom_two: List[str]   # bottom two candidates (method-defined)
    table: pd.DataFrame     # scoring table for auditing


class Q2ComparatorV2:
    def __init__(self, judge_data_path: str, fan_votes_path: str):
        self.judge_data = pd.read_csv(judge_data_path, encoding="utf-8")
        self.fan_votes = pd.read_csv(fan_votes_path)

        self._standardize_fan_votes_columns()
        self._preprocess()

    # -----------------------------
    # Cleaning / preprocessing
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
            raise ValueError(f"fan_votes 缺少必要列: {missing}; 需要包含 {required}")

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

        # official elimination week from results
        if "results" in self.judge_data.columns:
            self.judge_data["elim_week_official"] = self.judge_data["results"].astype(str).apply(_extract_elimination_week)
        else:
            self.judge_data["elim_week_official"] = None

        # placement for celebrity summary
        if "placement" in self.judge_data.columns:
            self.judge_data["placement"] = pd.to_numeric(self.judge_data["placement"], errors="coerce")
        else:
            self.judge_data["placement"] = np.nan

        self.seasons = sorted(self.fan_votes["season"].unique().tolist())

    # -----------------------------
    # Season/week helpers
    # -----------------------------
    def get_season_weeks(self, season: int) -> List[int]:
        return sorted(self.fan_votes[self.fan_votes["season"] == season]["week"].unique().tolist())

    def get_initial_active(self, season: int) -> List[str]:
        """Initial active set = all contestants appearing in the first week of fan vote table."""
        weeks = self.get_season_weeks(season)
        if not weeks:
            return []
        w0 = min(weeks)
        wdf = self.fan_votes[(self.fan_votes["season"] == season) & (self.fan_votes["week"] == w0)]
        return wdf["celebrity"].tolist() if not wdf.empty else []

    def get_weekly_judge_scores(self, season: int, week: int) -> Dict[str, float]:
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

    def get_official_elimination_count_by_week(self, season: int, final_week: int) -> Dict[int, int]:
        """Count how many were eliminated in each week from official results; final week forced to 0."""
        sdf = self.judge_data[self.judge_data["season"] == season]
        counts: Dict[int, int] = {}
        elim_df = sdf.dropna(subset=["elim_week_official"])
        if not elim_df.empty:
            for w, g in elim_df.groupby("elim_week_official"):
                try:
                    counts[int(w)] = int(len(g))
                except Exception:
                    continue
        counts[int(final_week)] = 0
        return counts

    # -----------------------------
    # Scoring tables
    # -----------------------------
    @staticmethod
    def _rank_table(judge_scores: Dict[str, float], fan_votes: Dict[str, float], active: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({
            "celebrity": active,
            "judge_score": [float(judge_scores.get(c, 0.0)) for c in active],
            "fan_votes": [float(fan_votes.get(c, 0.0)) for c in active],
        })
        df["judge_rank"] = df["judge_score"].rank(method="min", ascending=False)
        df["fan_rank"] = df["fan_votes"].rank(method="min", ascending=False)
        df["total_rank"] = df["judge_rank"] + df["fan_rank"]
        return df

    @staticmethod
    def _percent_table(judge_scores: Dict[str, float], fan_votes: Dict[str, float], active: List[str], alpha: float) -> pd.DataFrame:
        judge = {c: float(judge_scores.get(c, 0.0)) for c in active}
        fan = {c: float(fan_votes.get(c, 0.0)) for c in active}

        eps = 1e-12
        jt = sum(judge.values())
        ft = sum(fan.values())
        if jt <= 0:
            judge = {c: judge[c] + eps for c in active}
            jt = sum(judge.values())
        if ft <= 0:
            fan = {c: fan[c] + eps for c in active}
            ft = sum(fan.values())

        judge_p = {c: judge[c] / jt for c in active}
        fan_p = {c: fan[c] / ft for c in active}
        combined = {c: alpha * judge_p[c] + (1 - alpha) * fan_p[c] for c in active}

        df = pd.DataFrame({
            "celebrity": active,
            "judge_score": [judge[c] for c in active],
            "fan_votes": [fan[c] for c in active],
            "judge_percent": [judge_p[c] for c in active],
            "fan_percent": [fan_p[c] for c in active],
            "combined_score": [combined[c] for c in active],
        })
        return df

    @staticmethod
    def _pick(df: pd.DataFrame, order_cols: List[Tuple[str, bool]], k: int) -> List[str]:
        """Deterministic pick first k by sort order."""
        if df.empty or k <= 0:
            return []
        cols = [c for c, _ in order_cols]
        asc = [a for _, a in order_cols]
        sdf = df.sort_values(cols, ascending=asc, kind="mergesort")
        return sdf["celebrity"].head(k).tolist()

    @staticmethod
    def _judge_save_eliminate(df: pd.DataFrame, bottom_two: List[str]) -> Optional[str]:
        """Judges-save proxy: eliminate lower judge_score in bottom_two (tie -> lower fan_votes)."""
        if df.empty or len(bottom_two) < 2:
            return None
        sub = df[df["celebrity"].isin(bottom_two)].copy()
        if sub.empty:
            return None
        sub = sub.sort_values(["judge_score", "fan_votes"], ascending=[True, True], kind="mergesort")
        return sub["celebrity"].iloc[0]

    # -----------------------------
    # Weekly outcomes (given active set)
    # -----------------------------
    def weekly_rank(self, season: int, week: int, active: List[str], k_elim: int) -> WeeklyOutcome:
        j = self.get_weekly_judge_scores(season, week)
        f = self.get_fan_votes_for_week(season, week)
        df = self._rank_table(j, f, active) if active else pd.DataFrame()

        # bottom two: worst by (total_rank desc, judge_score asc, fan_votes asc)
        bottom2 = self._pick(df, [("total_rank", False), ("judge_score", True), ("fan_votes", True)], 2)

        # eliminated k: same ordering
        elim = []
        if k_elim > 0 and len(active) > 1:
            elim = self._pick(df, [("total_rank", False), ("judge_score", True), ("fan_votes", True)], min(k_elim, len(active) - 1))
        return WeeklyOutcome(eliminated=elim, bottom_two=bottom2, table=df)

    def weekly_percent(self, season: int, week: int, active: List[str], k_elim: int, alpha: float) -> WeeklyOutcome:
        j = self.get_weekly_judge_scores(season, week)
        f = self.get_fan_votes_for_week(season, week)
        df = self._percent_table(j, f, active, alpha=alpha) if active else pd.DataFrame()

        bottom2 = self._pick(df, [("combined_score", True), ("judge_score", True), ("fan_votes", True)], 2)

        elim = []
        if k_elim > 0 and len(active) > 1:
            elim = self._pick(df, [("combined_score", True), ("judge_score", True), ("fan_votes", True)], min(k_elim, len(active) - 1))
        return WeeklyOutcome(eliminated=elim, bottom_two=bottom2, table=df)

    def weekly_rank_judges_save(self, season: int, week: int, active: List[str], k_elim: int) -> WeeklyOutcome:
        """If k_elim>1: eliminate iteratively, recomputing bottom_two after removing one."""
        if not active:
            return WeeklyOutcome([], [], pd.DataFrame())

        eliminated = []
        working = active[:]
        last_table = pd.DataFrame()
        last_bottom2 = []

        for _ in range(min(k_elim, max(0, len(working) - 1))):
            out = self.weekly_rank(season, week, working, k_elim=0)  # compute bottom2 & table
            last_table = out.table
            last_bottom2 = out.bottom_two
            elim = self._judge_save_eliminate(last_table, last_bottom2)
            if elim is None:
                break
            eliminated.append(elim)
            working = [c for c in working if c != elim]

        # audit table based on original active
        audit = self.weekly_rank(season, week, active, k_elim=0)
        return WeeklyOutcome(eliminated=eliminated, bottom_two=audit.bottom_two, table=audit.table)

    def weekly_percent_judges_save(self, season: int, week: int, active: List[str], k_elim: int, alpha: float) -> WeeklyOutcome:
        if not active:
            return WeeklyOutcome([], [], pd.DataFrame())

        eliminated = []
        working = active[:]
        for _ in range(min(k_elim, max(0, len(working) - 1))):
            out = self.weekly_percent(season, week, working, k_elim=0, alpha=alpha)
            elim = self._judge_save_eliminate(out.table, out.bottom_two)
            if elim is None:
                break
            eliminated.append(elim)
            working = [c for c in working if c != elim]

        audit = self.weekly_percent(season, week, active, k_elim=0, alpha=alpha)
        return WeeklyOutcome(eliminated=eliminated, bottom_two=audit.bottom_two, table=audit.table)

    # -----------------------------
    # Season simulation
    # -----------------------------
    def simulate_season(self, season: int, alpha: float) -> Dict[str, dict]:
        weeks = self.get_season_weeks(season)
        if not weeks:
            return {}

        final_week = max(weeks)
        elim_counts = self.get_official_elimination_count_by_week(season, final_week=final_week)

        # schemes
        schemes = ["rank", "percent", "rank_js", "percent_js"]
        active = {s: set(self.get_initial_active(season)) for s in schemes}
        elim_weeks = {s: {} for s in schemes}  # contestant -> week eliminated (first time)
        weekly_details = {s: {} for s in schemes}

        for week in weeks:
            k = int(elim_counts.get(week, 1))
            if week == final_week:
                k = 0  # finals ranking week

            for scheme in schemes:
                current_active = sorted(active[scheme])
                if not current_active:
                    weekly_details[scheme][week] = {"k": k, "eliminated": [], "bottom_two": [], "active_count": 0}
                    continue

                if scheme == "rank":
                    out = self.weekly_rank(season, week, current_active, k_elim=k)
                elif scheme == "percent":
                    out = self.weekly_percent(season, week, current_active, k_elim=k, alpha=alpha)
                elif scheme == "rank_js":
                    out = self.weekly_rank_judges_save(season, week, current_active, k_elim=k)
                else:  # percent_js
                    out = self.weekly_percent_judges_save(season, week, current_active, k_elim=k, alpha=alpha)

                # update active / elim_weeks
                for name in out.eliminated:
                    if name in active[scheme]:
                        active[scheme].remove(name)
                    if name not in elim_weeks[scheme]:
                        elim_weeks[scheme][name] = week

                weekly_details[scheme][week] = {
                    "k": k,
                    "eliminated": out.eliminated,
                    "bottom_two": out.bottom_two,
                    "active_count": len(current_active),
                }

        # survivors get elimination_week = None
        final_active = {s: sorted(active[s]) for s in schemes}
        return {
            "weekly": weekly_details,
            "elim_week": elim_weeks,
            "final_active": final_active,
            "final_week": final_week,
        }

    # -----------------------------
    # Table outputs
    # -----------------------------
    def build_weekly_table(self, alpha: float) -> pd.DataFrame:
        rows = []
        for season in self.seasons:
            sim = self.simulate_season(season, alpha=alpha)
            if not sim:
                continue
            final_week = sim["final_week"]
            for week, rec in sim["weekly"]["rank"].items():
                r = sim["weekly"]["rank"].get(week, {})
                p = sim["weekly"]["percent"].get(week, {})
                rjs = sim["weekly"]["rank_js"].get(week, {})
                pjs = sim["weekly"]["percent_js"].get(week, {})

                r_el = set(r.get("eliminated", []) or [])
                p_el = set(p.get("eliminated", []) or [])
                rjs_el = set(rjs.get("eliminated", []) or [])
                pjs_el = set(pjs.get("eliminated", []) or [])

                rows.append({
                    "season": season,
                    "week": week,
                    "k_eliminations_official": r.get("k", np.nan),
                    "is_final_week": (week == final_week),

                    "rank_eliminated": ";".join(r.get("eliminated", []) or []),
                    "percent_eliminated": ";".join(p.get("eliminated", []) or []),
                    "rank_bottom_two": ";".join(r.get("bottom_two", []) or []),
                    "percent_bottom_two": ";".join(p.get("bottom_two", []) or []),

                    "rank_judges_save_eliminated": ";".join(rjs.get("eliminated", []) or []),
                    "percent_judges_save_eliminated": ";".join(pjs.get("eliminated", []) or []),
                    "rank_js_bottom_two": ";".join(rjs.get("bottom_two", []) or []),
                    "percent_js_bottom_two": ";".join(pjs.get("bottom_two", []) or []),

                    "rank_vs_percent_diff": (r_el != p_el),
                    "rank_js_changes": (r_el != rjs_el),
                    "percent_js_changes": (p_el != pjs_el),
                })

        return pd.DataFrame(rows).sort_values(["season", "week"])

    def build_season_summary(self, weekly_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        rows = []
        for season, g in weekly_df.groupby("season"):
            rows.append({
                "season": int(season),
                "alpha_percent_method": alpha,
                "weeks_count": int(len(g)),
                "rank_vs_percent_diff_rate": float(g["rank_vs_percent_diff"].mean()) if len(g) else 0.0,
                "rank_js_change_rate": float(g["rank_js_changes"].mean()) if len(g) else 0.0,
                "percent_js_change_rate": float(g["percent_js_changes"].mean()) if len(g) else 0.0,
            })
        return pd.DataFrame(rows).sort_values("season")

    def build_controversy_summary(
        self,
        weekly_df: pd.DataFrame,
        alpha: float,
        controversial_cases: List[Tuple[int, str]]
    ) -> pd.DataFrame:
        """
        For each (season, celebrity):
        - official elimination week/placement
        - counterfactual elimination week under each scheme (rank/percent/rank_js/percent_js)
        - whether method choice changes their elimination week
        - whether adding judges-save changes elimination week
        """
        rows = []
        # simulate once per season to get counterfactual elim weeks
        sim_cache = {season: self.simulate_season(season, alpha=alpha) for season in sorted({s for s, _ in controversial_cases})}

        for season, celeb in controversial_cases:
            season = int(season)
            celeb = str(celeb).strip()

            sdf = self.judge_data[(self.judge_data["season"] == season) & (self.judge_data["celebrity_name"] == celeb)]
            official_elim_week = None
            official_place = None
            if not sdf.empty:
                official_elim_week = sdf["elim_week_official"].iloc[0] if "elim_week_official" in sdf.columns else None
                official_place = sdf["placement"].iloc[0] if "placement" in sdf.columns else None

            sim = sim_cache.get(season, {})
            ew = sim.get("elim_week", {})
            rank_ew = ew.get("rank", {}).get(celeb)
            percent_ew = ew.get("percent", {}).get(celeb)
            rank_js_ew = ew.get("rank_js", {}).get(celeb)
            percent_js_ew = ew.get("percent_js", {}).get(celeb)

            # If survived to finals in a scheme, elim week is None. Record as "Survived".
            def _ew_label(x):
                return "Survived" if x is None or (isinstance(x, float) and pd.isna(x)) else int(x)

            rows.append({
                "season": season,
                "celebrity": celeb,
                "official_elimination_week": official_elim_week,
                "official_placement": official_place,

                "rank_elimination_week": _ew_label(rank_ew),
                "percent_elimination_week": _ew_label(percent_ew),
                "rank_js_elimination_week": _ew_label(rank_js_ew),
                "percent_js_elimination_week": _ew_label(percent_js_ew),

                "method_choice_changes_elim_week": (rank_ew != percent_ew),
                "rank_judges_save_changes_elim_week": (rank_ew != rank_js_ew),
                "percent_judges_save_changes_elim_week": (percent_ew != percent_js_ew),
            })

        return pd.DataFrame(rows)

    def export_tables(
        self,
        out_dir: str,
        alpha: float,
        controversial_cases: Optional[List[Tuple[int, str]]] = None
    ) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)

        weekly_df = self.build_weekly_table(alpha=alpha)
        season_df = self.build_season_summary(weekly_df, alpha=alpha)

        weekly_path = os.path.join(out_dir, "2_2_weekly_methods_compare.csv")
        season_path = os.path.join(out_dir, "2_2_season_summary.csv")
        weekly_df.to_csv(weekly_path, index=False, encoding="utf-8-sig")
        season_df.to_csv(season_path, index=False, encoding="utf-8-sig")

        outputs = {"weekly_compare": weekly_path, "season_summary": season_path}

        if controversial_cases:
            celeb_df = self.build_controversy_summary(weekly_df, alpha=alpha, controversial_cases=controversial_cases)
            celeb_path = os.path.join(out_dir, "2_2_controversy_celebrity_summary.csv")
            celeb_df.to_csv(celeb_path, index=False, encoding="utf-8-sig")
            outputs["controversy_summary"] = celeb_path

        return outputs


def _parse_controversy_arg(arg: str) -> List[Tuple[int, str]]:
    """
    Parse: "2|Jerry Rice,4|Billy Ray Cyrus"
    """
    cases = []
    if not arg:
        return cases
    for item in str(arg).split(","):
        item = item.strip()
        if not item or "|" not in item:
            continue
        s, name = item.split("|", 1)
        try:
            cases.append((int(s), name.strip()))
        except Exception:
            continue
    return cases


def main():
    parser = argparse.ArgumentParser(description="Q2: controversy + judges-save rule comparison (tables only).")
    parser.add_argument("--judge", default="2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--fan", default="optimized_fan_votes.csv")
    parser.add_argument("--out", default=".")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--controversy",
        default="2|Jerry Rice,4|Billy Ray Cyrus,11|Bristol Palin,27|Bobby Bones",
        help="Comma-separated 'season|celebrity' list"
    )
    args = parser.parse_args()

    comp = Q2ComparatorV2(judge_data_path=args.judge, fan_votes_path=args.fan)
    cases = _parse_controversy_arg(args.controversy)
    outputs = comp.export_tables(out_dir=args.out, alpha=args.alpha, controversial_cases=cases)

    print("\n✅ Exported tables:")
    for k, v in outputs.items():
        print(f" - {k}: {v}")

        # ---- CLI 文字总结：反事实推演的“大致结论” ----
    season_df = pd.read_csv(outputs["season_summary"])

    print("\n📌 反事实推演：总体影响概览（按赛季平均）")

    avg_rank_vs_percent = season_df["rank_vs_percent_diff_rate"].mean()
    avg_rank_js = season_df["rank_js_change_rate"].mean()
    avg_percent_js = season_df["percent_js_change_rate"].mean()

    print(
        f"• Rank vs Percent：平均有 {avg_rank_vs_percent:.1%} 的周次淘汰结果不同"
    )
    print(
        f"• Rank + Judges Save：平均有 {avg_rank_js:.1%} 的周次发生改变"
    )
    print(
        f"• Percent + Judges Save：平均有 {avg_percent_js:.1%} 的周次发生改变"
    )

    print("\n🧠 简要解读：")
    if avg_rank_vs_percent < 0.1:
        print("• Rank 与 Percent 在多数情况下给出相同结果，方法选择影响有限")
    else:
        print("• Rank 与 Percent 在不少周次给出不同结果，方法选择具有实质影响")

    if avg_percent_js > avg_rank_js:
        print("• Judges Save 在 Percent 方法下影响更大，更容易改变淘汰结果")
    else:
        print("• Judges Save 在 Rank 方法下影响更明显")

    print("• Judges Save 整体上降低了粉丝投票的决定性，提高了评委干预程度")



if __name__ == "__main__":
    main()
