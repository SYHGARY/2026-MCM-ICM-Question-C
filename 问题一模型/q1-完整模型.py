import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ProjectedFanVoteEstimatorMCMPlus")

# =============================================================================
# Helpers (keep same behavior as original)
# =============================================================================
def average_tie_ranks_desc(values: np.ndarray) -> np.ndarray:
    """
    Rank with average ties for descending values.
    Highest value => rank 1. Returns float ranks.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return v

    order = np.argsort(-v, kind="stable")
    sorted_v = v[order]

    ranks = np.zeros(n, dtype=float)
    cur = 1
    i = 0
    while i < n:
        j = i
        while j < n and np.isclose(sorted_v[j], sorted_v[i]):
            j += 1
        avg_rank = cur + (j - i - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        cur += (j - i)
        i = j
    return ranks


def project_to_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    """
    Euclidean projection onto simplex {w >= 0, sum w = s}.
    Duchi et al. algorithm.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return v
    if s <= 0:
        return np.zeros_like(v)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0]
    if len(rho) == 0:
        theta = (cssv[-1] - s) / n
    else:
        rho = rho[-1]
        theta = (cssv[rho] - s) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    if w.sum() > 0:
        w *= (s / w.sum())
    return w


def project_with_lower_bound(v: np.ndarray, eps: float) -> np.ndarray:
    """
    Project onto {p_i >= eps, sum p_i = 1}.
    Equivalent: p = q + eps, q>=0, sum q = 1 - n*eps.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    s = 1.0 - n * eps
    if s <= 0:
        return np.ones(n) / n
    q = project_to_simplex(v - eps, s=s)
    return q + eps


def _pairwise_constraints_for_elim(total: np.ndarray, elim_local: List[int], non_elim_local: List[int], tol: float = 1e-12):
    """
    Utility: check if elim are all <= all non-elim (with tolerance), i.e. elim are worst in 'smaller is worse' scoring.
    """
    if not elim_local or not non_elim_local:
        return True
    return np.all(total[np.array(elim_local)] <= np.min(total[np.array(non_elim_local)]) + tol)


# =============================================================================
# Season Config (same fields to keep compatibility)
# =============================================================================
@dataclass
class SeasonConfig:
    season: int
    method: str  # 'rank' or 'percent'
    max_week: int
    contestants: List[str]
    contestant_to_idx: Dict[str, int]
    idx_to_contestant: Dict[int, str]
    weekly_contestants: Dict[int, List[str]]
    weekly_indices: Dict[int, List[int]]
    judge_scores: np.ndarray  # (N, W)
    elimination_weeks: Dict[int, List[str]]
    special_rules: bool = False  # seasons 28-34 bottom-2


# =============================================================================
# Main estimator (OUTPUT-COMPATIBLE with your original q1.py)
# - Still produces:
#   optimized_fan_votes.csv with columns:
#       season, celebrity, week, fan_votes, judge_score, eliminated
#   optimized_fan_votes_summary.csv with columns:
#       season, method, special, n_contestants, n_weeks, n_eliminations,
#       consistency, repair_failed_weeks, smooth_iters
# =============================================================================
class ProjectedFanVoteEstimatorMCMPlus:
    """
    Upgraded MCM-friendly model while keeping ORIGINAL output format.

    Improvements vs V3:
    - Percent seasons (3-27): solve a constrained optimization each week to enforce eliminations as HARD constraints.
      We use projected gradient on a convex objective:
          min ||p - prior||^2 + lambda * ||p - p_prev||^2
          s.t. p >= eps, sum p = 1, and (judge+p)_elim <= (judge+p)_non_elim for all pairs.
      (Hard elimination consistency; no repair loops.)

    - Rank seasons (1-2, 28-34): keep a robust feasibility search via Dirichlet sampling around a prior,
      selecting the best feasible point under a smooth objective. This is practical and stable for MCM.
    """

    def __init__(
        self,
        data_path: str = "2026_MCM_Problem_C_Data.csv",
        eps: float = 1e-4,
        votes_scale: float = 10000.0,
        smooth_alpha: float = 0.35,   # kept for cross-week smoothing of priors
        prior_beta: float = 0.02,     # kept for mixing judge-based prior
        lambda_temporal: float = 0.8, # weight on p_prev in weekly objective (percent)
        pg_iters: int = 600,          # projected-gradient iterations per week (percent)
        pg_lr: float = 0.15,          # step size (percent)
        rank_samples: int = 6000,     # feasibility samples per week (rank)
        rank_keep_best: int = 40,     # local refinements on best samples
        random_seed: int = 7,
    ):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.season_configs: Dict[int, SeasonConfig] = {}

        self.eps = float(eps)
        self.votes_scale = float(votes_scale)

        self.smooth_alpha = float(smooth_alpha)
        self.prior_beta = float(prior_beta)

        self.lambda_temporal = float(lambda_temporal)
        self.pg_iters = int(pg_iters)
        self.pg_lr = float(pg_lr)

        self.rank_samples = int(rank_samples)
        self.rank_keep_best = int(rank_keep_best)

        self.rng = np.random.default_rng(int(random_seed))

    # ---------- Load & preprocess (same logic as original) ----------
    def load_data(self) -> "ProjectedFanVoteEstimatorMCMPlus":
        logger.info("1) Loading data...")
        self.df = pd.read_csv(self.data_path)
        self.df.replace("N/A", np.nan, inplace=True)
        self._calculate_judge_scores()
        self._build_season_configs()
        logger.info("Loaded: %d rows, %d seasons", len(self.df), len(self.season_configs))
        return self

    def _calculate_judge_scores(self) -> None:
        for week in range(1, 12):
            judge_cols = [f"week{week}_judge{i}_score" for i in range(1, 5)]
            available = [c for c in judge_cols if c in self.df.columns]
            if available:
                self.df[f"week{week}_total_score"] = self.df[available].sum(axis=1, min_count=1)

    def _infer_max_week(self, season_df: pd.DataFrame) -> int:
        max_week = 0
        for week in range(1, 12):
            col = f"week{week}_total_score"
            if col in season_df.columns:
                s = season_df[col]
                if s.notna().any() and (s.fillna(0) > 0).any():
                    max_week = week
        return max_week

    def _parse_elimination_week(self, result_str: str) -> Optional[int]:
        m = re.search(r"Eliminated\s*Week\s*(\d+)", result_str, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        if "eliminat" in result_str.lower():
            m2 = re.search(r"Week\s*(\d+)", result_str, flags=re.IGNORECASE)
            if m2:
                try:
                    return int(m2.group(1))
                except Exception:
                    return None
        return None

    def _is_top3_result(self, result_str: str) -> bool:
        s = result_str.lower()
        top_markers = ["winner", "1st", "first", "runner-up", "2nd", "second", "3rd", "third"]
        return any(m in s for m in top_markers)

    def _build_season_configs(self) -> None:
        logger.info("2) Building season configs...")
        for season in sorted(self.df["season"].dropna().unique()):
            season = int(season)
            season_df = self.df[self.df["season"] == season].copy()

            if season in (1, 2):
                method = "rank"
                special_rules = False
            elif 3 <= season <= 27:
                method = "percent"
                special_rules = False
            elif 28 <= season <= 34:
                method = "rank"
                special_rules = True
            else:
                continue

            max_week = self._infer_max_week(season_df)
            if max_week <= 0:
                continue

            contestants = []
            for _, row in season_df.iterrows():
                name = row["celebrity_name"]
                for week in range(1, max_week + 1):
                    col = f"week{week}_total_score"
                    if col in row and pd.notna(row.get(col)) and float(row[col]) > 0:
                        contestants.append(name)
                        break
            contestants = sorted(set(contestants))
            if not contestants:
                continue

            c2i = {n: i for i, n in enumerate(contestants)}
            i2c = {i: n for n, i in c2i.items()}

            weekly_contestants: Dict[int, List[str]] = {}
            weekly_indices: Dict[int, List[int]] = {}
            for week in range(1, max_week + 1):
                col = f"week{week}_total_score"
                names = []
                for name in contestants:
                    row = season_df[season_df["celebrity_name"] == name]
                    if row.empty or col not in row.columns:
                        continue
                    v = row[col].iloc[0]
                    if pd.notna(v) and float(v) > 0:
                        names.append(name)
                if names:
                    weekly_contestants[week] = names
                    weekly_indices[week] = [c2i[n] for n in names]

            if not weekly_contestants:
                continue

            N = len(contestants)
            judge_scores = np.full((N, max_week), np.nan)
            for name in contestants:
                idx = c2i[name]
                row = season_df[season_df["celebrity_name"] == name]
                if row.empty:
                    continue
                for week in range(1, max_week + 1):
                    col = f"week{week}_total_score"
                    if col in row.columns:
                        v = row[col].iloc[0]
                        if pd.notna(v):
                            judge_scores[idx, week - 1] = float(v)

            elimination_weeks: Dict[int, List[str]] = {}
            for _, row in season_df.iterrows():
                res = row.get("results", np.nan)
                if pd.isna(res):
                    continue
                res_str = str(res)
                name = row["celebrity_name"]

                elim_w = self._parse_elimination_week(res_str)
                if elim_w is not None and 1 <= elim_w <= max_week:
                    elimination_weeks.setdefault(elim_w, [])
                    if name not in elimination_weeks[elim_w]:
                        elimination_weeks[elim_w].append(name)
                    continue

                if "withdrew" in res_str.lower() or "withdraw" in res_str.lower():
                    row2 = season_df[season_df["celebrity_name"] == name]
                    if row2.empty:
                        continue
                    last_week = 0
                    for week in range(1, max_week + 1):
                        col = f"week{week}_total_score"
                        if col in row2.columns:
                            v = row2[col].iloc[0]
                            if pd.notna(v) and float(v) > 0:
                                last_week = week
                    if last_week > 0:
                        elimination_weeks.setdefault(last_week, [])
                        if name not in elimination_weeks[last_week]:
                            elimination_weeks[last_week].append(name)

            last_w = max_week
            if last_w in weekly_contestants and len(weekly_contestants[last_w]) > 3:
                elimination_weeks.setdefault(last_w, [])
                for name in weekly_contestants[last_w]:
                    row = season_df[season_df["celebrity_name"] == name]
                    if row.empty:
                        continue
                    res = row.get("results", pd.Series([np.nan])).iloc[0]
                    res_str = "" if pd.isna(res) else str(res)
                    if not self._is_top3_result(res_str):
                        if name not in elimination_weeks[last_w]:
                            elimination_weeks[last_w].append(name)

            cfg = SeasonConfig(
                season=season,
                method=method,
                max_week=max_week,
                contestants=contestants,
                contestant_to_idx=c2i,
                idx_to_contestant=i2c,
                weekly_contestants=weekly_contestants,
                weekly_indices=weekly_indices,
                judge_scores=judge_scores,
                elimination_weeks=elimination_weeks,
                special_rules=special_rules,
            )
            self.season_configs[season] = cfg
            logger.info(
                "Season %d: %d contestants, %d weeks, elim_weeks=%d, method=%s, special=%s",
                season, len(contestants), max_week, len(elimination_weeks), method, special_rules
            )

    # ---------- Judge derived ----------
    def judge_percent(self, cfg: SeasonConfig, week: int) -> np.ndarray:
        out = np.zeros(len(cfg.contestants), dtype=float)
        if week not in cfg.weekly_indices:
            return out
        idxs = cfg.weekly_indices[week]
        scores = np.array([cfg.judge_scores[i, week - 1] for i in idxs], dtype=float)
        scores = np.where(np.isnan(scores), 0.0, scores)
        s = scores.sum()
        out[idxs] = (scores / s) if s > 0 else (1.0 / len(idxs))
        return out

    def judge_rank(self, cfg: SeasonConfig, week: int) -> np.ndarray:
        out = np.full(len(cfg.contestants), np.nan, dtype=float)
        if week not in cfg.weekly_indices:
            return out
        idxs = cfg.weekly_indices[week]
        scores = np.array([cfg.judge_scores[i, week - 1] for i in idxs], dtype=float)
        ranks_local = average_tie_ranks_desc(scores)
        for pos, idx in enumerate(idxs):
            out[idx] = ranks_local[pos]
        return out

    # ---------- Init weekly shares ----------
    def _init_week_shares(self, cfg: SeasonConfig) -> Dict[int, np.ndarray]:
        """
        Same initialization idea as V3: start near judge performance.
        """
        p: Dict[int, np.ndarray] = {}
        for w in sorted(cfg.weekly_indices.keys()):
            idxs = cfg.weekly_indices[w]
            n = len(idxs)
            base = np.ones(n) / n

            scores = np.array([cfg.judge_scores[i, w - 1] for i in idxs], dtype=float)
            if np.isnan(scores).all():
                v = base
            else:
                scores = np.where(np.isnan(scores), np.nanmean(scores), scores)
                z = scores - np.min(scores)
                v = (z / np.sum(z)) if np.max(z) > 0 else base

            v = project_with_lower_bound(v, self.eps)
            full = np.zeros(len(cfg.contestants), dtype=float)
            full[idxs] = v
            p[w] = full
        return p

    # ---------- Cross-week smoothing & prior pull (kept to preserve "look" of results) ----------
    def _smooth_step(self, cfg: SeasonConfig, p: Dict[int, np.ndarray]) -> None:
        weeks = sorted(p.keys())
        if len(weeks) <= 1:
            return

        new_p = {}
        for w in weeks:
            idxs = cfg.weekly_indices[w]
            v = p[w][idxs].copy()

            pulls = []
            if (w - 1) in p:
                common = sorted(set(cfg.weekly_indices[w]).intersection(cfg.weekly_indices[w - 1]))
                if common:
                    loc = [idxs.index(i) for i in common]
                    pulls.append((np.array(loc), p[w - 1][common]))
            if (w + 1) in p:
                common = sorted(set(cfg.weekly_indices[w]).intersection(cfg.weekly_indices[w + 1]))
                if common:
                    loc = [idxs.index(i) for i in common]
                    pulls.append((np.array(loc), p[w + 1][common]))

            if pulls:
                v2 = v.copy()
                for loc, neigh_vals in pulls:
                    v2[loc] = (1.0 - self.smooth_alpha) * v2[loc] + self.smooth_alpha * neigh_vals
                v = v2

            v = project_with_lower_bound(v, self.eps)
            full = np.zeros(len(cfg.contestants), dtype=float)
            full[idxs] = v
            new_p[w] = full

        p.update(new_p)

    def _prior_pull_step(self, cfg: SeasonConfig, p: Dict[int, np.ndarray]) -> None:
        for w in sorted(p.keys()):
            idxs = cfg.weekly_indices[w]
            prior = self.judge_percent(cfg, w)[idxs]
            cur = p[w][idxs]
            mixed = (1.0 - self.prior_beta) * cur + self.prior_beta * prior
            mixed = project_with_lower_bound(mixed, self.eps)
            full = np.zeros(len(cfg.contestants), dtype=float)
            full[idxs] = mixed
            p[w] = full

    # ---------- Percent-week solver (hard constraints) ----------
    def _solve_percent_week(
        self,
        judge_share: np.ndarray,
        prior: np.ndarray,
        p_prev: Optional[np.ndarray],
        elim_local: List[int],
        eps: float
    ) -> np.ndarray:
        """
        Solve:
            min ||p-prior||^2 + lambda||p-p_prev||^2
            s.t. p>=eps, sum p=1, (judge+p)_elim <= (judge+p)_non_elim for all pairs.
        We use projected gradient with penalty for violated constraints.
        """
        n = len(prior)
        if n == 0:
            return prior

        p = prior.copy()
        if p_prev is not None and len(p_prev) == n:
            p = (1.0 - self.lambda_temporal) * p + self.lambda_temporal * p_prev
        p = project_with_lower_bound(p, eps)

        non_elim = [i for i in range(n) if i not in elim_local]
        # penalty weight (large to enforce hard-like behavior)
        mu = 80.0

        for _ in range(self.pg_iters):
            # objective gradient
            grad = 2.0 * (p - prior)
            if p_prev is not None:
                grad += 2.0 * self.lambda_temporal * (p - p_prev)

            total = judge_share + p

            # hinge penalties for each violated pair: (total_elim - total_non_elim) <= 0
            if elim_local and non_elim:
                # compute max violation per elim against best (smallest) non-elim
                min_non = float(np.min(total[non_elim]))
                for e in elim_local:
                    viol = total[e] - min_non
                    if viol > 0:
                        # increase p for non-elim side not tracked; simplest: push p_e down
                        # d/dp_e of viol is 1
                        grad[e] += 2.0 * mu * viol

            # gradient step
            p = p - self.pg_lr * grad
            p = project_with_lower_bound(p, eps)

            # early stop if constraints satisfied
            total = judge_share + p
            if _pairwise_constraints_for_elim(total, elim_local, non_elim, tol=1e-10):
                # a little extra to stabilize
                break

        # final polish: if still tiny violations, do a deterministic fix by shifting mass from elim to worst non-elim
        total = judge_share + p
        if elim_local and non_elim and (not _pairwise_constraints_for_elim(total, elim_local, non_elim, tol=1e-10)):
            # choose the most-violating elim, and the smallest-total non-elim
            for __ in range(2000):
                total = judge_share + p
                min_non_idx = non_elim[int(np.argmin(total[non_elim]))]
                # find elim with largest violation
                viols = [(e, total[e] - total[min_non_idx]) for e in elim_local]
                e, v = max(viols, key=lambda x: x[1])
                if v <= 1e-10:
                    break
                delta = min(0.02, p[e] - eps)
                if delta <= 1e-12:
                    break
                p[e] -= delta
                p[min_non_idx] += delta
                p = project_with_lower_bound(p, eps)

        return p

    # ---------- Rank-week feasibility search ----------
    def _solve_rank_week(
        self,
        judge_ranks: np.ndarray,
        prior: np.ndarray,
        p_prev: Optional[np.ndarray],
        elim_local: List[int],
        k: int,
        eps: float
    ) -> Tuple[np.ndarray, bool]:
        """
        For rank regime, we use sampling around prior (and optionally p_prev) and pick the best feasible sample.

        Feasibility (tie-aware): eliminated must be within bottom-k of (judge_rank + fan_rank),
        where fan_rank is computed from share p (descending = rank1 best).
        """
        n = len(prior)
        if n == 0:
            return prior, True

        # center
        center = prior.copy()
        if p_prev is not None and len(p_prev) == n:
            center = 0.6 * center + 0.4 * p_prev
        center = project_with_lower_bound(center, eps)

        # convert center to Dirichlet alpha; bigger => more concentrated
        conc = 35.0
        alpha = np.maximum(center * conc, 1e-3)

        def is_feasible(p: np.ndarray) -> bool:
            fan_r = average_tie_ranks_desc(p)
            total = judge_ranks + fan_r  # larger is worse
            order = np.argsort(-total, kind="stable")  # largest worst
            worst = set(order[:k])
            return all(e in worst for e in elim_local)

        def score(p: np.ndarray) -> float:
            # smaller is better
            s = float(np.sum((p - center) ** 2))
            if p_prev is not None:
                s += 0.3 * float(np.sum((p - p_prev) ** 2))
            return s

        best_p = center
        best_s = score(center)
        best_ok = is_feasible(center)

        # sampling
        for _ in range(self.rank_samples):
            p = self.rng.dirichlet(alpha)
            p = project_with_lower_bound(p, eps)
            ok = is_feasible(p)
            if ok:
                s = score(p)
                if (not best_ok) or (s < best_s):
                    best_ok = True
                    best_s = s
                    best_p = p

        # small local refinements on feasible best (optional)
        if best_ok and self.rank_keep_best > 0:
            p = best_p.copy()
            for _ in range(self.rank_keep_best):
                noise = self.rng.normal(0.0, 0.01, size=n)
                cand = project_with_lower_bound(p + noise, eps)
                if is_feasible(cand) and score(cand) <= score(p):
                    p = cand
            best_p = p

        return best_p, best_ok

    # ---------- Estimate season (keeps return signature) ----------
    def estimate_season(self, season: int, prior_jitter: float = 0.0) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[str, float]]:
        cfg = self.season_configs[season]
        p = self._init_week_shares(cfg)

        # We keep smoothing/prior-pull iterations for "look" consistency, but no repair.
        it = 0
        for it in range(12):  # fewer iterations needed now
            self._smooth_step(cfg, p)
            if self.prior_beta > 0:
                self._prior_pull_step(cfg, p)

        # Now solve each week with hard regime-specific constraints
        weeks = sorted(cfg.weekly_indices.keys())
        for w in weeks:
            idxs = cfg.weekly_indices[w]
            n = len(idxs)
            if n == 0:
                continue

            prior_local = p[w][idxs].copy()
            # Optional: perturb the local prior to probe determinacy/uncertainty
            if prior_jitter and prior_jitter > 0:
                prior_local = project_with_lower_bound(prior_local + self.rng.normal(0.0, float(prior_jitter), size=n), self.eps)
            p_prev_local = None
            if (w - 1) in p:
                common = sorted(set(cfg.weekly_indices[w]).intersection(cfg.weekly_indices[w - 1]))
                if common:
                    # align p_prev to current local ordering
                    loc_map = {g: j for j, g in enumerate(idxs)}
                    prev_vals = np.zeros(n)
                    for g in common:
                        prev_vals[loc_map[g]] = p[w - 1][g]
                    # for non-common, use prior
                    for j in range(n):
                        if prev_vals[j] <= 0:
                            prev_vals[j] = prior_local[j]
                    p_prev_local = project_with_lower_bound(prev_vals, self.eps)

            elim_names = [e for e in cfg.elimination_weeks.get(w, []) if e in cfg.weekly_contestants.get(w, [])]
            elim_local = []
            for e in elim_names:
                gi = cfg.contestant_to_idx[e]
                try:
                    elim_local.append(idxs.index(gi))
                except ValueError:
                    pass

            if cfg.method == "percent":
                judge_share = self.judge_percent(cfg, w)[idxs]
                p_local = self._solve_percent_week(
                    judge_share=judge_share,
                    prior=prior_local,
                    p_prev=p_prev_local,
                    elim_local=elim_local,
                    eps=self.eps
                )
                full = np.zeros(len(cfg.contestants), dtype=float)
                full[idxs] = p_local
                p[w] = full
            else:
                judge_r = self.judge_rank(cfg, w)[idxs]
                k = 2 if cfg.special_rules else max(1, len(elim_local))
                k = min(k, n)
                p_local, ok = self._solve_rank_week(
                    judge_ranks=judge_r,
                    prior=prior_local,
                    p_prev=p_prev_local,
                    elim_local=elim_local,
                    k=k,
                    eps=self.eps
                )
                if not ok and elim_local:
                    logger.info("Season %d Week %d: rank feasibility not found (kept best approximation).", season, w)
                full = np.zeros(len(cfg.contestants), dtype=float)
                full[idxs] = p_local
                p[w] = full

        # Build votes matrix F (same as original)
        N = len(cfg.contestants)
        W = cfg.max_week
        F = np.zeros((N, W), dtype=float)
        for w in sorted(p.keys()):
            F[:, w - 1] = p[w] * self.votes_scale

        info = {"smooth_iters": float(it + 1), "repair_failed_weeks": 0.0}
        return F, p, info

    # ---------- Validate (keep same logic, but tie-aware threshold to avoid false negatives) ----------
    def validate_consistency(self, season: int, F: np.ndarray) -> Tuple[float, List[str]]:
        cfg = self.season_configs[season]
        details = []
        ok = 0
        tot = 0

        for w in sorted(cfg.elimination_weeks.keys()):
            participants = cfg.weekly_contestants.get(w, [])
            if not participants:
                continue
            elim_names = [e for e in cfg.elimination_weeks[w] if e in participants]
            if not elim_names:
                continue

            idxs = [cfg.contestant_to_idx[n] for n in participants]
            fan_votes = F[idxs, w - 1]
            fan_share = fan_votes / max(np.sum(fan_votes), 1e-12)

            if cfg.method == "percent":
                judge = self.judge_percent(cfg, w)[idxs]
                total = judge + fan_share
                k = len(elim_names)
                # tie-aware: threshold at k-th smallest
                order = np.argsort(total, kind="stable")
                thresh = total[order[min(k - 1, len(order) - 1)]]
                worst = [participants[i] for i in range(len(participants)) if total[i] <= thresh + 1e-12]
            else:
                judge_r = self.judge_rank(cfg, w)[idxs]
                fan_r = average_tie_ranks_desc(fan_share)
                total = judge_r + fan_r
                k = 2 if cfg.special_rules else len(elim_names)
                order = np.argsort(-total, kind="stable")
                thresh = total[order[min(k - 1, len(order) - 1)]]
                worst = [participants[i] for i in range(len(participants)) if total[i] >= thresh - 1e-12]

            tot += 1
            good = all(e in worst for e in elim_names)
            if good:
                ok += 1
                details.append(f"Week {w}: eliminated {elim_names} in bottom-{k} {worst} ✓")
            else:
                details.append(f"Week {w}: eliminated {elim_names} NOT in bottom-{k} {worst} ✗")

        return (ok / tot if tot > 0 else 1.0), details

    # ---------- Output (SAME columns as original) ----------
    def generate_output_table(self, season_results: Dict[int, Dict]) -> pd.DataFrame:
        rows = []
        for season, result in season_results.items():
            cfg = self.season_configs[season]
            F = result["F"]
            for w in range(1, cfg.max_week + 1):
                if w not in cfg.weekly_contestants:
                    continue
                participants = set(cfg.weekly_contestants[w])
                for i, name in enumerate(cfg.contestants):
                    if name not in participants:
                        continue
                    judge_score = cfg.judge_scores[i, w - 1]
                    eliminated = name in cfg.elimination_weeks.get(w, [])
                    rows.append({
                        "season": season,
                        "celebrity": name,
                        "week": w,
                        "fan_votes": float(F[i, w - 1]),
                        "judge_score": float(judge_score) if np.isfinite(judge_score) else 0.0,
                        "eliminated": bool(eliminated),
                    })
        return pd.DataFrame(rows)

    # ---------- Run (SAME return + SAME output filenames) ----------
    def run(self, seasons: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
        if seasons is None:
            seasons = sorted(self.season_configs.keys())

        results: Dict[int, Dict] = {}
        summary = []

        for s in seasons:
            if s not in self.season_configs:
                continue
            cfg = self.season_configs[s]
            logger.info("=== Season %d (%s) ===", s, cfg.method)

            F, p, info = self.estimate_season(s)
            consistency, details = self.validate_consistency(s, F)

            results[s] = {"F": F, "p": p, "info": info, "consistency": consistency, "details": details}

            summary.append({
                "season": s,
                "method": cfg.method,
                "special": cfg.special_rules,
                "n_contestants": len(cfg.contestants),
                "n_weeks": cfg.max_week,
                "n_eliminations": sum(len(v) for v in cfg.elimination_weeks.values()),
                "consistency": consistency,
                "repair_failed_weeks": int(info.get("repair_failed_weeks", 0.0)),
                "smooth_iters": int(info.get("smooth_iters", 0.0)),
            })

            logger.info("Season %d consistency: %.2f%%", s, 100 * consistency)
            for d in details:
                if "✗" in d:
                    logger.info("  %s", d)

        df_out = self.generate_output_table(results)
        df_sum = pd.DataFrame(summary)

        logger.info("\nSummary:\n%s", df_sum.to_string(index=False))
        if not df_sum.empty:
            logger.info("Average consistency: %.2f%%", 100 * df_sum["consistency"].mean())

        # Keep EXACT same filenames as your original script
        df_out.to_csv("optimized_fan_votes.csv", index=False)
        df_sum.to_csv("optimized_fan_votes_summary.csv", index=False)
        logger.info("Saved: optimized_fan_votes.csv and optimized_fan_votes_summary.csv")

        return df_out, results




    # -------------------------------------------------------------------------
    # Uncertainty / determinacy diagnostics (ENSEMBLE-BASED)
    # -------------------------------------------------------------------------
    def run_uncertainty(
        self,
        seasons: Optional[List[int]] = None,
        n_runs: int = 50,
        prior_jitter: float = 0.02,
        seed_base: int = 7,
        out_csv: str = "optimized_fan_votes_uncertainty.csv",
    ) -> pd.DataFrame:
        """
        Provide a practical uncertainty measure for the reconstructed fan votes.

        NOTE: This model reconstructs (infers) fan-vote distributions that are
        CONSISTENT with observed eliminations under the given rules. In general
        there can be many consistent solutions, so we quantify uncertainty via an
        ensemble: rerun the reconstruction with small perturbations of the priors
        (and different RNG seeds for rank seasons), then summarize variability.

        Output columns:
          season, celebrity, week,
          fan_votes_mean, fan_votes_std, fan_votes_p05, fan_votes_p95, n_runs
        """
        if seasons is None:
            seasons = sorted(self.season_configs.keys())

        all_runs = []

        for r in range(int(n_runs)):
            self.rng = np.random.default_rng(int(seed_base) + r)

            results_run: Dict[int, Dict] = {}
            for s in seasons:
                if s not in self.season_configs:
                    continue
                F, p, info = self.estimate_season(s, prior_jitter=float(prior_jitter))
                results_run[s] = {"F": F, "p": p, "info": info}

            df_r = self.generate_output_table(results_run)
            df_r["run"] = r
            all_runs.append(df_r[["season", "celebrity", "week", "fan_votes", "run"]])

        df_all = pd.concat(all_runs, ignore_index=True)

        def q05(x): return float(np.quantile(x, 0.05))
        def q95(x): return float(np.quantile(x, 0.95))

        df_u = (
            df_all.groupby(["season", "celebrity", "week"], as_index=False)
                  .agg(
                      fan_votes_mean=("fan_votes", "mean"),
                      fan_votes_std=("fan_votes", "std"),
                      fan_votes_p05=("fan_votes", q05),
                      fan_votes_p95=("fan_votes", q95),
                      n_runs=("fan_votes", "count"),
                  )
        )

        df_u.to_csv(out_csv, index=False)
        logger.info("Saved uncertainty table: %s", out_csv)
        return df_u

# =============================================================================
# Main: keep the SAME train/test split + prints as your original q1.py
# =============================================================================
if __name__ == "__main__":

    estimator = ProjectedFanVoteEstimatorMCMPlus(
        data_path="2026_MCM_Problem_C_Data.csv",
        eps=1e-4,
        votes_scale=10000.0,
        smooth_alpha=0.35,
        prior_beta=0.02,
        lambda_temporal=0.8,
        pg_iters=600,
        pg_lr=0.15,
        rank_samples=6000,
        rank_keep_best=40,
        random_seed=7,
    )

    estimator.load_data()

    all_seasons = sorted(estimator.season_configs.keys())
    train_seasons = [s for s in all_seasons if s <= 24]
    test_seasons  = [s for s in all_seasons if s > 24]

    print("\n=======================================")
    print("USING MCM+ RULES (HARD CONSTRAINTS FOR PERCENT; FEASIBLE SEARCH FOR RANK)")
    print("---------------------------------------")
    print("Train seasons:", train_seasons)
    print("Test  seasons:", test_seasons)
    print("=======================================\n")

    print("===== RUN ALL SEASONS (for full CSV output) =====")
    _, all_results = estimator.run(all_seasons)

    # ===== Uncertainty / Certainty Metrics (Ensemble) =====
    df_u = estimator.run_uncertainty(
    seasons=all_seasons,
    n_runs=50,          # 建议 30~80；先 30/50 跑通
    prior_jitter=0.02,  # percent 周先验扰动强度：0.01~0.03 常用
    seed_base=7,
    out_csv="optimized_fan_votes_uncertainty.csv",)
    print("\nSaved uncertainty metrics -> optimized_fan_votes_uncertainty.csv")


    # 下面只做统计/打印，不再调用 run()
    train_consistencies = [all_results[s]["consistency"] for s in train_seasons if s in all_results]
    test_consistencies  = [all_results[s]["consistency"] for s in test_seasons  if s in all_results]

    train_avg = float(np.mean(train_consistencies)) if train_consistencies else 1.0
    test_avg  = float(np.mean(test_consistencies))  if test_consistencies  else 1.0

    print("\n=======================================")
    print("FINAL TRAIN / TEST SUMMARY (MCM+)")
    print("=======================================")
    print(f"Train seasons: {train_seasons[0]}–{train_seasons[-1]}")
    print(f"Test  seasons: {test_seasons[0]}–{test_seasons[-1]}")
    print("---------------------------------------")
    print(f"Train average consistency: {train_avg:.3f}")
    print(f"Test  average consistency: {test_avg:.3f}")
    print("=======================================\n")

    print("NOTE: optimized_fan_votes.csv now contains ALL seasons (1–34).")

