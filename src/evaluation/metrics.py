"""Candidate-level and observation-level metrics.

Candidate level (threshold 0.5 on P(correct)):
    accuracy, precision_macro, recall_macro, macro_f1, log_loss
    precision_pos / recall_pos  = positive class (is_correct = 1) only
Observation level (the metric that matters for map matching):
    obs_top1_all          Top-1 over ALL observations (a candidate-generation miss counts as wrong)
    obs_top1_recallable   Top-1 over observations whose true segment is among the candidates
    obs_top3_all          true candidate ranked in top 3
Ties in score are broken by nearest distance (dist_rank) if present, else by a seeded random key,
so results never depend on row order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from .. import config as C

METRIC_COLS = [
    "accuracy", "precision_macro", "recall_macro", "precision_pos", "recall_pos",
    "macro_f1", "log_loss", "obs_top1_all", "obs_top1_recallable", "obs_top3_all",
]


def candidate_metrics(y_true, p, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y_pred = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_pos": precision_score(y_true, y_pred, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
    }


def obs_codes(df: pd.DataFrame) -> np.ndarray:
    """Integer id per (trip_id, obs_id) - never use obs_id alone."""
    return df.groupby(C.OBS_KEYS, sort=False).ngroup().to_numpy()


def rank_within_obs(df: pd.DataFrame, scores, tiebreak=None, seed: int = 0) -> np.ndarray:
    """1-based rank of each candidate inside its observation (1 = highest score)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    codes = obs_codes(df)
    tb = np.asarray(tiebreak, dtype=float) if tiebreak is not None else np.random.default_rng(seed).random(n)
    order = np.lexsort((tb, -scores, codes))        # primary: obs, then score desc, then tiebreak asc
    sc = codes[order]
    new_group = np.r_[True, sc[1:] != sc[:-1]]
    starts = np.flatnonzero(new_group)
    grp = np.cumsum(new_group) - 1
    rank_sorted = np.arange(n) - starts[grp] + 1
    rank = np.empty(n, dtype=int)
    rank[order] = rank_sorted
    return rank


def normalise_within_obs(df: pd.DataFrame, scores) -> np.ndarray:
    """Scores rescaled to sum to 1 within each observation (uniform if all zero)."""
    s = np.asarray(scores, dtype=float)
    codes = obs_codes(df)
    tot = np.bincount(codes, weights=s)
    cnt = np.bincount(codes)
    out = np.where(tot[codes] > 0, s / np.where(tot[codes] > 0, tot[codes], 1), 1.0 / cnt[codes])
    return out


def observation_metrics(df: pd.DataFrame, scores, label_col: str = C.LABEL_COL,
                        tiebreak_col: str | None = "dist_rank", ks=(1, 3), seed: int = 0) -> dict:
    d = df.reset_index(drop=True)
    tb = d[tiebreak_col].to_numpy() if tiebreak_col and tiebreak_col in d else None
    rank = rank_within_obs(d, scores, tb, seed)
    codes = obs_codes(d)
    pos = d[label_col].to_numpy() == 1
    n_obs = int(codes.max() + 1)
    truth_rank = pd.Series(rank[pos]).groupby(codes[pos]).min()
    n_truth = int(len(truth_rank))
    out = {}
    for k in ks:
        hits = int((truth_rank <= k).sum())
        out[f"obs_top{k}_all"] = hits / n_obs
        out[f"obs_top{k}_recallable"] = hits / n_truth if n_truth else float("nan")
    out.update(n_obs=n_obs, n_obs_with_truth=n_truth, candidate_recall=n_truth / n_obs,
               mean_truth_rank=float(truth_rank.mean()) if n_truth else float("nan"))
    return out


def evaluate_scores(df: pd.DataFrame, p, threshold: float = 0.5) -> dict:
    out = candidate_metrics(df[C.LABEL_COL].to_numpy(), p, threshold)
    out.update(observation_metrics(df, p))
    return out


def per_trip_top1(df: pd.DataFrame, scores) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d["_score"] = np.asarray(scores, dtype=float)
    rows = []
    for trip, t in d.groupby(C.TRIP_COL, sort=True):
        m = observation_metrics(t.reset_index(drop=True), t["_score"].to_numpy())
        rows.append({C.TRIP_COL: trip, "n_obs": m["n_obs"], "obs_top1_all": m["obs_top1_all"],
                     "candidate_recall": m["candidate_recall"]})
    return pd.DataFrame(rows)
