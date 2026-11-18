"""Loading, validating and featurising candidate-level data from Member 1.

One row = one (trip_id, obs_id, candidate_id). The target is_correct says whether
that candidate is the reference road segment for that observation.

Important: obs_id and candidate_id are only unique inside a trip. Always use
(trip_id, obs_id) as the observation key, never obs_id alone.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def validate_schema(df: pd.DataFrame, source: str = "data", need_label: bool = True) -> None:
    required = C.REQUIRED_TRAIN_COLS if need_label else C.ID_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}. Found: {list(df.columns)}")
    if df[C.ID_COLS].isna().any().any():
        raise ValueError(f"{source}: null values in ID columns {C.ID_COLS}")
    dup = df.duplicated(C.ID_COLS).sum()
    if dup:
        raise ValueError(
            f"{source}: {dup} duplicated ({', '.join(C.ID_COLS)}) rows. "
            "Each candidate must appear once per observation per trip."
        )
    if need_label:
        bad = set(pd.unique(df[C.LABEL_COL].dropna())) - {0, 1, True, False}
        if bad or df[C.LABEL_COL].isna().any():
            raise ValueError(f"{source}: {C.LABEL_COL} must be 0/1 with no nulls (found {bad or 'nulls'})")
        n_pos = df.groupby(C.OBS_KEYS)[C.LABEL_COL].sum()
        if (n_pos > 1).any():
            warnings.warn(
                f"{source}: {(n_pos > 1).sum()} observations have more than one positive candidate. "
                "Top-1 counts a hit if the top-ranked candidate is any positive."
            )


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Member 1's raw columns (observation_id, distance_m, ...) to the contract names."""
    ren = {old: new for old, new in C.ALIASES.items() if old in df.columns and new not in df.columns}
    return df.rename(columns=ren) if ren else df


def load_candidates(path: str | Path, need_label: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Member 1 must supply this file (docs/DATA_CONTRACT.md).")
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        # trip ids like "00000001" must stay strings (leading zeros, consistent across files)
        df = pd.read_csv(path, dtype={C.TRIP_COL: str})
    df = apply_aliases(df)
    validate_schema(df, source=str(path), need_label=need_label)
    if need_label:
        df[C.LABEL_COL] = df[C.LABEL_COL].astype(int)
    return df.reset_index(drop=True)


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-observation features (uses only candidates of the same observation,
    so it is available at inference time and does not leak the label).
    Existing columns with the same name (e.g. supplied by Member 1) are kept as-is.
    Also adds obs_has_truth (from the label) if the label exists; it is NOT a feature.
    """
    new: dict[str, pd.Series] = {}
    g = df.groupby(C.OBS_KEYS, sort=False)
    if "n_candidates" not in df:
        new["n_candidates"] = g[C.CAND_COL].transform("size")
    if C.DIST_COL in df:
        dmin = g[C.DIST_COL].transform("min")
        if "dist_minus_min" not in df:
            new["dist_minus_min"] = df[C.DIST_COL] - dmin
        if "dist_ratio_to_min" not in df:
            new["dist_ratio_to_min"] = df[C.DIST_COL] / (dmin + 1.0)
        if "dist_rank" not in df:
            new["dist_rank"] = g[C.DIST_COL].rank(method="first")
    if C.HEADING_COL in df:
        hmin = g[C.HEADING_COL].transform("min")
        if "heading_minus_min" not in df:
            new["heading_minus_min"] = df[C.HEADING_COL] - hmin
        if "heading_rank" not in df:
            new["heading_rank"] = g[C.HEADING_COL].rank(method="first")
    if C.LABEL_COL in df and C.TRUTH_FLAG_COL not in df:
        new[C.TRUTH_FLAG_COL] = g[C.LABEL_COL].transform("max").astype(int)
    if not new:
        return df.copy()
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def infer_feature_columns(df: pd.DataFrame, explicit: list[str] | None = None,
                          verbose: bool = True) -> list[str]:
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Requested features not in data: {missing}")
        return list(explicit)
    feats, skipped = [], []
    for c in df.columns:
        if c in C.NON_FEATURE_COLS or C.LEAKAGE_PATTERN.search(c):
            skipped.append(c)
        elif pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
        else:
            skipped.append(c)
    if verbose:
        print(f"[features] using {len(feats)}: {feats}")
        print(f"[features] excluded: {skipped}")
    if not feats:
        raise ValueError("No numeric feature columns found.")
    return feats


def feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].astype("float64")
    return X.replace([np.inf, -np.inf], np.nan)


def training_rows(df: pd.DataFrame, drop_unrecallable: bool = True) -> pd.DataFrame:
    """Observations where no candidate is correct carry no ranking signal for training."""
    if drop_unrecallable and C.TRUTH_FLAG_COL in df:
        return df[df[C.TRUTH_FLAG_COL] == 1]
    return df


def candidate_recall_report(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """How often does candidate generation contain the true segment at all?
    An upper bound for observation-level Top-1: the classifier cannot fix a miss."""
    d = add_context_features(df)
    obs = d.groupby(C.OBS_KEYS, sort=False).agg(
        n_cand=(C.CAND_COL, "size"), has_truth=(C.LABEL_COL, "max"))
    summary = {
        "n_trips": int(d[C.TRIP_COL].nunique()),
        "n_observations": int(len(obs)),
        "n_candidate_rows": int(len(d)),
        "obs_with_true_candidate": int(obs.has_truth.sum()),
        "candidate_recall": float(obs.has_truth.mean()),
        "mean_candidates_per_obs": float(obs.n_cand.mean()),
        "median_candidates_per_obs": float(obs.n_cand.median()),
        "max_candidates_per_obs": int(obs.n_cand.max()),
        "positive_rate_rows": float(d[C.LABEL_COL].mean()),
    }
    if "dist_rank" in d:
        tr = d.loc[d[C.LABEL_COL] == 1, "dist_rank"]
        for k in (1, 2, 3, 5):
            summary[f"truth_within_{k}_nearest_of_all_obs"] = float((tr <= k).sum() / len(obs))
    per_trip = d.groupby(C.TRIP_COL).apply(
        lambda t: pd.Series({
            "n_obs": t.groupby(C.OBS_COL).ngroups,
            "recall": t.groupby(C.OBS_COL)[C.LABEL_COL].max().mean(),
            "mean_candidates": t.groupby(C.OBS_COL).size().mean(),
        }), include_groups=False).reset_index()
    return summary, per_trip
