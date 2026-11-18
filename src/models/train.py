"""Train + evaluate all five models with GroupKFold(trip_id), then evaluate on the
ten student-testing trips. Appends one row per (model, variant) to experiments/results.csv.

    python -m src.models.train --train data/train_candidates.csv \\
                               --student-test data/student_test_candidates.csv
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from .. import config as C
from ..candidate_data import (add_context_features, candidate_recall_report, feature_matrix,
                    infer_feature_columns, load_candidates, sha256_file, training_rows)
from ..evaluation.metrics import METRIC_COLS, evaluate_scores, normalise_within_obs
from ..predict import save_bundle
from . import MODELS


# ---------------------------------------------------------------- helpers
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(C.ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def make_cv_splits(df: pd.DataFrame, n_splits: int, seed: int):
    """GroupKFold with trip_id as groups (shuffled with a fixed seed for reproducibility)."""
    groups = df[C.TRIP_COL].to_numpy()
    k = min(n_splits, pd.unique(groups).size)
    if k < 2:
        raise ValueError("Need at least 2 trips for GroupKFold.")
    return list(GroupKFold(n_splits=k, shuffle=True, random_state=seed).split(df, groups=groups))


def predict_pos(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def fit_checked(model, X, y, max_rows=None):
    """Fit a model, optionally using only the first max_rows rows (DEV speed setting)."""
    if len(np.unique(y)) < 2:
        raise ValueError("Training fold contains a single class.")
    if max_rows is not None and len(y) > max_rows:
        X = X.iloc[:max_rows] if hasattr(X, "iloc") else X[:max_rows]
        y = y.iloc[:max_rows] if hasattr(y, "iloc") else y[:max_rows]
    t0 = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t0


# ---------------------------------------------------------------- cross-validation
def cross_validate_model(name, params, df, features, *, n_splits=C.N_SPLITS, seed=C.SEED,
                         drop_unrecallable_train=True, fold_callback=None):
    """Group-aware CV. Returns fold metrics, summary (mean/std) and out-of-fold scores."""
    mod = MODELS[name]
    splits = make_cv_splits(df, n_splits, seed)
    oof = np.full(len(df), np.nan)
    fold_rows, fit_t, pred_t = [], [], []
    for i, (tr, va) in enumerate(splits):
        tr_df, va_df = df.iloc[tr], df.iloc[va]
        assert set(tr_df[C.TRIP_COL]).isdisjoint(set(va_df[C.TRIP_COL])), "trip leakage across folds!"
        tr_fit = training_rows(tr_df, drop_unrecallable_train)
        model = mod.build(params, seed)
        # DEV speed setting: limit KNN's fit data; other models use all fold rows.
        fit_limit = 20000 if name == "knn" else None
        fit_t.append(fit_checked(model, feature_matrix(tr_fit, features),
                                 tr_fit[C.LABEL_COL], max_rows=fit_limit))
        t0 = time.perf_counter()
        p = predict_pos(model, feature_matrix(va_df, features))
        pred_t.append(time.perf_counter() - t0)
        oof[va] = p
        m = evaluate_scores(va_df, p)
        m.update(fold=i, n_val_trips=int(va_df[C.TRIP_COL].nunique()), n_val_rows=len(va_df))
        fold_rows.append(m)
        if fold_callback:
            fold_callback(i, m)
    folds = pd.DataFrame(fold_rows)
    summary = {}
    for c in METRIC_COLS:
        summary[f"{c}_mean"] = float(folds[c].mean())
        summary[f"{c}_std"] = float(folds[c].std(ddof=0))
    return {"folds": folds, "summary": summary, "oof": oof, "n_splits": len(splits),
            "fit_time_mean": float(np.mean(fit_t)), "pred_time_mean": float(np.mean(pred_t))}


# ---------------------------------------------------------------- one model, end to end
def run_model(name, params, variant, train_df, test_df, features, *, n_splits, seed,
              out_root, drop_unrecallable_train=True, data_meta=None, extra=None):
    P = C.paths(out_root)
    for k in ("models", "configs", "oof", "student_preds"):
        P[k].mkdir(parents=True, exist_ok=True)
    P["results"].parent.mkdir(parents=True, exist_ok=True)
    tag = f"{name}__{variant}"
    mod = MODELS[name]
    print(f"\n=== {tag} ===", flush=True)

    cv = cross_validate_model(name, params, train_df, features, n_splits=n_splits, seed=seed,
                              drop_unrecallable_train=drop_unrecallable_train)
    s = cv["summary"]
    print(f"  CV macroF1 {s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f} | "
          f"obs Top-1 {s['obs_top1_all_mean']:.4f} | logloss {s['log_loss_mean']:.4f}", flush=True)

    oof_df = train_df[C.ID_COLS + [C.LABEL_COL]].copy()
    oof_df["score"] = cv["oof"]
    oof_df["prob_norm"] = normalise_within_obs(train_df, cv["oof"])
    oof_df.to_csv(P["oof"] / f"{tag}.csv", index=False)

    # full fit on all training trips (timed); KNN uses a 20,000-row DEV subset.
    fit_df = training_rows(train_df, drop_unrecallable_train)
    model = mod.build(params, seed)
    fit_limit = 20000 if name == "knn" else None
    fit_time = fit_checked(model, feature_matrix(fit_df, features),
                           fit_df[C.LABEL_COL], max_rows=fit_limit)

    # student-test evaluation + inference timing (median of 3 runs)
    Xte = feature_matrix(test_df, features)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        p_te = predict_pos(model, Xte)
        times.append(time.perf_counter() - t0)
    infer_time = float(np.median(times))
    te = evaluate_scores(test_df, p_te)
    print(f"  STUDENT macroF1 {te['macro_f1']:.4f} | obs Top-1 {te['obs_top1_all']:.4f} "
          f"| fit {fit_time:.2f}s | infer {infer_time:.3f}s", flush=True)

    pred = test_df[C.ID_COLS + [C.LABEL_COL]].copy()
    pred["score"] = p_te
    pred["prob_norm"] = normalise_within_obs(test_df, p_te)
    if "dist_rank" in test_df:
        pred["dist_rank"] = test_df["dist_rank"].to_numpy()
    pred.to_csv(P["student_preds"] / f"{tag}.csv", index=False)

    meta = {
        "model_name": name, "variant": variant, "seed": seed, "n_splits": cv["n_splits"],
        "params": {**mod.DEFAULT_PARAMS, **(params or {})},
        "estimator_params": json.loads(json.dumps(model.get_params(deep=False), default=str)),
        "feature_columns": list(features), "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(), "git_commit": git_commit(),
        "drop_unrecallable_train": drop_unrecallable_train,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(data_meta or {}),
    }
    save_bundle(P["models"] / f"{tag}.joblib", model, features, meta)
    (P["configs"] / f"{tag}.json").write_text(json.dumps(meta, indent=2, default=str))

    row = {
        "run_id": uuid.uuid4().hex[:8], "timestamp_utc": meta["created_utc"],
        "model": name, "variant": variant, "seed": seed, "n_splits": cv["n_splits"],
        "params_json": json.dumps(meta["params"], default=str, sort_keys=True),
        "n_features": len(features), "n_train_rows": len(train_df),
        "n_train_trips": int(train_df[C.TRIP_COL].nunique()),
        "n_student_rows": len(test_df), "n_student_trips": int(test_df[C.TRIP_COL].nunique()),
        "fit_time_full_s": fit_time, "cv_fit_time_mean_s": cv["fit_time_mean"],
        "infer_time_student_s": infer_time,
        "infer_us_per_candidate": 1e6 * infer_time / len(test_df),
        "sklearn_version": sklearn.__version__, "git_commit": meta["git_commit"],
        **{k: v for k, v in (data_meta or {}).items() if k.endswith("sha256")},
        "max_train_trips": (data_meta or {}).get("max_train_trips"),
        **(extra or {}),
    }
    for k, v in s.items():
        row[f"cv_{k}"] = v
    for c in METRIC_COLS:
        row[f"student_{c}"] = te[c]
    row["student_candidate_recall"] = te["candidate_recall"]
    append_result(P["results"], row)
    return row


def append_result(path, row: dict) -> None:
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


# ---------------------------------------------------------------- data prep shared with optimisation
def load_and_prepare(train_path, test_path, features_arg=None, max_train_trips=None, seed=C.SEED):
    train = add_context_features(load_candidates(train_path))
    test = add_context_features(load_candidates(test_path))
    shared = set(train[C.TRIP_COL].astype(str)) & set(test[C.TRIP_COL].astype(str))
    if shared:
        raise ValueError(f"Trips appear in BOTH train and student-test: {sorted(shared)[:5]} ...")
    all_trips = sorted(train[C.TRIP_COL].unique(), key=str)
    if max_train_trips and max_train_trips < len(all_trips):
        keep = np.random.default_rng(seed).choice(np.array(all_trips, dtype=object),
                                                  size=max_train_trips, replace=False)
        train = train[train[C.TRIP_COL].isin(set(keep))].reset_index(drop=True)
        print(f"[subsample] DEV RUN: using {max_train_trips} of {len(all_trips)} training trips (seed {seed})")
    features = infer_feature_columns(train, features_arg.split(",") if features_arg else None)
    missing = [f for f in features if f not in test.columns]
    if missing:
        raise ValueError(f"Student-test data lacks features {missing}")
    meta = {"train_path": str(train_path), "student_test_path": str(test_path),
            "train_sha256": sha256_file(train_path), "student_test_sha256": sha256_file(test_path),
            "max_train_trips": max_train_trips, "n_train_trips_available": len(all_trips)}
    return train, test, features, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="candidate-level training file (csv/parquet)")
    ap.add_argument("--student-test", required=True, help="candidate-level file for the 10 student-testing trips")
    ap.add_argument("--models", default="all", help="comma list or 'all': " + ",".join(MODELS))
    ap.add_argument("--n-splits", type=int, default=C.N_SPLITS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--features", default=None, help="comma-separated feature list (default: infer)")
    ap.add_argument("--keep-unrecallable-train", action="store_true",
                    help="keep observations with no correct candidate in training")
    ap.add_argument("--max-train-trips", type=int, default=None,
                    help="DEV ONLY: random subset of training trips (recorded in results.csv)")
    ap.add_argument("--out-root", default=None, help="where artifacts/ experiments/ outputs/ go (default: repo root)")
    a = ap.parse_args(argv)

    train, test, features, data_meta = load_and_prepare(a.train, a.student_test, a.features,
                                                        a.max_train_trips, a.seed)
    P = C.paths(a.out_root)
    P["recall"].mkdir(parents=True, exist_ok=True)
    for split, d in (("train", train), ("student_test", test)):
        summ, per_trip = candidate_recall_report(d)
        (P["recall"] / f"{split}_summary.json").write_text(json.dumps(summ, indent=2))
        per_trip.to_csv(P["recall"] / f"{split}_per_trip.csv", index=False)
        print(f"[recall] {split}: candidate recall {summ['candidate_recall']:.4f} "
              f"({summ['obs_with_true_candidate']}/{summ['n_observations']} obs), "
              f"{summ['mean_candidates_per_obs']:.1f} candidates/obs")
    if test[C.TRIP_COL].nunique() != 10:
        print(f"[warn] student-test has {test[C.TRIP_COL].nunique()} trips (expected 10)")

    names = list(MODELS) if a.models == "all" else a.models.split(",")
    for n in names:
        if n not in MODELS:
            sys.exit(f"unknown model {n}; choose from {list(MODELS)}")
    for n in names:
        run_model(n, None, "untuned", train, test, features, n_splits=a.n_splits, seed=a.seed,
                  out_root=a.out_root, drop_unrecallable_train=not a.keep_unrecallable_train,
                  data_meta=data_meta)
    print(f"\nDone. Results: {P['results']}")


if __name__ == "__main__":
    main()
