"""Bayesian hyperparameter optimisation with Optuna (TPE sampler + median pruner).

Objective: mean group-aware (GroupKFold on trip_id) CV Macro F1 (default), maximised.
Per model it writes, under experiments/optimisation/<model>/:
    search_space.json   the search space with a justification per parameter
    trials.csv          EVERY trial (params, value, state, duration, per-fold scores)
    best_config.json    best params, value, #trials, optimisation time, seed
    study.db            Optuna sqlite storage
then retrains the best config ("tuned" variant) and appends it to experiments/results.csv.

    python -m src.optimisation.bayesian_search --train data/train_candidates.csv \
        --student-test data/student_test_candidates.csv --n-trials 40
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import optuna

from .. import config as C
from ..models import MODELS
from ..models.train import cross_validate_model, load_and_prepare, run_model

# (param, kind, kwargs, justification) -------------------------------------------------
SPACES: dict[str, list] = {
    "logistic_regression": [
        ("C", "float", {"low": 1e-3, "high": 1e2, "log": True},
         "Inverse L2 strength; spans strong to weak regularisation on a log scale."),
        ("class_weight", "cat", {"choices": [None, "balanced"]},
         "Positives are a small fraction of rows; test whether re-weighting helps."),
    ],
    "knn": [
        ("n_neighbors", "int", {"low": 3, "high": 101, "log": True},
         "Small k = low bias/high variance; log scale because the effect is relative."),
        ("weights", "cat", {"choices": ["uniform", "distance"]},
         "Distance weighting lets very close neighbours dominate (also the only imbalance lever)."),
        ("p", "cat", {"choices": [1, 2]}, "Manhattan vs Euclidean metric on standardised features."),
    ],
    "decision_tree": [
        ("max_depth", "int", {"low": 3, "high": 40}, "Main capacity control; deep trees overfit trips."),
        ("min_samples_leaf", "int", {"low": 1, "high": 100, "log": True},
         "Smooths leaf probabilities (important for log loss)."),
        ("min_samples_split", "int", {"low": 2, "high": 100, "log": True}, "Minimum rows to split."),
        ("criterion", "cat", {"choices": ["gini", "entropy"]}, "Split quality measure."),
        ("ccp_alpha", "float", {"low": 1e-6, "high": 1e-2, "log": True}, "Cost-complexity pruning."),
        ("class_weight", "cat", {"choices": [None, "balanced"]}, "Imbalance handling."),
    ],
    "random_forest": [
        ("n_estimators", "int", {"low": 100, "high": 500, "step": 50},
         "More trees = lower variance, linear cost; capped for time."),
        ("max_depth", "int", {"low": 6, "high": 40}, "Depth cap per tree."),
        ("min_samples_leaf", "int", {"low": 1, "high": 30, "log": True}, "Leaf size, smooths probabilities."),
        ("max_features", "cat", {"choices": ["sqrt", "log2", 0.5, 0.8]},
         "Feature subsampling decorrelates trees; features here are few."),
        ("class_weight", "cat", {"choices": [None, "balanced", "balanced_subsample"]}, "Imbalance handling."),
    ],
    "hist_gradient_boosting": [
        ("learning_rate", "float", {"low": 0.01, "high": 0.3, "log": True}, "Step size, log scale."),
        ("max_iter", "int", {"low": 100, "high": 600, "step": 50},
         "Boosting rounds (early stopping disabled to keep trip grouping intact)."),
        ("max_leaf_nodes", "int", {"low": 8, "high": 63}, "Tree complexity."),
        ("max_depth", "int", {"low": 3, "high": 14}, "Depth cap."),
        ("min_samples_leaf", "int", {"low": 10, "high": 200, "log": True}, "Regularises leaves."),
        ("l2_regularization", "float", {"low": 1e-6, "high": 10.0, "log": True}, "L2 on leaf values."),
        ("class_weight", "cat", {"choices": [None, "balanced"]}, "Imbalance handling."),
    ],
}


def suggest_params(name: str, trial) -> dict:
    out = {}
    for p, kind, kw, _ in SPACES[name]:
        if kind == "float":
            out[p] = trial.suggest_float(p, **kw)
        elif kind == "int":
            out[p] = trial.suggest_int(p, **kw)
        else:
            out[p] = trial.suggest_categorical(p, kw["choices"])
    return out


def space_doc(name: str) -> dict:
    return {p: {"type": kind, **kw, "justification": why} for p, kind, kw, why in SPACES[name]}


def make_objective(name, df, features, *, n_splits, seed, metric, drop_unrecallable_train):
    def objective(trial):
        params = suggest_params(name, trial)
        scores: list[float] = []

        def on_fold(i, fold_metrics):
            scores.append(float(fold_metrics[metric]))
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        res = cross_validate_model(name, params, df, features, n_splits=n_splits, seed=seed,
                                   drop_unrecallable_train=drop_unrecallable_train,
                                   fold_callback=on_fold)
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("cv_fit_time_mean_s", res["fit_time_mean"])
        trial.set_user_attr("cv_obs_top1_all_mean", res["summary"]["obs_top1_all_mean"])
        return float(np.mean(scores))
    return objective


def tune_model(name, train, test, features, data_meta, *, n_trials, timeout, n_splits, seed,
               metric, out_root, tune_max_trips, resume, drop_unrecallable_train):
    P = C.paths(out_root)
    odir = P["optim"] / name
    odir.mkdir(parents=True, exist_ok=True)
    db = odir / "study.db"
    if db.exists() and not resume:
        db.unlink()

    tune_df = train
    if tune_max_trips:
        trips = np.array(sorted(train[C.TRIP_COL].unique(), key=str))
        if tune_max_trips < len(trips):
            keep = np.random.default_rng(seed).choice(trips, size=tune_max_trips, replace=False)
            tune_df = train[train[C.TRIP_COL].isin(keep)].reset_index(drop=True)
    print(f"\n##### Optuna: {name} | {n_trials} trials | objective=CV {metric} | "
          f"{tune_df[C.TRIP_COL].nunique()} trips / {len(tune_df)} rows")

    (odir / "search_space.json").write_text(json.dumps(space_doc(name), indent=2, default=str))
    study = optuna.create_study(
        direction="maximize", study_name=f"{name}_{metric}", storage=f"sqlite:///{db}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, max(2, n_trials // 4))),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    t0 = time.perf_counter()
    study.optimize(make_objective(name, tune_df, features, n_splits=n_splits, seed=seed, metric=metric,
                                  drop_unrecallable_train=drop_unrecallable_train),
                   n_trials=n_trials, timeout=timeout)
    opt_time = time.perf_counter() - t0

    trials = study.trials_dataframe()
    trials.to_csv(odir / "trials.csv", index=False)
    states = [t.state.name for t in study.trials]
    best = study.best_trial
    best_cfg = {
        "model": name, "objective": f"mean GroupKFold(trip_id) CV {metric}", "direction": "maximize",
        "sampler": "TPESampler", "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
        "seed": seed, "n_splits": n_splits, "n_trials_requested": n_trials, "timeout_s": timeout,
        "n_trials_run": len(states), "n_complete": states.count("COMPLETE"),
        "n_pruned": states.count("PRUNED"), "n_failed": states.count("FAIL"),
        "best_trial_number": best.number, "best_value": best.value, "best_params": best.params,
        "optimisation_time_s": opt_time, "tune_max_trips": tune_max_trips,
        "tuning_trips": int(tune_df[C.TRIP_COL].nunique()), "optuna_version": optuna.__version__,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (odir / "best_config.json").write_text(json.dumps(best_cfg, indent=2, default=str))
    print(f"  best CV {metric} = {best.value:.4f} (trial {best.number}) in {opt_time:.0f}s; "
          f"{best_cfg['n_complete']} complete / {best_cfg['n_pruned']} pruned")

    # retrain the best config on the full training set with the full set of metrics
    run_model(name, best.params, "tuned", train, test, features, n_splits=n_splits, seed=seed,
              out_root=out_root, drop_unrecallable_train=drop_unrecallable_train, data_meta=data_meta,
              extra={"optuna_n_trials": len(states), "optuna_best_value": best.value,
                     "optuna_time_s": opt_time, "optuna_objective": metric})
    return best_cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--student-test", required=True)
    ap.add_argument("--models", default="all")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None, help="seconds per model (optional)")
    ap.add_argument("--objective", default="macro_f1", choices=["macro_f1", "obs_top1_all"])
    ap.add_argument("--n-splits", type=int, default=C.N_SPLITS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--features", default=None)
    ap.add_argument("--tune-max-trips", type=int, default=None,
                    help="tune on a random subset of trips (speed); final retrain uses all trips")
    ap.add_argument("--max-train-trips", type=int, default=None,
                    help="DEV ONLY: random subset of training trips (use the same value as the baseline run)")
    ap.add_argument("--resume", action="store_true", help="continue an existing study.db")
    ap.add_argument("--keep-unrecallable-train", action="store_true")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train, test, features, data_meta = load_and_prepare(a.train, a.student_test, a.features,
                                                        a.max_train_trips, a.seed)
    names = list(MODELS) if a.models == "all" else a.models.split(",")
    for n in names:
        tune_model(n, train, test, features, data_meta, n_trials=a.n_trials, timeout=a.timeout,
                   n_splits=a.n_splits, seed=a.seed, metric=a.objective, out_root=a.out_root,
                   tune_max_trips=a.tune_max_trips, resume=a.resume,
                   drop_unrecallable_train=not a.keep_unrecallable_train)


if __name__ == "__main__":
    main()
