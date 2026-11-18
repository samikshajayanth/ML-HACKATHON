"""Comparison tables, plots and final-model selection - all read from real result files.

    python -m src.evaluation.comparison                # tables + plots
    python -m src.evaluation.comparison --select       # also pick + export the final model

Selection rule (documented, applied to CV results - NOT to the student-test trips, which
are only 10 trips and are used as a confirmation):
  1. keep the (model, variant) rows whose CV obs_top1_all is within --tolerance of the best,
  2. among those take the highest CV macro_f1,
  3. then the lowest inference time per candidate.
"""
from __future__ import annotations

import argparse
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as C
from .metrics import per_trip_top1

COL = {"untuned": "#4C78A8", "tuned": "#F58518"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def latest_rows(results: pd.DataFrame) -> pd.DataFrame:
    return (results.sort_values("timestamp_utc").groupby(["model", "variant"], as_index=False).tail(1)
            .sort_values(["model", "variant"]).reset_index(drop=True))


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary_table(L: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({
        "model": L["model"], "variant": L["variant"],
        "cv_macro_f1": L["cv_macro_f1_mean"].round(4).astype(str) + " ± " + L["cv_macro_f1_std"].round(4).astype(str),
        "cv_accuracy": L["cv_accuracy_mean"], "cv_precision_macro": L["cv_precision_macro_mean"],
        "cv_recall_macro": L["cv_recall_macro_mean"], "cv_log_loss": L["cv_log_loss_mean"],
        "cv_obs_top1": L["cv_obs_top1_all_mean"],
        "student_macro_f1": L["student_macro_f1"], "student_log_loss": L["student_log_loss"],
        "student_obs_top1": L["student_obs_top1_all"],
        "student_obs_top1_recallable": L["student_obs_top1_recallable"],
        "fit_time_s": L["fit_time_full_s"], "infer_us_per_cand": L["infer_us_per_candidate"],
    })
    return t


def tuned_vs_untuned(L: pd.DataFrame) -> pd.DataFrame:
    u = L[L.variant == "untuned"].set_index("model")
    t = L[L.variant == "tuned"].set_index("model")
    common = u.index.intersection(t.index)
    rows = []
    for m in common:
        rows.append({
            "model": m,
            "cv_macro_f1_untuned": u.at[m, "cv_macro_f1_mean"], "cv_macro_f1_tuned": t.at[m, "cv_macro_f1_mean"],
            "cv_macro_f1_delta": t.at[m, "cv_macro_f1_mean"] - u.at[m, "cv_macro_f1_mean"],
            "cv_obs_top1_untuned": u.at[m, "cv_obs_top1_all_mean"], "cv_obs_top1_tuned": t.at[m, "cv_obs_top1_all_mean"],
            "cv_obs_top1_delta": t.at[m, "cv_obs_top1_all_mean"] - u.at[m, "cv_obs_top1_all_mean"],
            "student_obs_top1_untuned": u.at[m, "student_obs_top1_all"],
            "student_obs_top1_tuned": t.at[m, "student_obs_top1_all"],
            "student_obs_top1_delta": t.at[m, "student_obs_top1_all"] - u.at[m, "student_obs_top1_all"],
            "cv_log_loss_untuned": u.at[m, "cv_log_loss_mean"], "cv_log_loss_tuned": t.at[m, "cv_log_loss_mean"],
            "fit_time_untuned_s": u.at[m, "fit_time_full_s"], "fit_time_tuned_s": t.at[m, "fit_time_full_s"],
            "optuna_trials": t.at[m, "optuna_n_trials"] if "optuna_n_trials" in t else np.nan,
            "optuna_time_s": t.at[m, "optuna_time_s"] if "optuna_time_s" in t else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def grouped_bars(ax, L, value_col, err_col=None, ylabel="", title=""):
    models = list(dict.fromkeys(L["model"]))
    variants = [v for v in ("untuned", "tuned") if v in set(L["variant"])]
    w = 0.8 / max(len(variants), 1)
    x = np.arange(len(models))
    for j, v in enumerate(variants):
        sub = L[L.variant == v].set_index("model").reindex(models)
        err = sub[err_col].to_numpy() if err_col else None
        xs = x + (j - (len(variants) - 1) / 2) * w
        vals = sub[value_col].to_numpy()
        ax.bar(xs, vals, w, yerr=err, capsize=3, label=v, color=COL[v])
        for xi, vi, ei in zip(xs, vals, err if err is not None else np.zeros(len(vals))):
            if np.isfinite(vi):
                ax.text(xi, vi + (ei if np.isfinite(ei) else 0), f"{vi:.3g}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def make_plots(L: pd.DataFrame, P: dict, student_dir) -> None:
    pl = P["plots"]
    pl.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bars(ax, L, "cv_macro_f1_mean", "cv_macro_f1_std", "Macro F1", "GroupKFold CV Macro F1 (mean ± std over folds)")
    save(fig, pl / "cv_macro_f1.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    grouped_bars(axs[0], L, "cv_obs_top1_all_mean", "cv_obs_top1_all_std", "Observation Top-1", "CV observation Top-1")
    grouped_bars(axs[1], L, "student_obs_top1_all", None, "", "Student-test (10 trips) observation Top-1")
    save(fig, pl / "obs_top1_cv_vs_student.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_bars(ax, L, "cv_log_loss_mean", "cv_log_loss_std", "Log loss (lower is better)", "CV log loss")
    save(fig, pl / "cv_log_loss.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    grouped_bars(axs[0], L, "fit_time_full_s", None, "seconds", "Training time (full train set)")
    grouped_bars(axs[1], L, "infer_us_per_candidate", None, "microseconds / candidate", "Inference time")
    for a in axs:
        a.set_yscale("log")
    save(fig, pl / "timing.png")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for v, mk in (("untuned", "o"), ("tuned", "s")):
        s = L[L.variant == v]
        ax.scatter(s["infer_us_per_candidate"], s["cv_obs_top1_all_mean"], s=60, marker=mk, color=COL[v], label=v)
        for _, r in s.iterrows():
            ax.annotate(r["model"].split("_")[0], (r["infer_us_per_candidate"], r["cv_obs_top1_all_mean"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("inference µs / candidate")
    ax.set_ylabel("CV observation Top-1")
    ax.set_title("Accuracy vs inference cost", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    save(fig, pl / "accuracy_vs_speed.png")

    tv = tuned_vs_untuned(L)
    if len(tv):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for a, col, ttl in ((axs[0], "cv_macro_f1_delta", "Δ CV Macro F1 (tuned − untuned)"),
                            (axs[1], "cv_obs_top1_delta", "Δ CV observation Top-1 (tuned − untuned)")):
            a.bar(tv["model"].str.replace("_", "\n"), tv[col], color=np.where(tv[col] >= 0, COL["tuned"], "#999999"))
            a.axhline(0, color="k", lw=0.8)
            a.set_title(ttl, fontsize=10)
            a.tick_params(axis="x", labelsize=8)
        save(fig, pl / "tuned_vs_untuned_delta.png")

    # per-trip heatmap on student-test trips
    rows = {}
    for _, r in L.iterrows():
        f = student_dir / f"{r['model']}__{r['variant']}.csv"
        if f.exists():
            d = pd.read_csv(f)
            t = per_trip_top1(d, d["score"].to_numpy())
            rows[f"{r['model']} ({r['variant']})"] = t.set_index(C.TRIP_COL)["obs_top1_all"]
    if rows:
        H = pd.DataFrame(rows).T
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * H.shape[1] + 3), 0.45 * len(H) + 1.8))
        im = ax.imshow(H.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(H.shape[1]))
        ax.set_xticklabels([str(c) for c in H.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(H)))
        ax.set_yticklabels(H.index, fontsize=7)
        ax.grid(False)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                ax.text(j, i, f"{H.iat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if H.iat[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, label="obs Top-1")
        ax.set_title("Student-test observation Top-1 per trip", fontsize=10)
        save(fig, pl / "student_test_per_trip_top1.png")
        H.to_csv(P["tables"] / "student_test_per_trip_top1.csv")


# ---------------------------------------------------------------- selection
def select_final(L: pd.DataFrame, tolerance: float):
    best = L["cv_obs_top1_all_mean"].max()
    pool = L[L["cv_obs_top1_all_mean"] >= best - tolerance]
    pool = pool.sort_values(["cv_macro_f1_mean", "infer_us_per_candidate"], ascending=[False, True])
    return pool.iloc[0], pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--select", action="store_true", help="select and export the final model")
    ap.add_argument("--tolerance", type=float, default=0.005, help="Top-1 tolerance for rule step 1")
    a = ap.parse_args(argv)
    P = C.paths(a.out_root)
    if not P["results"].exists():
        raise SystemExit(f"{P['results']} not found - run src.models.train first.")
    L = latest_rows(pd.read_csv(P["results"]))
    P["tables"].mkdir(parents=True, exist_ok=True)

    st = summary_table(L)
    st.to_csv(P["tables"] / "model_comparison.csv", index=False)
    (P["tables"] / "model_comparison.md").write_text(md_table(st))
    tv = tuned_vs_untuned(L)
    if len(tv):
        P["optim"].mkdir(parents=True, exist_ok=True)
        tv.to_csv(P["optim"] / "tuned_vs_untuned.csv", index=False)
        (P["tables"] / "tuned_vs_untuned.md").write_text(md_table(tv))
    print(md_table(st))
    make_plots(L, P, P["student_preds"])

    if a.select:
        row, pool = select_final(L, a.tolerance)
        src = P["models"] / f"{row['model']}__{row['variant']}.joblib"
        shutil.copy(src, P["models"] / "final_model.joblib")
        cfg = {
            "final_model": row["model"], "variant": row["variant"], "run_id": row["run_id"],
            "rule": (f"CV obs_top1_all within {a.tolerance} of best, then highest CV macro_f1, then lowest "
                     "inference time per candidate. Student-test used only as confirmation."),
            "cv_obs_top1_all": row["cv_obs_top1_all_mean"], "cv_macro_f1": row["cv_macro_f1_mean"],
            "cv_log_loss": row["cv_log_loss_mean"], "student_obs_top1_all": row["student_obs_top1_all"],
            "student_macro_f1": row["student_macro_f1"], "infer_us_per_candidate": row["infer_us_per_candidate"],
            "params": json.loads(row["params_json"]), "seed": int(row["seed"]),
            "models_within_tolerance": [f"{m}/{v}" for m, v in zip(pool["model"], pool["variant"])],
            "artifact": "artifacts/models/final_model.joblib",
        }
        (P["configs"] / "final_model.json").write_text(json.dumps(cfg, indent=2))
        print(f"\nFINAL MODEL: {row['model']} ({row['variant']}) -> {P['models'] / 'final_model.joblib'}")
        print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
