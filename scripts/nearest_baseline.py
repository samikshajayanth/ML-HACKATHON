"""Trivial baseline: always pick the nearest candidate. No training, so it is evaluated on the
full training trips and on the student-test trips. Your models must clearly beat this.

    python scripts/nearest_baseline.py --train data/train_candidates.csv --student-test data/student_test_candidates.csv
Writes experiments/baseline_nearest_distance.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.candidate_data import add_context_features, load_candidates  # noqa: E402
from src.evaluation.metrics import observation_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--student-test", required=True)
ap.add_argument("--out-root", default=None)
a = ap.parse_args()

rows = []
for name, path in (("train_all_trips", a.train), ("student_test", a.student_test)):
    d = add_context_features(load_candidates(path))
    m = observation_metrics(d, 1.0 / (1.0 + d[C.DIST_COL].to_numpy()))   # closer = higher score
    rows.append({"baseline": "nearest_distance", "evaluated_on": name, "n_trips": int(d[C.TRIP_COL].nunique()), **m})
    print(f"{name}: obs Top-1 {m['obs_top1_all']:.4f} | Top-1 on recallable obs {m['obs_top1_recallable']:.4f} "
          f"| Top-3 {m['obs_top3_all']:.4f} | candidate recall {m['candidate_recall']:.4f}")
out = C.paths(a.out_root)["results"].parent / "baseline_nearest_distance.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
