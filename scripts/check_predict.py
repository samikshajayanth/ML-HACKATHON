"""Integration check of the prediction interface Members 3 and 4 will use.

    python scripts/check_predict.py --student-test data/student_test_candidates.csv
(default model: artifacts/models/final_model.joblib; override with --model)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.candidate_data import load_candidates  # noqa: E402
from src.predict import CandidateScorer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--student-test", required=True)
ap.add_argument("--model", default=None)
a = ap.parse_args()

scorer = CandidateScorer.load(a.model)
df = load_candidates(a.student_test)
out = scorer.predict(df.drop(columns=[C.LABEL_COL]))          # the model never sees the label

assert len(out) == len(df), "row count changed"
assert (out[C.ID_COLS].to_numpy() == df[C.ID_COLS].to_numpy()).all(), "row order/IDs changed"
assert out["score"].between(0, 1).all(), "scores outside [0,1]"
sums = out.groupby(C.OBS_KEYS)["prob_norm"].sum()
assert np.allclose(sums, 1.0), "prob_norm does not sum to 1 within observations"
assert (out.groupby(C.OBS_KEYS)["rank"].apply(lambda r: (r == 1).sum()) == 1).all(), "each observation needs exactly one rank 1"

n_obs = df.groupby(C.OBS_KEYS).ngroups
top1 = df[C.LABEL_COL].to_numpy()[out["rank"].to_numpy() == 1].sum() / n_obs
print("ALL CHECKS PASSED")
print(f"model: {scorer.meta.get('model_name')} ({scorer.meta.get('variant')}) | seed {scorer.meta.get('seed')}")
print(f"student-test observation Top-1 recomputed through the public interface: {top1:.4f}")
print("raw columns a FastAPI request must contain:", scorer.required_columns)
print(out.head(8).to_string(index=False))
