"""SMOKE-TEST DATA ONLY. Generates fake candidate data so the pipeline can be tested before
Member 1 delivers. NEVER report any number produced from this data.

    python scripts/make_synthetic_smoke_data.py --out data/_synthetic_smoke
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_trips, rng, trip_offset=0, miss_rate=0.04):
    rows = []
    for t in range(n_trips):
        trip = f"T{trip_offset + t:03d}"
        for o in range(int(rng.integers(30, 70))):
            k = int(rng.integers(2, 9))
            truth = int(rng.integers(0, k)) if rng.random() > miss_rate else -1
            for c in range(k):
                is_t = c == truth
                rows.append({
                    "trip_id": trip, "obs_id": o, "candidate_id": c,
                    "arc_id_local": int(rng.integers(0, 500)),
                    "dist_m": abs(rng.normal(0, 9)) if is_t else rng.uniform(3, 70),
                    "heading_diff_deg": abs(rng.normal(0, 18)) if is_t else rng.uniform(0, 180),
                    "proj_frac": rng.uniform(0, 1), "seg_length_m": rng.uniform(20, 400),
                    "speed_mps": rng.uniform(0, 25), "road_class": int(rng.integers(0, 5)),
                    "is_correct": int(is_t),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/_synthetic_smoke")
    ap.add_argument("--train-trips", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    make(a.train_trips, rng).to_csv(out / "train_candidates.csv", index=False)
    make(10, rng, trip_offset=1000).to_csv(out / "student_test_candidates.csv", index=False)
    print("wrote synthetic SMOKE data to", out)
