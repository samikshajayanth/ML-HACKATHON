"""Build data/train_candidates.csv and data/student_test_candidates.csv from the raw trips,
using Member 1's candidate generator + feature extractor and the route-alignment labels.

Run from the repo root of the MERGED repo (needs Member 1's src/data, src/features, and
`pip install pyproj shapely`):

    python scripts/build_candidate_dataset.py --dataset-root "/path/to/Student folder" --out data
    # quick check first:   ... --max-trips 3

NOTE: written against Member 1's code as pushed; not executed end-to-end by Member 2
(raw trip files were not available). Check the build_report_*.csv it writes.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_data import apply_aliases  # noqa: E402
from src.data.loader import load_trip  # noqa: E402  (Member 1)
from src.features.candidates import CandidateGenerator  # noqa: E402  (Member 1)
from src.features.feature_extraction import extract_features  # noqa: E402  (Member 1)
from src.route_labels import assign_route_labels  # noqa: E402


TRAIN_SPLIT_NAME = "train"


def build_split(root: Path, split: str, gen: CandidateGenerator, out_dir: Path, max_trips):
    trip_ids = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
    if max_trips:
        trip_ids = trip_ids[:max_trips]
    frames, report = [], []
    for i, tid in enumerate(trip_ids, 1):
        t0 = time.perf_counter()
        try:
            trip = load_trip(root, split, tid)
            feats = apply_aliases(extract_features(trip, gen.generate(trip)))
            lab = assign_route_labels(feats, trip.route)
            frames.append(lab)
            per_obs = lab.groupby("obs_id")["is_correct"].max()
            report.append({"trip_id": tid, "status": "OK", "n_obs": len(per_obs), "n_rows": len(lab),
                           "obs_with_label": float(per_obs.mean()),
                           "obs_at_candidate_cap": float((lab.groupby("obs_id").size() >= gen.max_candidates).mean()),
                           "seconds": round(time.perf_counter() - t0, 2)})
        except Exception as exc:  # keep going, record the failure
            report.append({"trip_id": tid, "status": f"ERROR: {exc}"})
            traceback.print_exc()
        print(f"[{split}] {i}/{len(trip_ids)} {tid} {report[-1]['status']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "train_candidates.csv" if split == TRAIN_SPLIT_NAME else "student_test_candidates.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_dir / name, index=False)
    pd.DataFrame(report).to_csv(out_dir / f"build_report_{split}.csv", index=False)
    ok = pd.DataFrame(report).query("status == 'OK'")
    print(f"[{split}] wrote {out_dir / name}: {len(ok)}/{len(trip_ids)} trips; "
          f"mean share of observations with a labelled candidate = {ok['obs_with_label'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--radius-m", type=float, default=100)
    ap.add_argument("--max-candidates", type=int, default=8, help="Member 1 used 8; try 12-16 if recall is low")
    ap.add_argument("--max-trips", type=int, default=None, help="quick check: first N trips per split")
    ap.add_argument("--train-split", default="train", help="name of the training folder")
    ap.add_argument("--test-split", default="auto",
                    help="name of the student-test folder; 'auto' = student_test, else test")
    a = ap.parse_args()
    if any(ch in str(a.dataset_root) for ch in "<>") or not a.dataset_root.is_dir():
        sys.exit(f"--dataset-root '{a.dataset_root}' is not a real folder. Use the actual path of the folder that "
                 "contains the train and test folders, in quotes if it has spaces. "
                 "Run: python scripts/check_setup.py --dataset-root \"<real path>\"")
    TRAIN_SPLIT_NAME = a.train_split
    test_split = a.test_split
    if test_split == "auto":
        test_split = next((n for n in ("student_test", "test") if (a.dataset_root / n).is_dir()), None)
        if test_split is None:
            sys.exit(f"No student_test/ or test/ folder in {a.dataset_root}. Folders found: "
                     f"{[p.name for p in a.dataset_root.iterdir() if p.is_dir()]}. Pass --test-split NAME.")
    for s in (a.train_split, test_split):
        if not (a.dataset_root / s).is_dir():
            sys.exit(f"{a.dataset_root / s} does not exist.")
    print(f"[build] train folder = {a.train_split!r}, student-test folder = {test_split!r}")
    gen = CandidateGenerator(radius_m=a.radius_m, max_candidates=a.max_candidates)
    for s in (a.train_split, test_split):
        build_split(a.dataset_root, s, gen, a.out, a.max_trips)
