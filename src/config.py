"""Central configuration: paths, seeds and column names.

Column names here ARE the data contract with Member 1 (see docs/DATA_CONTRACT.md).
Change them in one place only.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
N_SPLITS = 5

# --- identifiers / target -------------------------------------------------
TRIP_COL = "trip_id"          # grouping column for GroupKFold
OBS_COL = "obs_id"            # GPS observation id (unique only WITHIN a trip)
CAND_COL = "candidate_id"     # candidate id (unique only WITHIN a trip+observation)
LABEL_COL = "is_correct"      # 1 if this candidate is the reference road segment
OBS_KEYS = [TRIP_COL, OBS_COL]
ID_COLS = [TRIP_COL, OBS_COL, CAND_COL]
REQUIRED_TRAIN_COLS = ID_COLS + [LABEL_COL]

# --- columns used to build within-observation context features ------------
DIST_COL = "dist_m"
HEADING_COL = "heading_diff_deg"

# Member 1's raw column names -> contract names (applied on load and at prediction time).
ALIASES = {
    "observation_id": "obs_id",
    "candidate_arc_id": "candidate_id",
    "distance_m": "dist_m",
    "arc_length_m": "seg_length_m",
    "fraction_along_arc": "proj_frac",
}

# Derived column that is computed from the label. NEVER a feature.
TRUTH_FLAG_COL = "obs_has_truth"

# Never used as features when features are inferred automatically.
NON_FEATURE_COLS = {
    TRIP_COL, OBS_COL, CAND_COL, LABEL_COL, TRUTH_FLAG_COL,
    "timestamp", "time", "datetime", "lat", "lon", "latitude", "longitude",
    "gps_lat", "gps_lon", "obs_index", "obs_idx",
    # absolute compass bearings (location/direction specific; heading_diff_deg is the useful one)
    "gps_heading_deg", "arc_heading_deg",
}
# Column names that look like they carry the answer (or a trip-local arc id).
LEAKAGE_PATTERN = re.compile(
    r"(^|_)(label|truth|true|gt|target|correct|reference|ref|matched|arc|edge|osm|node|epsg)(_|$)"
    r"|segment_id|is_correct",
    re.I,
)

# --- output locations (relative to --out-root, default = repo root) --------
def paths(out_root: str | Path | None = None) -> dict[str, Path]:
    r = Path(out_root) if out_root else ROOT
    return {
        "root": r,
        "models": r / "artifacts" / "models",
        "configs": r / "artifacts" / "configurations",
        "results": r / "experiments" / "results.csv",
        "oof": r / "experiments" / "oof_predictions",
        "student_preds": r / "experiments" / "student_test_predictions",
        "optim": r / "experiments" / "optimisation",
        "recall": r / "experiments" / "candidate_recall",
        "plots": r / "outputs" / "plots",
        "tables": r / "outputs" / "tables",
    }


DEFAULT_MODEL_PATH = ROOT / "artifacts" / "models" / "final_model.joblib"
