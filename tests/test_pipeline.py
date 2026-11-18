"""Sanity tests. Run:  python -m pytest tests -q   (or: python tests/test_pipeline.py)"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_synthetic_smoke_data import make  # noqa: E402
from src import config as C  # noqa: E402
from src.candidate_data import add_context_features, infer_feature_columns, validate_schema  # noqa: E402
from src.evaluation.metrics import observation_metrics, rank_within_obs  # noqa: E402
from src.models.train import cross_validate_model, make_cv_splits  # noqa: E402


def _df():
    return add_context_features(make(12, np.random.default_rng(1)))


def test_top1_hand_computed():
    # trip A obs 0: truth ranked 1st; obs 1: truth ranked 2nd. trip B obs 0 (same obs_id!) truth 1st.
    df = pd.DataFrame({
        "trip_id": ["A"] * 4 + ["B"] * 2, "obs_id": [0, 0, 1, 1, 0, 0], "candidate_id": [0, 1, 0, 1, 0, 1],
        "is_correct": [1, 0, 0, 1, 0, 1]})
    s = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.7])
    m = observation_metrics(df, s, tiebreak_col=None)
    assert abs(m["obs_top1_all"] - 2 / 3) < 1e-9 and m["n_obs"] == 3
    assert list(rank_within_obs(df, s)) == [1, 2, 1, 2, 2, 1]


def test_cv_has_no_trip_leakage_and_same_obs_id_is_not_merged():
    df = _df()
    for tr, va in make_cv_splits(df, 5, 42):
        assert set(df.iloc[tr].trip_id).isdisjoint(set(df.iloc[va].trip_id))
    assert df.obs_id.nunique() < len(df.groupby(C.OBS_KEYS))  # obs_id repeats across trips


def test_features_exclude_ids_label_and_leaky_names():
    df = _df()
    f = infer_feature_columns(df, verbose=False)
    for bad in ("trip_id", "obs_id", "candidate_id", "is_correct", "obs_has_truth", "arc_id_local"):
        assert bad not in f, bad
    assert "dist_m" in f and "dist_rank" in f


def test_schema_rejects_duplicates():
    df = _df()
    try:
        validate_schema(pd.concat([df, df.iloc[:1]]), need_label=True)
    except ValueError:
        return
    raise AssertionError("duplicate rows not rejected")


def test_cv_runs_all_models_and_beats_chance_on_synthetic():
    from src.models import MODELS
    df = _df()
    feats = infer_feature_columns(df, verbose=False)
    for name in MODELS:
        r = cross_validate_model(name, None, df, feats, n_splits=3, seed=0)
        assert r["summary"]["obs_top1_all_mean"] > 0.5, name
        assert not np.isnan(r["oof"]).any()


def test_route_labels_monotone_and_loops():
    from src.route_labels import assign_route_labels
    # route revisits arc 1: [1, 2, 1, 3]. Observation k is truly on route position k, decoy arcs are NOT on the route.
    truth = [1, 2, 1, 3]
    rows = []
    for o, t in enumerate(truth):
        for arc in (1, 2, 3, 100, 101):
            d = 1.0 if arc == t else (0.5 if arc == 100 else 30.0)   # off-route decoy 100 is even closer
            rows.append({"trip_id": "T", "obs_id": o, "candidate_id": arc, "dist_m": d})
    out = assign_route_labels(pd.DataFrame(rows), [1, 2, 1, 3])
    got = out[out.is_correct == 1].sort_values("obs_id").candidate_id.tolist()
    assert got == truth, got
    assert out.groupby("obs_id").is_correct.sum().eq(1).all()
    # an observation whose candidates are all off-route gets no label (a candidate-generation miss)
    miss = pd.DataFrame([{"trip_id": "T", "obs_id": 0, "candidate_id": 100, "dist_m": 2.0}])
    assert assign_route_labels(miss, [1, 2]).is_correct.sum() == 0


def test_member1_schema_is_adapted():
    # columns exactly as Member 1's feature_extraction.py writes them
    raw = pd.DataFrame({
        "trip_id": ["00000001"] * 3, "observation_id": [0, 0, 1], "timestamp": [1.0, 1.0, 2.0],
        "candidate_rank": [0, 1, 0], "candidate_arc_id": [64973, 117639, 64973],
        "from_node_id": [1, 2, 1], "to_node_id": [3, 4, 3], "distance_m": [0.7, 15.1, 0.3],
        "fraction_along_arc": [0.1, 0.5, 0.2], "is_fallback": [False, False, False],
        "gps_heading_deg": [9.0, 9.0, 7.0], "speed_mps": [43.6, 43.6, 46.4],
        "arc_heading_deg": [7.6, 187.7, 7.6], "heading_diff_deg": [1.3, 178.8, 0.7],
        "arc_length_m": [293.0, 1020.0, 293.0], "timestamp_gap_s": [np.nan, np.nan, 1.0],
        "projected_epsg": [32632] * 3, "is_correct": [1, 0, 1]})
    from src.candidate_data import apply_aliases
    d = add_context_features(apply_aliases(raw))
    f = infer_feature_columns(d, verbose=False)
    for bad in ("from_node_id", "to_node_id", "projected_epsg", "gps_heading_deg", "arc_heading_deg",
                "timestamp", "candidate_id", "obs_id", "is_correct", "obs_has_truth"):
        assert bad not in f, bad
    for good in ("dist_m", "proj_frac", "seg_length_m", "heading_diff_deg", "speed_mps", "is_fallback",
                 "timestamp_gap_s", "dist_rank"):
        assert good in f, good


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
