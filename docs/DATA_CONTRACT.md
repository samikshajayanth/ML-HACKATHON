# Data contract: Member 1 → Member 2 (candidate-level features)

Send this to Member 1 and agree on it **before** they finish the pipeline.

## Files (CSV, or Parquet if pyarrow is installed)

| File | Content |
|---|---|
| `data/train_candidates.csv` | all training trips (NOT the student-testing trips, NOT evaluator data) |
| `data/student_test_candidates.csv` | the 10 student-testing trips, same columns |

Evaluator-testing data must never be given to the training script.

## One row = one candidate road segment for one GPS observation

### Required columns (exact names)

| Column | Meaning |
|---|---|
| `trip_id` | trip identifier (string or int). Used as the GroupKFold group. |
| `obs_id` | GPS observation identifier. Only unique **within a trip**. |
| `candidate_id` | candidate identifier. Unique within (trip_id, obs_id). |
| `is_correct` | 1 if this candidate is the reference (correctly aligned) segment for that observation, else 0. |

`(trip_id, obs_id, candidate_id)` must be unique. Every candidate of an observation must be present.
Keep the original row order within an observation if possible (not required).

### Feature columns (all numeric; NaN allowed; no strings)

Minimum useful set (names are what the code looks for to build extra features):

| Column | Meaning |
|---|---|
| `dist_m` | perpendicular distance GPS point → candidate segment (metres) |
| `heading_diff_deg` | absolute angle between travel heading and segment bearing, 0–180 (direction-aware for one-way roads) |
| `proj_frac` | position of the projected point along the segment, 0–1 |
| `seg_length_m` | segment length |
| `speed_mps` | observation speed |
| `road_class` | integer-coded (or give one-hot columns `road_class_*`) |

Add anything else useful and numeric (one-way flag, GPS accuracy, bearing from neighbouring GPS points,
number of lanes, …). The code uses **all numeric columns** except IDs/label automatically; the list
used is printed and saved in `artifacts/configurations/*.json`.

I compute these myself from your columns (do not send them): `n_candidates`, `dist_rank`,
`dist_minus_min`, `dist_ratio_to_min`, `heading_rank`, `heading_minus_min`.

### Must NOT be features / leakage rules

* Never include anything derived from the reference path or the label (e.g. "segment is on the true route",
  distance to the true segment). Names matching `label|truth|true|gt|target|correct|reference|ref|matched|arc|edge|osm`
  are auto-excluded with a printed list.
* The trip-local arc id is **not** a feature (it has no meaning across trips). Keep it only as a
  differently named column (e.g. `arc_id_local`) so it can be mapped back to road segments.
* Observations with no correct candidate (candidate-generation miss) must still be included, with all
  `is_correct = 0`. I report candidate recall from them. Please also tell me the generation radius / K.
* Transition features (route distance vs straight distance between consecutive observations) belong to
  Member 3's HMM, not to this classifier.

## What Member 2 returns (for Members 3 and 4)

`CandidateScorer.predict(df)` returns, in input order:

`trip_id, obs_id, candidate_id, score, prob_norm, rank`

* `score` = P(candidate is the correct segment) in [0, 1]
* `prob_norm` = `score` normalised to sum to 1 within each (trip_id, obs_id): use for HMM emissions
  (`log(max(prob_norm, 1e-12))`).
* `rank` = 1 for the best candidate.

At inference, send **all** candidates of an observation together with the same raw feature columns.


## Update after reviewing Member 1's first push (trip 00000001)

Member 1's raw column names are accepted and renamed automatically (`src/config.py: ALIASES`):
`observation_id→obs_id`, `candidate_arc_id→candidate_id`, `distance_m→dist_m`, `arc_length_m→seg_length_m`,
`fraction_along_arc→proj_frac`. `trip_id` is read as text so `00000001` keeps its zeros.

Auto-excluded (trip-local or non-generalising): `from_node_id`, `to_node_id`, `projected_epsg`, `timestamp`,
`gps_heading_deg`, `arc_heading_deg` (absolute bearings; `heading_diff_deg` is used instead).

Still required from Member 1: the **`is_correct` target**. Draft alignment: `src/route_labels.py`, applied by
`scripts/build_candidate_dataset.py` for all train + student_test trips. Member 1 must confirm the definition.
