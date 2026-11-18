# PathSnap ML Hackathon: Member 2 (Machine Learning)  [file: README_MEMBER2.md]

Candidate classifier for GPS map matching. Each row is one candidate road segment for one GPS
observation; the model predicts `P(candidate is the correct segment)`. Scores feed Member 3's HMM.

* Target = `is_correct` (binary) per candidate. Not a global arc-id class (arc ids are trip-local).
* CV = `GroupKFold(trip_id)`, seeded. Trips never straddle train/validation.
* Five models: Logistic Regression, KNN, Decision Tree, Random Forest, HistGradientBoosting.
* Bayesian tuning with Optuna (TPE + median pruner), objective = group-CV Macro F1.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_pipeline.py                            # must print 5 "ok" lines
```

## Run order (see docs/DATA_CONTRACT.md)
```bash
# 0. build labelled candidate files from the raw trips (needs Member 1's src/data, src/features; pip install pyproj shapely)
python scripts/build_candidate_dataset.py --dataset-root "/path/to/Student folder" --out data    # add --max-trips 3 first

# 1. untuned: 5 models, GroupKFold CV + student-test + timings + recall report
python -m src.models.train --train data/train_candidates.csv --student-test data/student_test_candidates.csv

# 2. Bayesian optimisation (40 trials/model; add --tune-max-trips 150 or --timeout 900 if slow)
python -m src.optimisation.bayesian_search --train data/train_candidates.csv --student-test data/student_test_candidates.csv --n-trials 40

# 3. tables + plots + final model selection (copies artifacts/models/final_model.joblib)
python -m src.evaluation.comparison --select
```
Dev shortcut: `--max-train-trips 20` on train/bayesian_search uses a random subset of training trips (recorded in results.csv; never use it for final numbers).
Every command also takes `--seed`, `--n-splits`, `--models a,b` and `--out-root DIR`.

## Where everything is written
| Path | Content |
|---|---|
| `experiments/results.csv` | one row per (model, variant): CV mean/std + student metrics, times, params, seed, data hashes, git commit |
| `experiments/candidate_recall/` | candidate-generation recall (upper bound on Top-1), per trip |
| `experiments/oof_predictions/` | out-of-fold scores for every training candidate (Member 3 can tune the HMM on these) |
| `experiments/student_test_predictions/` | scores on the 10 student trips |
| `experiments/optimisation/<model>/` | `search_space.json`, `trials.csv` (every trial), `best_config.json`, `study.db` |
| `experiments/optimisation/tuned_vs_untuned.csv` | tuned vs untuned |
| `artifacts/models/*.joblib`, `artifacts/configurations/*.json` | model bundles + full configs; `final_model.*` = selected model |
| `outputs/plots/`, `outputs/tables/` | figures and markdown/csv tables for the report |

## Metrics
Candidate level (threshold 0.5): accuracy, precision/recall (macro and positive-class), Macro F1, log loss.
Observation level: **Top-1 accuracy** (primary; a candidate-generation miss counts as wrong), Top-1 over
recallable observations, Top-3. Score ties are broken by nearest distance, never by row order.

## Final-model rule (applied to CV, student-test = confirmation only)
Within 0.5 pp of the best CV observation Top-1 → highest CV Macro F1 → lowest inference time.

## Integration
```python
# Member 3 (HMM)
from src.predict import CandidateScorer
scored = CandidateScorer.load("artifacts/models/final_model.joblib").predict(candidates_df)
# columns: trip_id, obs_id, candidate_id, score, prob_norm, rank   (input order preserved)
```
```python
# Member 4 (FastAPI) - untested sketch, fastapi was not available when this was written
from fastapi import FastAPI
from src.predict import load_scorer
app = FastAPI(); scorer = load_scorer()        # loads once at startup
@app.post("/score")
def score(records: list[dict]):                # all candidates of an observation together
    return scorer.predict_records(records)
```
`scorer.required_columns` lists the raw columns a request must contain.

## Honest limitations (put these in the report)
* Student-test is only 10 trips: its numbers are noisy. Select on CV, confirm on student-test.
* `class_weight="balanced"` shifts probabilities upward; `prob_norm` (within-observation normalisation) removes the
  shift for HMM use. Log loss is computed on the raw scores.
* KNN has no class weights (uses distance weighting). HGB uses `early_stopping=False` to keep trip grouping intact.
* Training drops observations with no correct candidate (no ranking signal); evaluation keeps them.
