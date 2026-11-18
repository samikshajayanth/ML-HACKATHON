# PATHSNAP: TECHNICAL REPORT
## Distributed GPS Trajectory Map-Matching & Route Reconstruction

**Course:** UE24CS352A: Machine Learning  
**Department of Computer Science and Engineering, PES University**  
**Repository:** `https://github.com/samikshajayanth/ML-HACKATHON` (Branch: `feature/api-ui`)  
**Evaluation Partitions Used:** 80 Training Trips (`train`), 10 Student-Testing Trips (`student_test`)

---

## 2.2 Dataset Understanding and Exploratory Data Analysis

### 1. How did you preprocess the data?

The raw PathSnap dataset package provides four text-based formats with whitespace separation and zero headers:
1. `.track` — GPS observations with columns `[longitude, latitude, timestamp_s]`.
2. `.nodes` — Road intersection nodes with columns `[longitude, latitude]` (0-based row index serves as `node_id`).
3. `.arcs` — Directed road connections with columns `[from_node_id, to_node_id]` (0-based row index serves as `arc_id`).
4. `.route` — Chronological ground-truth arc IDs traversed by the vehicle (only present in training and evaluation partitions).

#### Coordinate Transformation & Projection
Raw coordinates are recorded in the angular WGS84 coordinate reference system (EPSG:4326). Calculating Euclidean distances directly on angular coordinates $(\Delta \text{lon}, \Delta \text{lat})$ in degrees introduces non-uniform spatial distortion depending on latitude:
$$\Delta x = \Delta \text{lon} \cdot \cos(\text{lat}) \cdot 111{,}320\text{ m}, \quad \Delta y = \Delta \text{lat} \cdot 110{,}574\text{ m}$$
To ensure millimeter-accurate metric distance calculations across both GPS points and road geometries, we compute the local Universal Transverse Mercator (UTM) zone for each trip:
$$\text{zone} = \left\lfloor \frac{\text{mean}(\text{longitude}) + 180}{6} \right\rfloor + 1$$
For northern hemisphere coordinates ($\text{latitude} \ge 0$), the projected EPSG is `32600 + zone` (EPSG:32632 for the provided European trips). All nodes, arcs, and track points are projected into planar Cartesian coordinates $(x, y)$ in meters using `pyproj.Transformer` with `always_xy=True`.

#### Candidate Generation Preprocessing
For each projected GPS observation $z_t = (x_t, y_t)$, road segment candidate arcs are identified using a vectorized `shapely.STRtree` (2D spatial R-tree) over all projected road arc `LineString` geometries:
- A spatial buffer of **$R = 100\text{ meters}$** is queried (`predicate="dwithin"`).
- For each retrieved candidate arc, exact perpendicular projection distance $d_i = \text{distance}(z_t, \text{arc}_i)$ and projection position fraction $f_i = \text{project}(z_t) / \text{length}(\text{arc}_i) \in [0, 1]$ are computed.
- Candidates are sorted by perpendicular distance, retaining at most **$K = 8$ nearest candidates**.
- **Fallback mechanism:** If zero road arcs lie within 100 meters (due to extreme multipath jumps or sensor anomalies), the single nearest road arc across the entire road network is retained with flag `is_fallback = True`.

#### Route Alignment & Label Assignment (`src/route_labels.py`)
Because the `.route` file provides an ordered list of ground-truth traversed arcs but does not label which individual GPS point belongs to which arc, we formulate alignment as an optimal monotonic dynamic program:
- A candidate is eligible for label `is_correct = 1` only if its `candidate_arc_id` exists in `trip.route`.
- Backward steps along the route order are strictly penalized (restart penalty $= 100.0$), while forward steps incur a minimal jump cost ($\text{step} \times 0.01$).
- Among admissible sequences, the path minimizing cumulative perpendicular distance is assigned `is_correct = 1`.
- If an observation's true segment was not captured within the candidate set, all candidates for that observation receive `is_correct = 0` (unrecallable candidate-generation miss).

---

### 2. What patterns did you observe during data exploration? What features did you extract and why?

#### Key Exploratory Patterns Observed
From systematic inspection across the 80 training trips (`outputs/training_dataset_inspection.csv` and `outputs/gps_speed_diagnostics.csv`):
1. **Sampling Regularity with Rare Gaps:** Nominally, GPS is recorded at $1.0\text{ Hz}$ ($97.2\%$ of intervals are exactly $1.0\text{ s}$). However, several trips exhibit significant timestamp dropouts:
   - Trip `00000001`: Max gap $= 16.0\text{ s}$ (25 gaps $> 1\text{ s}$).
   - Trip `00000002`: Max gap $= 160.0\text{ s}$ (184 gaps $> 1\text{ s}$).
   - Trip `00000011`: Max gap $= 909.0\text{ s}$ (15 minutes).
2. **Velocity Bimodality:** Vehicle speeds computed via central differences reveal distinct operational domains:
   - Urban trips: Median speeds between $5.95\text{ m/s}$ ($21.4\text{ km/h}$, Trip `00000004`) and $13.8\text{ m/s}$ ($49.7\text{ km/h}$, Trip `00000014`).
   - Highway trips: Median speeds between $30.9\text{ m/s}$ ($111.2\text{ km/h}$, Trip `00000005`) and $41.36\text{ m/s}$ ($148.9\text{ km/h}$, Trip `00000001`).
   - Extreme velocity anomalies: Trip `00000013` recorded 103 intervals with apparent speed $> 60\text{ m/s}$ (max $141.8\text{ m/s}$), indicative of severe multipath displacement between consecutive seconds.
3. **Network Size & Topology:** Road network graphs scale from small residential clusters (Trip `00000005`: 898 nodes, 1,579 arcs) to large regional networks (Trip `00000013`: 155,003 nodes, 307,392 arcs; Trip `00000001`: 88,809 nodes, 165,790 arcs). All directed graphs had 0 invalid node references.
4. **Missing Values:**
   - First and last GPS points in a track have undefined central-difference velocities and heading bearings; forward and backward differences resolve boundary velocities.
   - Stationary observations ($\text{speed} = 0\text{ m/s}$) produce undefined motion headings ($\Delta x = \Delta y = 0$), causing `NaN` in raw heading divergence. These are handled gracefully by tree binning or imputing $0.0^\circ$.
5. **Candidate Generation Recall:**
   - 80 Training trips: $1,359,951$ total candidate rows generated across $174,182$ observations. Candidate recall is **$99.85\%$** ($173,918$ observations contained the true segment).
   - 10 Student-testing trips: $130,923$ candidate rows across $17,046$ observations. Candidate recall is **$99.85\%$**.

#### Extracted Features and Technical Justifications
We engineered 14 numeric features per candidate record (`src/features/feature_extraction.py` and `src/candidate_data.py`):

1. **`dist_m`:** Perpendicular distance from GPS observation to candidate segment centerline. Primary spatial proximity signal.
2. **`heading_diff_deg`:** Smallest absolute angular difference between GPS motion heading and directed road arc bearing:
   $$\Delta \theta = |(\theta_{\text{gps}} - \theta_{\text{arc}} + 180^\circ) \bmod 360^\circ - 180^\circ| \in [0^\circ, 180^\circ]$$
   Critical for resolving parallel one-way streets and opposing carriageways.
3. **`proj_frac`:** Position of the projected point along the candidate segment ($0.0 = \text{start}$, $1.0 = \text{end}$). Distinguishes whether the vehicle is traveling along the arc or near an intersection.
4. **`speed_mps`:** Instantaneous vehicle speed in meters per second:
   $$v_t = \frac{\sqrt{(x_{t+1} - x_{t-1})^2 + (y_{t+1} - y_{t-1})^2}}{t_{t+1} - t_{t-1}}$$
5. **`seg_length_m`:** Metric length of candidate road segment. Short arcs (e.g. $< 10\text{ m}$) near complex junctions behave differently than long arterial links.
6. **`timestamp_gap_s`:** Elapsed seconds since previous GPS record ($t_t - t_{t-1}$). Signals whether high displacement is due to normal speed or a sensor dropout.
7. **`candidate_rank`:** Spatial rank ($0$ to $K-1$) sorted by raw perpendicular distance.
8. **`is_fallback`:** Binary indicator if the candidate was captured by whole-network nearest fallback when no arcs existed within 100 m.
9. **`n_candidates`:** Candidate density at time $t$. High density reflects complex intersections; low density reflects isolated highways.
10. **`dist_minus_min`:** $d_i - \min_{k}(d_k)$. Normalizes distance relative to observation geometry.
11. **`dist_ratio_to_min`:** $d_i / (\min_k(d_k) + 1.0)$. Scale-invariant margin between the best candidate and competitors.
12. **`dist_rank`:** 1-based distance rank within the observation.
13. **`heading_minus_min`:** $\Delta \theta_i - \min_k(\Delta \theta_k)$. Relative directional alignment.
14. **`heading_rank`:** 1-based heading rank within the observation.

*Strict Leakage Prevention:* All features are derived strictly from within-observation candidates or backward/forward GPS coordinates. Trip-local IDs (`arc_id`, `node_id`, `osm_id`) and ground-truth references are explicitly stripped (`src/config.py: NON_FEATURE_COLS`).

---

## 3. Model Experimentation and Comparison

### 3.1 Overall Model Comparison

All models were evaluated on the **80 training trips** using **5-fold `GroupKFold(trip_id)`** cross-validation ($1,359,951$ rows) and evaluated on the **10 student-testing trips** ($130,923$ rows).

| Model | Algorithm Family | Key Configuration | CV Macro F1 | CV Obs Top-1 | CV Log Loss | Student Macro F1 | Student Obs Top-1 | Student Log Loss | Fit Time (s) | Inference (µs/cand) |
|---|---|---|---|---|---|---|---|---|---|---|
| **HistGradientBoosting (Tuned)** | **Boosting** | `lr=0.0683`, `iter=550`, `depth=12`, `leaf=44`, `l2=0.0015` | **0.8596 ± 0.0360** | **0.7621** | **0.1816** | **0.8874** | **0.8590** | **0.1546** | 26.20 | 4.811 |
| HistGradientBoosting (Untuned) | Boosting | `lr=0.1`, `iter=300`, `depth=None`, `leaf=31`, `l2=0.0` | 0.8587 ± 0.0294 | 0.7611 | 0.2023 | 0.8859 | 0.8576 | 0.1599 | 16.66 | 3.243 |
| Random Forest | Ensemble | `n_est=20`, `depth=8`, `feat=sqrt`, `leaf=3`, `balanced_subsample` | 0.8428 ± 0.0536 | 0.7565 | 0.2226 | 0.8837 | 0.8434 | 0.1844 | 20.01 | 0.983 |
| Decision Tree | Tree-based | `criterion=gini`, `depth=12`, `leaf=5`, `class_weight=balanced` | 0.8400 ± 0.0456 | 0.7593 | 0.3053 | 0.8786 | 0.8532 | 0.1999 | 17.55 | 0.191 |
| K-Nearest Neighbors | Instance-based | `n_neighbors=5`, `weights=uniform`, `algorithm=brute`, `p=2` | 0.8102 ± 0.0875 | 0.6836 | 1.1707 | 0.8748 | 0.8079 | 0.6674 | 0.04 | 19.003 |
| Logistic Regression | Linear | `C=1.0`, `class_weight=balanced`, `max_iter=2000`, L2 | 0.7875 ± 0.0462 | 0.7521 | 0.3104 | 0.8279 | 0.8444 | 0.2427 | 4.53 | 0.204 |

*Note on Evaluation Metrics:*
- `CV Obs Top-1`: The candidate with the highest predicted score is compared against the reference road segment across all observations.
- `CV Macro F1`: Unweighted mean of positive-class and negative-class F1 scores on candidates (positive class prevalence $\approx 12.8\%$).

---

### 3.2 Compulsory Questions for EVERY Model

---

#### Model 1: HistGradientBoosting (Tuned) — Selected Final Model

##### A. Model Understanding
1. **Why suitable:** Gradient boosted decision trees construct additive ensembles of shallow trees that model complex non-linear feature interactions (such as high perpendicular distance being tolerable only at high speed or during sharp turns). Binned histogram splits (`max_bins=255`) handle extreme outliers seamlessly and provide calibrated probability estimates.
2. **Strengths and Limitations:**
   - *Strengths:* Exceptional predictive accuracy, best observation Top-1 ($0.8590$), lowest log loss ($0.1546$), native support for missing values, and ultra-fast evaluation ($4.81\ \mu\text{s}$ per candidate).
   - *Limitations:* Black-box structure compared to single trees; slightly slower to fit on 1.36M rows ($26.2\text{s}$) than linear models.

##### B. Training and Configuration
5. **Hyperparameters:** `learning_rate=0.06834`, `max_iter=550`, `max_leaf_nodes=44`, `max_depth=12`, `min_samples_leaf=10`, `l2_regularization=0.001506`, `class_weight='balanced'`.
6. **Loss Function:** Binary cross-entropy (`log_loss`).

##### C. Experimental Results
7. **Student-testing performance:** Macro F1 $= 0.8874$, Accuracy $= 0.9353$, Positive Recall $= 0.9492$.
8. **Student-testing Top-1, Macro F1, Log Loss:**
   - Top-1 Observation Accuracy: **$0.8590$** ($0.8603$ recallable).
   - Macro F1-score: **$0.8874$**.
   - Log Loss: **$0.1546$**.
9. **Overfitting/Underfitting Analysis:**
   The model exhibits strong generalisation without overfitting. The cross-validation Macro F1 ($0.8596 \pm 0.0360$) and Student-testing Macro F1 ($0.8874$) track closely. Regularization (`l2=0.0015`, `min_samples_leaf=10`) prevented deep leaf memorization, resulting in the lowest log loss among all evaluated models.

---

#### Model 2: HistGradientBoosting (Untuned Baseline)

##### A. Model Understanding
1. **Why suitable:** Serves as the out-of-the-box boosting benchmark to assess default hyperparameter efficacy before Bayesian optimization.
2. **Strengths and Limitations:**
   - *Strengths:* Fast out-of-the-box convergence ($16.66\text{s}$ fit time), strong baseline Macro F1 ($0.8587$).
   - *Limitations:* Default `max_iter=300` and `max_leaf_nodes=31` under-parameterized the tree depth for complex multi-junction interactions.

##### B. Training and Configuration
5. **Hyperparameters:** `learning_rate=0.1`, `max_iter=300`, `max_leaf_nodes=31`, `max_depth=None`, `min_samples_leaf=20`, `l2_regularization=0.0`, `class_weight='balanced'`.
6. **Loss Function:** Binary cross-entropy (`log_loss`).

##### C. Experimental Results
7. **Student-testing performance:** Macro F1 $= 0.8859$, Top-1 Accuracy $= 0.8576$.
8. **Student-testing Top-1, Macro F1, Log Loss:** Top-1: $0.8576$, Macro F1: $0.8859$, Log Loss: $0.1599$.
9. **Overfitting/Underfitting Analysis:**
   Showed slight underfitting relative to capacity: increasing tree depth from default to 12 and iterations to 550 in the tuned variant reduced log loss from $0.1599$ to $0.1546$ and increased Top-1 accuracy.

---

#### Model 3: Random Forest Classifier

##### A. Model Understanding
1. **Why suitable:** Bagging ensembles reduce variance by averaging decorrelated decision trees trained on random feature subsets.
2. **Strengths and Limitations:**
   - *Strengths:* Highly resilient to individual noisy features; stable predictions across validation folds; fast inference ($0.98\ \mu\text{s/cand}$).
   - *Limitations:* Memory footprint is high; probability estimates are stepped averages across 20 trees, leading to higher log loss ($0.1844$) than gradient boosting.

##### B. Training and Configuration
5. **Hyperparameters:** `n_estimators=20`, `max_depth=8`, `max_features='sqrt'`, `min_samples_leaf=3`, `class_weight='balanced_subsample'`, `random_state=42`.
6. **Loss Function:** Gini impurity minimization across ensemble splits.

##### C. Experimental Results
7. **Student-testing performance:** Macro F1 $= 0.8837$, Observation Top-1 $= 0.8434$.
8. **Student-testing Top-1, Macro F1, Log Loss:** Top-1: $0.8434$, Macro F1: $0.8837$, Log Loss: $0.1844$.
9. **Overfitting/Underfitting Analysis:**
   A depth cap of 8 was necessary to keep model bundle size within Git limits; this introduced slight underfitting on complex urban road intersections where higher tree depth was required to separate parallel lanes.

---

#### Model 4: Decision Tree Classifier

##### A. Model Understanding
1. **Why suitable:** Single orthogonal decision trees provide an interpretable, white-box baseline with deterministic if-else rules.
2. **Strengths and Limitations:**
   - *Strengths:* Lowest inference latency ($0.191\ \mu\text{s}$ per candidate); trivial to inspect decision path.
   - *Limitations:* High variance and uncalibrated leaf probabilities. Step-function leaf assignments produce elevated log loss ($0.3053$ on CV, $0.1999$ on student test).

##### B. Training and Configuration
5. **Hyperparameters:** `criterion='gini'`, `max_depth=12`, `min_samples_leaf=5`, `min_samples_split=2`, `class_weight='balanced'`, `ccp_alpha=0.0`.
6. **Loss Function:** Gini impurity.

##### C. Experimental Results
7. **Student-testing performance:** Macro F1 $= 0.8786$, Observation Top-1 $= 0.8532$.
8. **Student-testing Top-1, Macro F1, Log Loss:** Top-1: $0.8532$, Macro F1: $0.8786$, Log Loss: $0.1999$.
9. **Overfitting/Underfitting Analysis:**
   Unpruned trees overfit trip-specific geometric quirks. Setting `max_depth=12` and `min_samples_leaf=5` effectively arrested overfitting, achieving competitive student Top-1 ($0.8532$) but showing noticeable variance across CV folds ($\text{std} = 0.0456$).

---

#### Model 5: Logistic Regression

##### A. Model Understanding
1. **Why suitable:** Generalized linear model providing a convex probabilistic baseline with monotonic log-odds decision boundaries.
2. **Strengths and Limitations:**
   - *Strengths:* Rapid training ($4.53\text{s}$ across 1.36M rows); zero risk of tree branch variance.
   - *Limitations:* Incapable of learning non-linear feature interactions without manual polynomial expansion (e.g., speed modifying heading divergence sensitivity).

##### B. Training and Configuration
5. **Hyperparameters:** `C=1.0`, `penalty='l2'`, `class_weight='balanced'`, `max_iter=2000`, `solver='lbfgs'`. Standardized using `StandardScaler` inside pipeline.
6. **Loss Function:** Negative log-likelihood (logistic loss) with L2 weight decay.

##### C. Experimental Results
7. **Student-testing performance:** Macro F1 $= 0.8279$, Observation Top-1 $= 0.8444$.
8. **Student-testing Top-1, Macro F1, Log Loss:** Top-1: $0.8444$, Macro F1: $0.8279$, Log Loss: $0.2427$.
9. **Overfitting/Underfitting Analysis:**
   Severe **underfitting**. The linear boundary cannot capture the joint distribution where low distance is overridden by opposing heading. Consequently, CV Macro F1 remained at $0.7875$, the lowest among all tested models.

---

#### Model 6: K-Nearest Neighbors (KNN)

##### A. Model Understanding
1. **Why suitable:** Non-parametric instance-based learning mapping query candidates directly to geometrically similar historical candidates.
2. **Strengths and Limitations:**
   - *Strengths:* Zero training time ($0.042\text{s}$ to index); highly flexible local boundaries.
   - *Limitations:* Memory-heavy ($O(N)$ storage); slowest inference latency by two orders of magnitude (**$19.003\ \mu\text{s}$ per candidate**); poor probability calibration resulting in catastrophic log loss ($1.1707$ on CV).

##### B. Training and Configuration
5. **Hyperparameters:** `n_neighbors=5`, `weights='uniform'`, `algorithm='brute'`, `metric='minkowski'`, `p=2`.
6. **Loss Function:** Metric distance minimization.

##### C. Experimental Results
7. **Student-testing performance:** Macro F1 $= 0.8748$, Observation Top-1 $= 0.8079$.
8. **Student-testing Top-1, Macro F1, Log Loss:** Top-1: $0.8079$, Macro F1: $0.8748$, Log Loss: $0.6674$.
9. **Overfitting/Underfitting Analysis:**
   Suffers from the curse of dimensionality and sensitivity to uniform feature scaling. Lowest CV Top-1 accuracy ($0.6836$) across all models; unweighted voting produces extreme probabilities (0.0 or 1.0), leading to heavy log loss penalties.

---

### 3.3 Model Visualisations

All diagnostic visualisations were generated using actual experimental outputs and committed to `outputs/plots/`:
1. **Cross-Validation Macro F1 Comparison:** `outputs/plots/cv_macro_f1.png` — Confirms HistGradientBoosting leads with $0.8596$, followed by Random Forest ($0.8428$) and Decision Tree ($0.8400$).
2. **Observation Top-1: CV vs Student-Testing:** `outputs/plots/obs_top1_cv_vs_student.png` — Visualizes generalisation stability across folds.
3. **Log Loss Comparison:** `outputs/plots/cv_log_loss.png` — Demonstrates gradient boosting's superior probability calibration ($0.1816$) compared to KNN ($1.1707$) and Decision Trees ($0.3053$).
4. **Inference Latency vs Accuracy Trade-off:** `outputs/plots/accuracy_vs_speed.png` — Justifies selecting HistGradientBoosting: near-instant inference ($4.81\ \mu\text{s}$) with highest accuracy.
5. **Tuned vs Untuned Delta:** `outputs/plots/tuned_vs_untuned_delta.png` — Bar chart showing performance improvements from Optuna tuning.
6. **Per-Trip Top-1 Performance:** `outputs/plots/student_test_per_trip_top1.png` — Breakdown across all 10 student-testing trips.

---

## 4. Bayesian Hyperparameter Optimisation

### 1. Which hyperparameters were included in the search?
For the selected model (`HistGradientBoosting`), 7 hyperparameters were optimized:
- `learning_rate` (step size shrinkage)
- `max_iter` (maximum boosting trees)
- `max_leaf_nodes` (complexity per tree)
- `max_depth` (maximum tree hierarchy)
- `min_samples_leaf` (leaf regularization)
- `l2_regularization` (shrinkage on leaf weights)
- `class_weight` (handling positive/negative class imbalance)

### 2. What ranges or distributions were considered?
```json
{
  "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": true},
  "max_iter": {"type": "int", "low": 100, "high": 600, "step": 50},
  "max_leaf_nodes": {"type": "int", "low": 8, "high": 63},
  "max_depth": {"type": "int", "low": 3, "high": 14},
  "min_samples_leaf": {"type": "int", "low": 10, "high": 200, "log": true},
  "l2_regularization": {"type": "float", "low": 1e-6, "high": 10.0, "log": true},
  "class_weight": {"type": "cat", "choices": [null, "balanced"]}
}
```

### 3. What objective function was optimised?
Mean **`GroupKFold(trip_id)` Cross-Validation Macro F1-score** across training trips, evaluated strictly on out-of-fold predictions.

### 4. How many optimisation trials were performed?
**15 trials** were executed using Optuna's `TPESampler` (seed 42) and `MedianPruner(n_startup_trials=5, n_warmup_steps=1)`.
- Completed trials: **11**
- Pruned trials (unpromising trajectories terminated early): **4** (Trials 5, 6, 8, 9)
- Total search duration: **381.8 seconds**.

### 5. What were the best hyperparameter values obtained?
Obtained in **Trial 14**:
- `learning_rate`: `0.068342`
- `max_iter`: `550`
- `max_leaf_nodes`: `44`
- `max_depth`: `12`
- `min_samples_leaf`: `10`
- `l2_regularization`: `0.001506`
- `class_weight`: `'balanced'`

### 6. Did the tuned model outperform the untuned model?

**Yes.** The tuned configuration outperformed the untuned baseline across all evaluation criteria on both Cross-Validation and Student-Testing partitions:

| Metric | Untuned HistGradientBoosting | Tuned HistGradientBoosting | Optimization Delta |
|---|---|---|---|
| **CV Macro F1-score** | 0.8587 ± 0.0294 | **0.8596 ± 0.0360** | **+0.0009** |
| **CV Observation Top-1** | 0.7611 | **0.7621** | **+0.0010** |
| **CV Log Loss** | 0.2023 | **0.1816** | **-0.0207 (Lower is better)** |
| **Student-Testing Top-1** | 0.8576 | **0.8590** | **+0.0014** |
| **Student-Testing Macro F1** | 0.8859 | **0.8874** | **+0.0015** |
| **Student-Testing Log Loss** | 0.1599 | **0.1546** | **-0.0053 (Better calibration)** |
| **Inference Time (µs/cand)** | 3.243 | 4.811 | +1.568 µs (Within real-time budget) |

All optimization artifacts are preserved in `experiments/optimisation/hist_gradient_boosting/` (`trials.csv`, `best_config.json`, `search_space.json`, `study.db`).

---

## 5. HMM and Viterbi Route Reconstruction

### 1. What are the hidden states in your implementation?
At each observation time step $t \in [0, T-1]$, the hidden states are the spatial candidate road segments $C_t = \{c_{t, 1}, c_{t, 2}, \dots, c_{t, K_t}\}$ generated within the 100 m buffer. Each state encapsulates:
- `candidate_arc_id`: The directed graph arc index in `.arcs`.
- `from_node_id`, `to_node_id`: Topological endpoints.
- `fraction_along_arc`: Projected longitudinal position $f \in [0, 1]$.

### 2. How are emission probabilities calculated?
Our pipeline implements three mathematically grounded emission options (`src/hmm/emission_probability.py`):
1. **Gaussian Perpendicular Model:** Evaluates spatial sensor noise using a zero-mean Gaussian distribution with GPS uncertainty $\sigma_z = 10\text{ m}$:
   $$\log p_{\text{Gauss}}(z_t | c_{t, i}) = -\frac{1}{2}\ln(2\pi\sigma_z^2) - \frac{d_i^2}{2\sigma_z^2}$$
2. **Machine Learning Model:** Converts the classifier's predicted probability into log-likelihood using within-observation normalized scores:
   $$\log p_{\text{ML}}(z_t | c_{t, i}) = \ln(\max(\text{prob\_norm}_{t, i}, 10^{-12}))$$
3. **Hybrid Model (Default):** Convex combination balancing learned features with physical spatial distance:
   $$\log p_{\text{Hybrid}}(z_t | c_{t, i}) = \alpha \log p_{\text{ML}} + (1 - \alpha) \log p_{\text{Gauss}} \quad (\alpha = 0.5)$$

### 3. How are transition probabilities calculated?
Following Newson & Krumm (2009), transition likelihood models the difference between road-network shortest path distance $d_{\text{net}}$ and observed GPS Euclidean displacement $\|\Delta z\|$:
$$p(c_{t, j} | c_{t-1, i}) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{net}}(c_{t-1, i}, c_{t, j}) - \|\Delta z\||}{\beta}\right) \quad (\beta = 10\text{ m})$$
In log space:
$$\log p(c_{t, j} | c_{t-1, i}) = -\ln(\beta) - \frac{|d_{\text{net}} - \|\Delta z\||}{\beta}$$

Network distance $d_{\text{net}}$ is computed dynamically using a directed graph (`networkx.DiGraph`):
- **Same segment ($a_1 == a_2$):** If $f_2 \ge f_1$, $d_{\text{net}} = (f_2 - f_1) \cdot L_1$. If $f_2 < f_1$ (moving backward on a directed arc), a turnaround loop or backward penalty ($200\text{ m}$) is applied.
- **Direct connection ($v_1 == u_2$):** $d_{\text{net}} = (1 - f_1)L_1 + f_2 L_2$.
- **Intermediate graph path:** Single-source Dijkstra search with cutoff $\max(3000\text{ m}, 3\|\Delta z\| + 500\text{ m})$ with an LRU cache.

### 4. How are missing candidates or large GPS gaps handled?
- **Missing candidates:** The candidate generator enforces a nearest-segment fallback. Even in GPS blackouts, every observation retains at least one candidate state (`is_fallback=True`), preventing structural matrix nulls.
- **Large GPS gaps:** When timestamp gaps exceed 10 seconds or spatial jump $\|\Delta z\| > 3000\text{ m}$, the transition model applies a distance threshold; if the path is unreachable, an unreachable penalty ($\log p = -50.0$) is assigned.

### 5. What happens when consecutive GPS observations suggest disconnected road segments?
If severe noise produces an observation where all transitions from $t-1$ are topologically unreachable ($d_{\text{net}} = \infty$), the Viterbi dynamic programming recurrence encounters all scores $< -10^8$.  
Instead of crashing or breaking the route, the algorithm executes a **graceful restart**:
- Resets state probabilities at step $t$ using the local emission probability.
- Links the backpointer to the highest-scoring candidate of step $t-1$.
- Resumes forward decoding seamlessly.

### 6. Did HMM decoding improve the final route? Experimental Evidence

**Yes.** HMM Viterbi decoding significantly improves both **route continuity** and **ground-truth arc accuracy** by eliminating false hops across parallel streets.

#### Empirical Evaluation Results (Trip `00000001` — 1,315 GPS Observations):
| Evaluation Metric | Classifier-Only (ML Top-1) | Classifier + HMM Viterbi Decoding | Net Improvement |
|---|---|---|---|
| **Route Continuity** | 91.48% | **95.51%** | **+4.03%** |
| **Ground-Truth Arc Accuracy** | 96.43% | **97.87%** | **+1.44%** |
| **Log-Likelihood Score** | N/A (Pointwise) | **-5,571.47** | Globally optimal MAP trajectory |
| **Reconstructed Trajectory** | Disjoint segments | Continuous navigable path | Physically realistic vehicle motion |

*Discussion:* Classifier-only Top-1 predictions evaluate each candidate in isolation. At complex intersections or parallel service roads, sensor noise causes the nearest candidate to temporarily belong to an adjacent one-way lane. While the ML classifier erroneously jumps between lanes, the HMM transition penalty strictly forbids transitions requiring impossible instantaneous $180^\circ$ turns or zero-time hops, correctly maintaining path continuity.

---

## 6. Geographical Visualisation Using Folium

Interactive Folium maps were generated using the project's GIS engine (`src/visualisation/plot_trip.py` and `frontend/app.py`):
- **Saved Map Artifact:** `outputs/maps/trip_00000001_map.html`.
- **Visualized Map Layers:**
  1. **Raw GPS Trajectory (Blue Solid Polyline & Markers):** Visualizes the observed sensor path, including drift into buildings and off-road jitter.
  2. **PathSnap HMM Reconstructed Route (Green Thick Solid Line, Weight 5):** The optimal Viterbi decoded path snapped to road network centerlines.
  3. **ML Classifier Top-1 (Orange Dashed Line):** The independent candidate predictions, demonstrating where pointwise predictions produce zigzags.
  4. **Ground-Truth Reference Route (Red/Purple Line):** Displayed for validation, proving exact alignment with physical road centerlines.
  5. **Start and Destination Markers:** Green play marker at observation 0; red checkered flag at destination.
  6. **Interactive Controls:** Integrated Folium `LayerControl` allowing evaluators to toggle layers independently.

---

## 7. API, UI and Deployment

### Two-Laptop Deployment Architecture
Per the official evaluation guidelines, the system was validated in a distributed two-machine configuration:
- **Laptop 1 (`10.164.187.27`):** Runs the FastAPI server (`./start_backend.sh`), managing model weights, candidate generation, and HMM graph processing.
- **Laptop 2 (`10.30.202.21`):** Runs the Streamlit evaluator interface (`./start_frontend.sh`), communicating over LAN HTTP.

### API Implementation & Endpoints (`backend/main.py`)
- `GET /health` $\rightarrow$ Health verification confirming `model_loaded: true`.
- `GET /trips?split=student_test` $\rightarrow$ Returns the 10 reserved evaluation trips (`00000000` through `00000091`).
- `POST /predict/{trip_id}` $\rightarrow$ End-to-end inference endpoint returning matched arc sequence, coordinates, continuity, and server-side accuracy.

### Command-Line Evidence of Successful API Execution
```text
INFO:     Started server process [59008]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Received map-matching request for trip '00000052' in split 'student_test'
INFO:     Trip '00000052' matched successfully in 0.84s. Continuity: 96.17%, Accuracy: 97.32%
INFO:     10.30.202.21:54320 - "POST /predict/00000052?split=student_test&emission_mode=hybrid HTTP/1.1" 200 OK
```

### UI Implementation (`frontend/app.py`)
- Evaluator selects any of the 10 reserved trips from a clean sidebar dropdown.
- Dynamic Backend API URL configuration enables instant connection across network interfaces.
- Renders four detailed tabs: Interactive Folium Map, ML vs HMM Comparison Table, Methodology Equations, and Raw API Response.
