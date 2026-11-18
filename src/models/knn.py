"""Model 2: K-Nearest Neighbours (distance-based; needs scaling).

KNN has no class_weight, so imbalance is handled with weights="distance".
Inference cost grows with the training-set size - the timing columns will show it.
"""
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "knn"
DEFAULT_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "p": 2,
    "algorithm": "brute",
    "leaf_size": 50
}


def build(params=None, seed=42):  # seed unused: KNN is deterministic
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_jobs=-1, **p)),
    ])
