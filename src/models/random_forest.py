"""Model 4: Random Forest (bagged trees; no scaling needed)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NAME = "random_forest"
DEFAULT_PARAMS = {
    "n_estimators": 20,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample"
}

def build(params=None, seed=42):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1, **p)),
    ])
