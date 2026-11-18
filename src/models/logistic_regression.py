"""Model 1: Logistic Regression (linear baseline; needs scaling)."""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "logistic_regression"
# Imbalance: class_weight="balanced" (few positives per observation).
DEFAULT_PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 2000}


def build(params=None, seed=42):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=seed, **p)),
    ])
