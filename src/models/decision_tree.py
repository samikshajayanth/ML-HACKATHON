"""Model 3: Decision Tree (interpretable; no scaling needed)."""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

NAME = "decision_tree"
DEFAULT_PARAMS = {"max_depth": 12, "min_samples_leaf": 5, "min_samples_split": 2,
                  "criterion": "gini", "ccp_alpha": 0.0, "class_weight": "balanced"}


def build(params=None, seed=42):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(random_state=seed, **p)),
    ])
