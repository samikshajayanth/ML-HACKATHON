"""Model 5: HistGradientBoostingClassifier (handles NaN natively).

early_stopping=False on purpose: sklearn's internal validation split is a random
ROW split, which would break trip grouping. We tune max_iter / learning_rate instead.
"""
from sklearn.ensemble import HistGradientBoostingClassifier

NAME = "hist_gradient_boosting"
DEFAULT_PARAMS = {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "max_depth": None,
                  "min_samples_leaf": 20, "l2_regularization": 0.0, "class_weight": "balanced"}


def build(params=None, seed=42):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **p)
