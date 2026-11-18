from . import decision_tree, gradient_boosting, knn, logistic_regression, random_forest

MODELS = {m.NAME: m for m in (logistic_regression, knn, decision_tree, random_forest, gradient_boosting)}
