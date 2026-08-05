from __future__ import annotations

from typing import Any

import numpy as np


def count_estimator_parameters(estimator: Any) -> int:
    if hasattr(estimator, "coef_"):
        return int(np.asarray(estimator.coef_).size + np.asarray(estimator.intercept_).size)
    if hasattr(estimator, "estimators_"):
        estimators = np.asarray(estimator.estimators_, dtype=object).reshape(-1)
        return int(sum(tree.tree_.node_count for tree in estimators if hasattr(tree, "tree_")))
    if hasattr(estimator, "_predictors"):
        return int(
            sum(
                predictor.nodes.shape[0]
                for iteration in estimator._predictors
                for predictor in iteration
            )
        )
    if hasattr(estimator, "booster_"):
        model = estimator.booster_.dump_model()
        return int(sum(tree["num_leaves"] for tree in model["tree_info"]))
    return 0
