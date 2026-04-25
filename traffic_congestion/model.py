from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier


def build_model(random_state: int = 42) -> DecisionTreeClassifier:
    """
    Shallow decision tree: easy to export to if/else and runs fast on Arduino-class MCUs.
    """
    return DecisionTreeClassifier(
        max_depth=4,
        criterion="gini",
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight="balanced",
        random_state=random_state,
    )

