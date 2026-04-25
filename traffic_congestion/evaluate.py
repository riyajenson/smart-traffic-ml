from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from .config import CLASS_ORDER


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str
    cv_scores: np.ndarray


def evaluate_model(clf, X_train, y_train, X_test, y_test, X_all=None, y_all=None, cv_folds: int = 5) -> EvaluationResult:
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    # sklearn report needs explicit label order to match our class order
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, target_names=CLASS_ORDER)

    if X_all is None:
        X_all = np.vstack([X_train, X_test])
    if y_all is None:
        y_all = np.concatenate([y_train, y_test])

    cv_scores = cross_val_score(clf, X_all, y_all, cv=cv_folds, scoring="accuracy")

    return EvaluationResult(
        accuracy=acc,
        confusion_matrix=cm,
        classification_report=report,
        cv_scores=cv_scores,
    )

