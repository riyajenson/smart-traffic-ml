from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree

from .config import CLASS_ORDER


def save_evaluation_plots(df, y_test, y_pred, feature_names: list[str], importances: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_idx = np.argsort(importances)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Traffic Congestion Classifier — Evaluation", fontsize=14, fontweight="bold")

    cm = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        display_labels=CLASS_ORDER,
        cmap="Blues",
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")

    colors = ["#00c896" if i == sorted_idx[0] else "#4d9fff" for i in range(len(feature_names))]
    axes[1].barh(feature_names, importances, color=colors)
    axes[1].set_xlabel("Gini Importance")
    axes[1].set_title("Feature Importances")
    axes[1].invert_yaxis()

    for cls, color in zip(["LOW", "MEDIUM", "HIGH"], ["#00c896", "#ffb830", "#ff4d6d"]):
        speeds = df[df["label"] == cls]["avg_speed"]
        axes[2].hist(speeds, bins=25, alpha=0.65, label=cls, color=color)
    axes[2].set_xlabel("Average Speed (cm/s)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Speed Distribution by Class")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_tree_plot(clf, feature_names: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=CLASS_ORDER,
        filled=True,
        rounded=True,
        fontsize=9,
        impurity=True,
    )
    plt.title("Decision Tree — Traffic Congestion Classifier", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

