"""
Model Evaluation Metrics Utilities.
Computes Top-1 & Top-5 accuracy, generates confusion matrices, and calculates classification reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from core.config import ASSETS_DIR, CIFAR10_CLASSES


def compute_top_k_accuracy(
    y_true: List[int],
    y_prob_preds: List[List[float]],
    k: int = 5,
) -> float:
    """
    Computes top-k categorical accuracy.
    """
    correct = 0
    total = len(y_true)
    if total == 0:
        return 0.0

    for true_label, probs in zip(y_true, y_prob_preds):
        top_k_indices = np.argsort(probs)[-k:][::-1]
        if true_label in top_k_indices:
            correct += 1

    return float(round((correct / total) * 100.0, 2))


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CIFAR10_CLASSES,
    output_path: Optional[Path] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix - CIFAR-10 Classification",
) -> Path:
    """
    Renders and exports a high-resolution confusion matrix heatmap.
    """
    if output_path is None:
        output_path = ASSETS_DIR / "confusion_matrix.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
    fmt = ".2f" if normalize else "d"

    plt.figure(figsize=(9, 7), dpi=300)
    sns.set_theme(style="white")

    ax = sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Sample Count"},
        linewidths=0.5,
        linecolor="#e0e0e0",
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=10)
    plt.ylabel("True Class", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def generate_evaluation_summary(
    y_true: List[int],
    y_pred: List[int],
    y_probs: List[List[float]],
    class_names: List[str] = CIFAR10_CLASSES,
    model_name: str = "MobileNetV2",
    save_json_path: Optional[Path] = None,
) -> Dict:
    """
    Computes complete benchmark evaluation suite and optionally dumps JSON.
    """
    top_1_acc = compute_top_k_accuracy(y_true, y_probs, k=1)
    top_5_acc = compute_top_k_accuracy(y_true, y_probs, k=5)
    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm_plot_path = plot_and_save_confusion_matrix(cm, class_names=class_names)

    summary = {
        "model_name": model_name,
        "dataset": "CIFAR-10",
        "total_test_samples": len(y_true),
        "top_1_accuracy": top_1_acc,
        "top_5_accuracy": top_5_acc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_plot": str(cm_plot_path.relative_to(ASSETS_DIR.parent)),
        "classification_report": clf_report,
    }

    if save_json_path is None:
        save_json_path = ASSETS_DIR / "evaluation_metrics.json"

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
