"""
Plotting utilities for evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from ..Model.config import CLASS_ALIASES

# Plot styling
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.facecolor"] = "#F4F7FD"
plt.rcParams["axes.facecolor"] = "#FAFCFF"
plt.rcParams["grid.color"] = "#E4EAED"

COLORS = {"blue": "#1f77b4", "green": "#2ca02c", "orange": "#ff7f0e", "gray": "#7f7f7f"}


def plot_metrics(
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies,
    classes,
    all_labels,
    all_preds,
    all_probs,
    fold_idx=1,
):
    """
    Plot training metrics including loss, accuracy, confusion matrix, and ROC curves

    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        classes: List of class names
        all_labels: Ground truth labels
        all_preds: Predicted labels
        all_probs: Predicted probabilities
        fold_idx: Fold index
    """
    output_dir = f"fold_{fold_idx}_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot loss curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss", color=COLORS["blue"])
    plt.plot(val_losses, label="Val Loss", color=COLORS["orange"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Fold {fold_idx} Loss Curve")
    plt.savefig(
        os.path.join(output_dir, f"fold_{fold_idx}_loss_curve.pdf"), format="pdf"
    )
    plt.close()

    # Plot accuracy curve
    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy", color=COLORS["blue"])
    plt.plot(val_accuracies, label="Val Accuracy", color=COLORS["orange"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title(f"Fold {fold_idx} Accuracy Curve")
    plt.savefig(
        os.path.join(output_dir, f"fold_{fold_idx}_accuracy_curve.pdf"), format="pdf"
    )
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    alias_labels = [CLASS_ALIASES[cls] for cls in classes]
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=alias_labels,
        yticklabels=alias_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Fold {fold_idx} Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"fold_{fold_idx}_confusion_matrix.pdf"), format="pdf"
    )
    plt.close()

    # Plot ROC curve
    y_bin = label_binarize(all_labels, classes=range(len(classes)))
    n_classes = len(classes)
    plt.figure()
    color_cycle = cycle(list(COLORS.values()))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{alias_labels[i]} (AUC = {roc_auc:.2f})",
            color=next(color_cycle),
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Fold {fold_idx} ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(output_dir, f"fold_{fold_idx}_roc_curve.pdf"), format="pdf"
    )
    plt.close()
