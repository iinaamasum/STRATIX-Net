"""
Metrics and parameter counting utilities
"""

import torch
import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def count_parameters(model):
    """Function to count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_model(
    model,
    dataloader,
    classes,
    fold_idx=1,
    dataset_type="test",
    confidence_interval_z=1.96,
):
    """
    Testing function with comprehensive metrics

    Args:
        model: The trained model
        dataloader: Data loader for test set
        classes: List of class names
        fold_idx: Fold index
        dataset_type: Type of dataset ('test', 'val', etc.)
        confidence_interval_z: Z-score for confidence intervals (default 1.96 for 95%)

    Returns:
        Tuple of (results_dict, all_labels, all_preds, all_probs)
    """
    model.eval()
    all_preds, all_labels, inference_times, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Fold {fold_idx} [{dataset_type.capitalize()}]"
        ):
            if batch is None:
                continue
            inputs, labels = batch
            device = next(model.parameters()).device
            inputs, labels = inputs.to(device), labels.to(device)

            start_infer = time.time()
            outputs = model(inputs)
            inference_times.append((time.time() - start_infer) * 1000 / inputs.size(0))

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    precision, recall, f1 = precision * 100, recall * 100, f1 * 100
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    roc_auc = roc_auc_score(
        label_binarize(all_labels, classes=range(len(classes))),
        all_probs,
        multi_class="ovr",
        average="weighted",
    )
    inference_time = np.mean(inference_times)

    # Calculate model size
    try:
        torch.save(model.state_dict(), "temp_model.pth")
        model_size = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
    except Exception as e:
        logger.error(f"Failed to save temporary model state: {str(e)}")
        model_size = 0.0

    # Calculate confidence intervals
    z = confidence_interval_z
    N = len(all_labels)
    metrics = {
        "Accuracy": accuracy / 100,
        "Precision": precision / 100,
        "Recall": recall / 100,
        "F1-Score": f1 / 100,
    }
    results = {}
    for metric_name, metric_value in metrics.items():
        r = z * np.sqrt(metric_value * (1 - metric_value) / N)
        results[f"{metric_name} ± r"] = f"{metric_value * 100:.4f} ± {r * 100:.4f}"

    results["MCC"] = f"{mcc:.4f}"
    results["Cohen's Kappa"] = f"{kappa:.4f}"
    results["ROC AUC"] = f"{roc_auc:.4f}"
    results["Inference-Time (ms)"] = f"{inference_time:.2f}"
    results["Model Size (MB)"] = f"{model_size:.2f}"

    # Add parameter counts
    total_params, trainable_params = count_parameters(model)
    results["Total Parameters"] = total_params
    results["Trainable Parameters"] = trainable_params

    return results, all_labels, all_preds, all_probs
