"""
GastroVisionNet - Main Training and Evaluation Script
Support for both GastroVisionNet and MobileGastroVisionNet models
"""

import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import gc
import logging
from tqdm import tqdm

# Local imports
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Load_Dataset import get_data_loaders
from Model import get_model, count_parameters
from Training import train_and_evaluate, get_optimizer_scheduler
from Evaluation import test_model, plot_metrics
from Model.config import EXPECTED_CLASSES

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Hyperparameters
HYPERPARAMS = {
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_workers": 0,
    "image_size": (224, 224),
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss",
    "confidence_interval_z": 1.96,
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(
        description="GastroVisionNet Training and Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gastrovisionnet",
        choices=["gastrovisionnet", "mobilegastrovisionnet"],
        help="Model to train (default: gastrovisionnet)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to kvasir-v2-5folds directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for regularization (default: 1e-5)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loader workers (default: 0)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size as height width (default: 224 224)",
    )

    args = parser.parse_args()

    # Update hyperparameters from arguments
    HYPERPARAMS["batch_size"] = args.batch_size
    HYPERPARAMS["num_epochs"] = args.num_epochs
    HYPERPARAMS["learning_rate"] = args.learning_rate
    HYPERPARAMS["weight_decay"] = args.weight_decay
    HYPERPARAMS["num_workers"] = args.num_workers
    HYPERPARAMS["image_size"] = tuple(args.image_size)

    # Set model name
    model_name = args.model.lower()

    all_results = []
    logger.info(f"Starting training with PyTorch version {torch.__version__}")
    logger.info(f"Using device: {device}")
    logger.info(f"Training model: {model_name}")

    for fold_idx in range(1, 6):
        torch.cuda.empty_cache()
        print(f"\nProcessing Fold {fold_idx}")

        try:
            train_loader, val_loader, test_loader, classes, preloaded_data = (
                get_data_loaders(
                    fold_path=os.path.join(
                        args.base_path, f"kvasir-v2-dataset-run-{fold_idx}"
                    ),
                    batch_size=HYPERPARAMS["batch_size"],
                    image_size=HYPERPARAMS["image_size"],
                    num_workers=HYPERPARAMS["num_workers"],
                )
            )
        except Exception as e:
            logger.error(f"Failed to load data for fold {fold_idx}: {str(e)}")
            continue

        try:
            model = get_model(
                model_name=model_name, num_classes=len(classes), device=device
            )
            total_params, trainable_params = count_parameters(model)
            logger.info(
                f"Fold {fold_idx} - Total Parameters: {total_params:,}, Trainable Parameters: {trainable_params:,}"
            )
            print(
                f"Fold {fold_idx} - Total Parameters: {total_params:,}, Trainable Parameters: {trainable_params:,}"
            )

            criterion = nn.CrossEntropyLoss()
            optimizer, scheduler = get_optimizer_scheduler(
                model,
                learning_rate=HYPERPARAMS["learning_rate"],
                weight_decay=HYPERPARAMS["weight_decay"],
                num_epochs=HYPERPARAMS["num_epochs"],
                optimizer_type=HYPERPARAMS["optimizer"],
            )
        except Exception as e:
            logger.error(f"Failed to initialize model for fold {fold_idx}: {str(e)}")
            continue

        try:
            train_losses, train_accuracies, val_losses, val_accuracies = (
                train_and_evaluate(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    num_epochs=HYPERPARAMS["num_epochs"],
                    fold_idx=fold_idx,
                )
            )
        except Exception as e:
            logger.error(f"Failed to train fold {fold_idx}: {str(e)}")
            continue

        # Load best model and evaluate
        model_path = os.path.join("Best_States", f"best_model_fold_{fold_idx}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded best model for fold {fold_idx} with val acc {checkpoint['best_val_acc']:.4f}"
            )

        try:
            val_results, val_labels, val_preds, val_probs = test_model(
                model,
                val_loader,
                classes,
                fold_idx,
                dataset_type="val",
                confidence_interval_z=HYPERPARAMS["confidence_interval_z"],
            )
            print(f"\nValidation Metrics for Fold {fold_idx}:")
            for metric_name, value in val_results.items():
                if metric_name not in ["Total Parameters", "Trainable Parameters"]:
                    print(f" {metric_name}: {value}")

            test_results, test_labels, test_preds, test_probs = test_model(
                model,
                test_loader,
                classes,
                fold_idx,
                dataset_type="test",
                confidence_interval_z=HYPERPARAMS["confidence_interval_z"],
            )
            test_results["Fold"] = fold_idx
            test_results["Model"] = model_name
            all_results.append(test_results)

            print(f"\nTest Metrics for Fold {fold_idx}:")
            for metric_name, value in test_results.items():
                if metric_name not in [
                    "Fold",
                    "Total Parameters",
                    "Trainable Parameters",
                    "Model",
                ]:
                    print(f" {metric_name}: {value}")

            plot_metrics(
                train_losses,
                train_accuracies,
                val_losses,
                val_accuracies,
                classes,
                test_labels,
                test_preds,
                test_probs,
                fold_idx,
            )
        except Exception as e:
            logger.error(f"Failed to test or plot for fold {fold_idx}: {str(e)}")
            continue

        # Release memory
        print(f"Releasing memory for fold {fold_idx}")
        train_images, train_labels, test_images, test_labels = preloaded_data
        del train_images, train_labels, test_images, test_labels
        del train_loader, val_loader, test_loader, model
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    try:
        results_df = pd.DataFrame(all_results)
        output_file = f"kvasir_v2_classification_results_{model_name}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results CSV: {str(e)}")


if __name__ == "__main__":
    main()
