"""
Training and validation functions for STRATIX-Net
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=100,
    fold_idx=1,
):
    """
    Training and evaluation function

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        fold_idx: Fold index for saving models

    Returns:
        Tuple of (train_losses, train_accuracies, val_losses, val_accuracies)
    """
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_state = None
    best_states_dir = "Best_States"
    os.makedirs(best_states_dir, exist_ok=True)

    for epoch in range(num_epochs):
        try:
            # Training phase
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            train_bar = tqdm(
                train_loader,
                desc=f"Fold {fold_idx} Epoch {epoch+1}/{num_epochs} [Train]",
            )

            for batch in train_bar:
                if batch is None:
                    continue
                inputs, labels = batch
                device = next(model.parameters()).device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)
                train_bar.set_postfix(
                    loss=running_loss / total, acc=(correct / total) * 100
                )

            epoch_loss = running_loss / total if total > 0 else float("inf")
            epoch_acc = (correct / total) * 100 if total > 0 else 0.0
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # Validation phase
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            val_bar = tqdm(
                val_loader, desc=f"Fold {fold_idx} Epoch {epoch+1}/{num_epochs} [Val]"
            )

            with torch.no_grad():
                for batch in val_bar:
                    if batch is None:
                        continue
                    inputs, labels = batch
                    device = next(model.parameters()).device
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels.data).item()
                    total += labels.size(0)
                    val_bar.set_postfix(
                        loss=val_loss / total, acc=(correct / total) * 100
                    )

            val_loss = val_loss / total if total > 0 else float("inf")
            val_acc = (correct / total) * 100 if total > 0 else 0.0
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                model_path = os.path.join(
                    best_states_dir, f"best_model_fold_{fold_idx}.pth"
                )
                try:
                    torch.save(
                        {
                            "model_state_dict": best_model_state,
                            "best_val_acc": best_val_acc,
                        },
                        model_path,
                    )
                    logger.info(
                        f"Saved best model state for fold {fold_idx} to {model_path} with val acc {best_val_acc:.4f}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save model state for fold {fold_idx} to {model_path}: {str(e)}"
                    )
                    try:
                        temp_path = os.path.join(
                            best_states_dir, f"temp_best_model_fold_{fold_idx}.pth"
                        )
                        torch.save(
                            {
                                "model_state_dict": best_model_state,
                                "best_val_acc": best_val_acc,
                            },
                            temp_path,
                        )
                        logger.info(
                            f"Saved best model state to temporary file {temp_path}"
                        )
                    except Exception as e2:
                        logger.error(
                            f"Failed to save to temporary file {temp_path}: {str(e2)}"
                        )

            scheduler.step()
            print(
                f"Fold {fold_idx} Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {epoch_loss:.3f}, Train Acc: {epoch_acc:.2f}, "
                f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}"
            )
        except Exception as e:
            logger.error(
                f"Error during training epoch {epoch+1} for fold {fold_idx}: {str(e)}"
            )
            raise

    return train_losses, train_accuracies, val_losses, val_accuracies
