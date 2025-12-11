"""
Dataset loading utilities for STRATIX-Net
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from tqdm import tqdm
from ..Preprocessing.preprocessing import get_transforms


class PreloadedDataset(Dataset):
    """Custom Dataset for preloaded images"""

    def __init__(self, images, labels, class_map, original_classes):
        self.images = images
        self.labels = labels
        self.class_map = class_map
        self.original_classes = original_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        cls_name = self.original_classes[label]
        if cls_name not in self.class_map:
            return None
        new_label = self.class_map[cls_name]
        return self.images[idx], new_label


def custom_collate_fn(batch):
    """Custom collate function to filter out None values"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images, labels = zip(*batch)
    import torch

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def get_data_loaders(fold_path, batch_size=16, image_size=(224, 224), num_workers=0):
    """
    Define dataset loader with preloading

    Args:
        fold_path: Path to the fold directory containing train/test folders
        batch_size: Batch size for data loading
        image_size: Size for image resizing
        num_workers: Number of data loader workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader, expected_classes, preloaded_data)
    """
    data_transforms = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(
        os.path.join(fold_path, "train"), transform=None
    )
    test_dataset = datasets.ImageFolder(os.path.join(fold_path, "test"), transform=None)

    expected_classes = [
        "normal-cecum",
        "dyed-lifted-polyps",
        "dyed-resection-margins",
        "esophagitis",
        "normal-pylorus",
        "polyps",
        "ulcerative-colitis",
        "normal-z-line",
    ]

    original_classes = train_dataset.classes
    class_map = {cls: idx for idx, cls in enumerate(expected_classes)}

    if not all(cls in original_classes for cls in expected_classes):
        raise ValueError(
            f"Dataset classes {original_classes} do not contain all expected classes {expected_classes}"
        )

    train_images, train_labels = [], []
    print(f"Preloading train data for fold at {fold_path}")
    train_total = 6400
    for img, label in tqdm(train_dataset, desc="Preloading Train", total=train_total):
        cls_name = original_classes[label]
        if cls_name in expected_classes:
            img = data_transforms["train"](img)
            train_images.append(img)
            train_labels.append(label)

    test_images, test_labels = [], []
    print(f"Preloading test data for fold at {fold_path}")
    test_total = 1600
    for img, label in tqdm(test_dataset, desc="Preloading Test", total=test_total):
        cls_name = original_classes[label]
        if cls_name in expected_classes:
            img = data_transforms["test"](img)
            test_images.append(img)
            test_labels.append(label)

    train_dataset = PreloadedDataset(
        train_images, train_labels, class_map, original_classes
    )
    test_dataset = PreloadedDataset(
        test_images, test_labels, class_map, original_classes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    return (
        train_loader,
        test_loader,
        test_loader,
        expected_classes,
        (train_images, train_labels, test_images, test_labels),
    )
