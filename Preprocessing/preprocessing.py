"""
Preprocessing utilities for STRATIX-Net
"""

from torchvision import transforms


def get_transforms(image_size=(224, 224)):
    """
    Get data transformation pipelines for training and testing

    Args:
        image_size: Tuple of (height, width) for image resize

    Returns:
        Dictionary containing 'train' and 'test' transforms
    """
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms
