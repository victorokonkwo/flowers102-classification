"""
Data transformations for the Flowers102 dataset.
"""

from torchvision import transforms


def get_transforms():
    """
    Get data transforms for train, validation, and test sets.

    Returns:
        Dictionary of transforms for each split
    """
    # Normalization stats for ImageNet pre-trained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Train transforms with data augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Validation and test transforms
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}
