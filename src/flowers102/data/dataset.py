"""
Flowers102 dataset implementation.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from PIL import Image
import numpy as np
import scipy.io
import tarfile
import logging

from flowers102.data.transforms import get_transforms

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d:%H-%M-%S"
)
logger = logging.getLogger(__name__)


class Flowers102Dataset(Dataset):
    """
    Flowers102 Dataset Class

    This class handles the loading and preprocessing of the Flowers102 dataset.
    """

    # Dataset URLS
    DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    SEGMENTATION_URL = (
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz"
    )
    LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    SPLIT_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

    def __init__(
        self,
        data_dir: typing.Union[str, Path],
        split: str = "train",
        transform=None,
        download: bool = True,
    ):
        """
        Intialize the dataset.

        Args:
            data_dir: Directory to store the dataset
            split: Which split to use ('train', 'val', or 'test')
            transform: Transformations to apply to the images
            download: Whether to download the dataset if it's not present
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Download and extract the dataset if needed
        if download:
            self._download_and_extract_dataset()

        # Load image paths, labels and splits
        self.image_paths, self.labels = self._load_dataset_info()

        logger.info(f"Loaded {len(self.image_paths)} images for {split} split")

    def _download_and_extract_dataset(self):
        """Download and extract the dataset files if they don't exist."""
        # Check if the image folder exists
        if not (self.data_dir / "jpg").exists():
            # Download and extract images
            images_path = self.data_dir / "102flowers.tgz"
            if not images_path.exists():
                logger.info("Downloading dataset images...")
                download_url(self.DATASET_URL, str(self.data_dir), "102flowers.tgz")

            logger.info("Extracting dataset images...")
            with tarfile.open(images_path, "r:gz") as tar:
                tar.extractall(path=str(self.data_dir))

        # Download labels if they don't exist
        labels_path = self.data_dir / "imagelabels.mat"
        if not labels_path.exists():
            logger.info("Downloading image labels...")
            download_url(self.LABELS_URL, str(self.data_dir), "imagelabels.mat")

        # Download splits if they don't exist
        splits_path = self.data_dir / "setid.mat"
        if not splits_path.exists():
            logger.info("Downloading dataset splits...")
            download_url(self.SPLITS_URL, str(self.data_dir), "setid.mat")

    def _load_dataset_info(self) -> Tuple[List[Path], List[int]]:
        """
        Load image paths and labels for the specified split.

        Returns:
            Tuple of image paths and corresponding labels
        """
        # Load the labels
        labels_data = scipy.io.loadmat(str(self.data_dir / "setid.mat"))
        labels = labels_data["labels"][0].tolist()

        # Convert to 0-indexed labels
        labels = [label - 1 for label in labels]

        # Load the splits
        splits_data = scipy.io.loadmat(str(self.data_dir / "setid.mat"))

        if self.split == "train":
            indices = splits_data["trnid"][0].tolist()
        elif self.split == "val":
            indices = splits_data["valid"][0].tolist()
        elif self.split == "test":
            indices = splits_data["tstid"][0].tolist()
        else:
            raise ValueError(f"Invalid split: {self.split}")

        # Convert  to 0-indexed indicies
        indices = [idx - 1 for idx in indices]

        # Get the image paths and labels for this split
        image_dir = self.data_dir / "jpg"
        image_paths = []
        split_labels = []

        for idx in indices:
            # Images are named 'image_xxxxx.jpg'
            image_path = image_dir / f"image_{idx+1:05d}.jpg"
            if image_path.exists():
                image_paths.append(image_path)
                split_labels.append(labels[idx])
            else:
                logger.warning(f"Image {image_path} not found!")

        return image_paths, split_labels

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item to get

        Returns:
            Tuple of (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_dataloaders(config):
    """
    Create dataloaders for train, validation and test sets.

    Args:
        config: Configuration object with dataset parameters

    Returns:
        Dictionary containing train, val and test data loaders
    """
    # Get data transforms
    transforms = get_transforms()

    # Create datasets
    datasets = {
        split: Flowers102Dataset(
            config.data_dir, split=split, transform=transforms[split], download=True
        )
        for split in ["train", "val", "test"]
    }

    # Create data loaders
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=config.device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    return dataloaders
