#!/usr/bin/env python
"""
Inference script for Flowers102 classification model.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.flowers102.config import Config
from src.flowers102.models.resnet import create_model, load_checkpoint
from src.flowers102.data.transforms import get_transforms
from src.flowers102.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict flower class with trained model"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Show top-k predictions")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def load_image(image_path, transform=None):
    """
    Load and preprocess image.

    Args:
        image_path: Path to image file
        transform: Transformation to apply

    Returns:
        Preprocessed image tensor
    """
    image = Image.open(image_path).convert("RGB")

    if transform:
        image = transform(image)

    image = image.unsqueeze(0)

    return image


def predict(model, image, device, top_k=5):
    """
    Make prediction on image.

    Args:
        model: Trained model
        image: Preprocessed image tensor
        device: Device to run inference on
        top_k: Number of top predictions to return

    Returns:
        Tuple of (probabilities, classes)
    """
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)

        top_probs, top_classes = torch.topk(probabilities, top_k)

    return top_probs[0].cpu().numpy(), top_classes[0].cpu().numpy()


def main():
    """Main function to run prediction."""
    args = parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    config = Config()
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    transforms = get_transforms()["test"]

    logger.info(f"Loading image: {args.image}")
    image = load_image(args.image, transforms)

    logger.info("Creating model...")
    model = create_model(config)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, _ = load_checkpoint(model, args, checkpoint, config.device)

    logger.info("Making prediction...")
    probs, classes = predict(model, image, config.device, args.top_k)

    logger.info("Top predictions:")
    for i, (prob, class_idx) in enumerate(zip(probs, classes)):
        logger.info(f"{i+1}. Class {class_idx}: {prob:.4f}")

    original_image = Image.open(args.image).convert("RGB")
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(probs))
    plt.barh(y_pos, probs)
    plt.yticks(y_pos, [f"Class {cls}" for cls in classes])
    plt.title("Top Predictions")

    result_path = config.results_dir / "prediction_result.png"
    plt.tight_layout()
    plt.savefig(result_path)
    logger.info(f"Visualization saved to: {result_path}")


if __name__ == "__main__":
    main()
