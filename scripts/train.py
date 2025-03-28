#!/usr/bin/env python
"""
Training script for Flowers102 classification model.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the src directory to the path to ensure imports work correctly
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

try:
    from src.flowers102.config import Config
    from src.flowers102.data.dataset import get_dataloaders
    from src.flowers102.models.resnet import create_model
    from src.flowers102.training.trainer import Trainer
    from src.flowers102.utils.logging_utils import setup_logging
except ImportError:
    sys.path.insert(0, str(project_root / "src"))
    from flowers102.config import Config
    from flowers102.data.dataset import get_dataloaders
    from flowers102.models.resnet import create_model
    from flowers102.training.trainer import Trainer
    from flowers102.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ResNet50 on Flowers102 dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--freeze", action="store_true", help="Freeze backbone")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def main():
    """Main function to run the training."""
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize configuration
    config = Config.from_args(args)

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=config.logs_dir)

    # Get dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = get_dataloaders(config)

    # Create model
    logger.info("Creating model...")
    model = create_model(config)

    # Log model architecture
    logger.info(f"Model architecture:\n{model}")

    # Create trainer
    trainer = Trainer(model, dataloaders, config)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Close tensorboard writer
    writer.close()


if __name__ == "__main__":
    main()
