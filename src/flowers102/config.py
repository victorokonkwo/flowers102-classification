"""
Configuration module for Flowers102 classification project.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    """Configuration for the model training process."""

    # Project paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parents[2].absolute()
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).parents[2] / "data" / "flowers102"
    )
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parents[2] / "outputs"
    )

    # Derived paths
    checkpoint_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    # Dataset parameters
    batch_size: int = 32
    num_workers: int = 4

    # Model parameters
    num_classes: int = 102
    freeze_backbone: bool = True

    # Training parameters
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    num_epochs: int = 30
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    def __post_init__(self):
        """Initialize derived paths and create required directories."""
        # Set derived paths
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [self.data_dir, self.checkpoint_dir, self.logs_dir, self.results_dir]

        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

    def __str__(self):
        """String representation of the configuration."""
        return "\n".join(
            f"{key}: {value}"
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        )

    @classmethod
    def from_args(cls, args):
        """Create configuration from command-line arguments."""
        config = cls()

        # Update with command line arguments if provided
        if hasattr(args, "batch_size") and args.batch_size is not None:
            config.batch_size = args.batch_size

        if hasattr(args, "num_workers") and args.num_workers is not None:
            config.num_workers = args.num_workers

        if hasattr(args, "learning_rate") and args.learning_rate is not None:
            config.learning_rate = args.learning_rate
        elif hasattr(args, "lr") and args.lr is not None:
            config.learning_rate = args.lr

        if hasattr(args, "num_epochs") and args.num_epochs is not None:
            config.num_epochs = args.num_epochs
        elif hasattr(args, "epochs") and args.epochs is not None:
            config.num_epochs = args.epochs

        if hasattr(args, "freeze"):
            config.freeze_backbone = args.freeze

        # Only update paths if they are provided
        if hasattr(args, "data_dir") and args.data_dir is not None:
            config.data_dir = Path(args.data_dir)

        if hasattr(args, "output_dir") and args.output_dir is not None:
            config.output_dir = Path(args.output_dir)

        # Re-initialize derived paths
        config._create_directories()

        return config
