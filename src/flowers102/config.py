"""
Configuration module for Flower102 classification project.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    """Configuration for the model training process."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parents[2].abs())
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

    def __post_init(self):
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

        # Update with command-line arguments if provided
        if hasattr(args, "batch_size"):
            config.batch_size = args.batch_size
        if hasattr(args, "num_workers"):
            config.num_workers = args.num_workers
        if hasattr(args, "learning_rate") or hasattr(args, "lr"):
            config.learning_rate = getattr(
                args, "learning_rate", getattr(args, "lr", 0.001)
            )
        if hasattr(args, "num_epochs") or hasattr(args, "epochs"):
            config.num_epochs = getattr(args, "num_epochs", getattr(args, "epochs", 30))
        if hasattr(args, "freeze"):
            config.freeze_backbone = args.freeze
        if hasattr(args, "data_dir"):
            config.data_dir = Path(args.data_dir)

        # Re-initialized derived paths
        config._create_directories()

        return config
