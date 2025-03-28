"""
Trainer class for model training and evaluation.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from src.flowers102.utils.visualization import (
    plot_learning_curves,
    plot_confusion_matrix,
)
from src.flowers102.models.resnet import load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """Class to handle the training, validation, and testing of the model."""

    def __init__(self, model, dataloaders, config):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            dataloaders: Dictionary of dataloaders for train, val, test
            config: Configuration object
        """
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.device = config.device

        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()

        # Set up optimizer
        self.optimizer = optim.SGD(
            model.get_trainable_parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        # Set up learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )

        # Initialize metrics tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        logger.info(f"Trainer initialized with device: {self.device}")

    def train(self):
        """Train the model for the specified number of epochs."""
        logger.info("Starting training...")
        logger.info(f"Configuration:\n{self.config}")

        # Track the start time
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")

            # Unfreeze the backbone after 10 epochs if it was frozen
            if epoch == 10 and self.config.freeze_backbone:
                logger.info("Unfreezing model backbone...")
                self.model.unfreeze_backbone()

                # Update the optimizer with all trainable parameters
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config.learning_rate
                    / 10,  # Lower learning rate for fine-tuning
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                )

                # Re-initialize the scheduler
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.1, patience=5, verbose=True
                )

            # Train for one epoch
            train_loss, train_acc = self._train_one_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self._validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Update the learning rate
            self.scheduler.step(val_loss)

            # Save the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, is_best=True)
                logger.info(
                    f"New best model saved with validation accuracy: {val_acc:.4f}"
                )

            # Save regular checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, val_acc)

        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes")

        # Plot and save the learning curves
        plot_learning_curves(
            self.train_losses,
            self.val_losses,
            self.train_accs,
            self.val_accs,
            save_path=self.config.results_dir / "learning_curves.png",
        )

        # Test the model
        self.test()

    def _train_one_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # Use tqdm for progress bar
        pbar = tqdm(self.dataloaders["train"], desc="Training")

        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += batch_size

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": torch.sum(preds == labels.data).item() / batch_size,
                }
            )

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        logger.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        return epoch_loss, epoch_acc

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders["val"], desc="Validating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        logger.info(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        return epoch_loss, epoch_acc

    def test(self) -> Tuple[float, float]:
        """
        Test the model on the test set.

        Returns:
            Tuple of (average loss, accuracy)
        """
        logger.info("Testing the model...")

        # Load the best model
        best_model_path = self.config.checkpoint_dir / "best_model.pth"
        if best_model_path.exists():
            self.model, checkpoint_metadata = load_checkpoint(
                self.model, best_model_path, self.device
            )
            logger.info(
                f"Loaded best model with validation accuracy: {checkpoint_metadata.get('accuracy', 'N/A')}"
            )

        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        # For confusion matrix and classification report
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders["test"], desc="Testing"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += batch_size

                # Store predictions and labels for confusion matrix
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = running_loss / total_samples
        test_acc = running_corrects / total_samples

        logger.info(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        # Generate and save classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        import pandas as pd

        report_df = pd.DataFrame(report).transpose()
        report_path = self.config.results_dir / "classification_report.csv"
        report_df.to_csv(report_path)
        logger.info(f"Classification report saved to {report_path}")

        # Generate and save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = self.config.results_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            cm, class_names=range(self.config.num_classes), save_path=cm_path
        )
        logger.info(f"Confusion matrix saved to {cm_path}")

        return test_loss, test_acc

    def _save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            accuracy: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accuracy": accuracy,
        }

        if is_best:
            torch.save(checkpoint, self.config.checkpoint_dir / "best_model.pth")

        torch.save(
            checkpoint, self.config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        )
