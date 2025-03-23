"""
ResNet50 model implementation for Flowers102 classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict


class ResNet50Classifier(nn.Module):
    """ResNet50 model adapted for the Flowers102 dataset."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize the ResNet50 model.

        Args:
            num_classes: Number of classes in the dataset
            pretrained: Whether to use pre-trained weights 
        """
        super(ResNet50Classifier, self).__init__()

        # Load the pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=pretrained)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all layers except the final fully connected layer."""
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the final fully connected layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get the parameters that should be trained."""
        return [p for p in self.parameters() if p.requires_grad]


def create_model(config):
    """
    Create a ResNet50 model.

    Args:
        config: Configuration object

    Returns:
        Initialized model
    """
    model = ResNet50Classifier(num_classes=config.num_classes)

    # Freeze the backbone if specified 
    if config.freeze_backbone:
        model.freeze_backbone()

    # Move model to device
    model = model.to(config.device)

    return model


def load_checkpoint(model, checkpoint_path, device=None):
    """
    Load model weights from a checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to

    Returns:
        Model with loaded weights and metadata about the checkpoint 
    """
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both direct state dict and checkpoint dictionary
    if 'model_state_dict' in checkpoint:
        model._load_state_dict(checkpoint['model_state_dict'])
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    else:
        model._load_state_dict(checkpoint)
        metadata = {}
    
    return model, metadata
    