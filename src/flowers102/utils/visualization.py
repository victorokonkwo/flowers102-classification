"""
Visualization utilities for the Flowers102 classification project.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from typing import List, Optional, Union


def plot_learning_curves(train_losses: List[float], val_losses: List[float], 
                         train_accs: List[float], val_accs: List[float],
                         save_path: Optional[Union[str, Path]] = None):
    """
    Plot and save the learning curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm, class_names, normalize=False, 
                          title='Confusion Matrix', 
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    Plot and save the confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Title of the plot
        cmap: Color map
        save_path: Path to save the plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Add axis labels
    num_classes = len(class_names)
    tick_marks = np.arange(num_classes)
    
    # If there are many classes, only show a subset
    if num_classes > 20:
        step = max(1, num_classes // 20)
        visible_class_names = [class_names[i] if i % step == 0 else '' for i in range(num_classes)]
        plt.xticks(tick_marks, visible_class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, visible_class_names)
    else:
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_dataset_samples(dataloader, class_names=None, n_samples=5, 
                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                             save_path=None):
    """
    Visualize random samples from a dataset.
    
    Args:
        dataloader: DataLoader containing the dataset
        class_names: List of class names
        n_samples: Number of samples to visualize
        mean: Mean values for denormalization
        std: Std values for denormalization
        save_path: Path to save the plot
    """
    # Get a batch of data
    images, labels = next(iter(dataloader))
    
    # Select a subset of images
    n_samples = min(n_samples, len(images))
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    # Create a figure
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    
    # Function to denormalize images
    def denormalize(tensor):
        tensor = tensor.clone().detach().numpy().transpose(1, 2, 0)
        tensor = tensor * np.array(std) + np.array(mean)
        tensor = np.clip(tensor, 0, 1)
        return tensor
    
    # Plot each image
    for i, idx in enumerate(indices):
        img = denormalize(images[idx])
        label_idx = labels[idx].item()
        
        if class_names is not None and label_idx < len(class_names):
            title = class_names[label_idx]
        else:
            title = f"Class {label_idx}"
        
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()