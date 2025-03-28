#!/usr/bin/env python
"""
Evaluation script for Flowers102 classification model.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.flowers102.config import Config
from src.flowers102.data.dataset import get_dataloaders
from src.flowers102.models.resnet import create_model, load_checkpoint
from src.flowers102.utils.logging_utils import setup_logging
from src.flowers102.utils.visualization import plot_confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate model on Flowers102 dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                      help='Dataset split to evaluate on')
    return parser.parse_args()


def evaluate(model, dataloader, device, criterion=None):
    """
    Evaluate model on dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        criterion: Loss function (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Calculate loss if criterion is provided
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            
            # Statistics
            batch_size = inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += batch_size
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = running_corrects / total_samples
    
    metrics = {
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
    }
    
    if criterion:
        metrics['loss'] = running_loss / total_samples
    
    return metrics


def main():
    """Main function to run evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config = Config()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    config.batch_size = args.batch_size
    
    # Get dataloader
    logger.info(f"Loading {args.split} dataset...")
    dataloaders = get_dataloaders(config)
    dataloader_to_evaluate = dataloaders[args.split]
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint_metadata = load_checkpoint(model, args.checkpoint, config.device)
    
    # Set up loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate model
    logger.info(f"Evaluating model on {args.split} set...")
    metrics = evaluate(model, dataloader_to_evaluate, config.device, criterion)
    
    # Log results
    logger.info(f"Evaluation results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    if 'loss' in metrics:
        logger.info(f"Loss: {metrics['loss']:.4f}")
    
    # Generate and save classification report
    report = classification_report(metrics['labels'], metrics['predictions'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = config.results_dir / f"{args.split}_classification_report.csv"
    report_df.to_csv(report_path)
    logger.info(f"Classification report saved to {report_path}")
    
    # Generate and save confusion matrix
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    cm_path = config.results_dir / f"{args.split}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names=range(config.num_classes), save_path=cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Save per-class accuracy
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    top_classes = np.argsort(-class_accuracy)
    
    logger.info("Top 5 best performing classes:")
    for i in top_classes[:5]:
        logger.info(f"Class {i}: {class_accuracy[i]:.4f}")
    
    logger.info("Top 5 worst performing classes:")
    for i in reversed(top_classes[-5:]):
        logger.info(f"Class {i}: {class_accuracy[i]:.4f}")


if __name__ == '__main__':
    main()