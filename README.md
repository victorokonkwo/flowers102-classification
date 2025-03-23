# Flowers102 Classification with ResNet50

This project implements a fine-tuned ResNet50 model for classifying the Oxford Flowers102 dataset. The project follows best practices for software engineering in machine learning projects.

## Project Overview

The [Flowers102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) consists of 102 flower categories commonly found in the United Kingdom. Each class contains between 40 and 258 images with significant variations in scale, pose, and lighting conditions.

## Features

- Fine-tuning of pre-trained ResNet50 model
- Data augmentation for improved generalization
- Learning rate scheduling
- Comprehensive evaluation metrics
- Model checkpointing
- Visualization tools
- Extensive logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flowers102-classification.git
cd flowers102-classification
```

2. Install dependencies:
```bash
pip install -e .
```

## Usage

### Training

Train the model with default parameters:
```bash
python scripts/train.py
```

Custom training configuration:
```bash
python scripts/train.py --epochs 50 --batch-size 64 --lr 0.0005 --freeze
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

### Inference

Make predictions on new images:
```bash
python scripts/predict.py --image path/to/flower.jpg --checkpoint outputs/checkpoints/best_model.pth
```

## Project Structure

flowers102-classification/
├── README.md                       # Project documentation
├── requirements.txt                # Dependencies
├── setup.py                        # Package installation script
├── .gitignore                      # Git ignore file
├── .flake8                         # Linting configuration
├── pyproject.toml                  # Python project metadata
├── src/                            # Source code directory
│   └── flowers102/                 # Main package
│       ├── __init__.py             # Package initialization
│       ├── config.py               # Configuration settings 
│       ├── data/                   # Data handling code
│       │   ├── __init__.py
│       │   ├── dataset.py          # Dataset class implementation
│       │   └── transforms.py       # Data transformations
│       ├── models/                 # Model implementations
│       │   ├── __init__.py
│       │   └── resnet.py           # ResNet50 model implementation
│       ├── training/               # Training utilities
│       │   ├── __init__.py
│       │   ├── trainer.py          # Trainer class
│       │   └── utils.py            # Training utilities
│       └── utils/                  # General utilities
│           ├── __init__.py
│           ├── logging_utils.py    # Logging configurations
│           └── visualization.py    # Visualization utilities
├── scripts/                        # Executable scripts
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── predict.py                  # Inference script
├── tests/                          # Test directory
│   ├── __init__.py
│   ├── conftest.py                 # Test configuration
│   ├── test_dataset.py             # Dataset tests
│   ├── test_model.py               # Model tests
│   └── test_training.py            # Training tests
├── notebooks/                      # Jupyter notebooks
│   └── exploratory_analysis.ipynb  # Data exploration
├── data/                           # Data directory (gitignored)
│   └── flowers102/                 # Flowers102 dataset
│       ├── jpg/                    # Images directory
│       ├── imagelabels.mat         # Image labels
│       └── setid.mat               # Train/val/test splits
└── outputs/                        # Output directory (gitignored)
    ├── checkpoints/                # Model checkpoints
    ├── logs/                       # Training logs
    └── results/                    # Results and metrics


## Results

The model achieves X% accuracy on the test set. Detailed performance metrics and visualizations can be found in the `outputs/results/` directory after training.

## License

This project is licensed under the MIT License - see the LICENSE file for details.