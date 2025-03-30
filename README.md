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
git clone https://github.com/victorokonkwo/flowers102-classification.git
cd flowers102-classification
```

2. Install dependencies:
```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
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

```
flowers102-classification/
├── README.md                   
├── requirements.txt              
├── setup.py                    
├── .gitignore                     
├── .flake8                       
├── pyproject.toml              
├── src/                           
│   └── flowers102/              
│       ├── __init__.py          
│       ├── config.py             
│       ├── data/               
│       │   ├── __init__.py
│       │   ├── dataset.py       
│       │   └── transforms.py    
│       ├── models/                
│       │   ├── __init__.py
│       │   └── resnet.py          
│       ├── training/          
│       │   ├── __init__.py
│       │   ├── trainer.py      
│       │   └── utils.py           
│       └── utils/              
│           ├── __init__.py
│           ├── logging_utils.py   
│           └── visualization.py  
├── scripts/                       
│   ├── train.py                
│   ├── evaluate.py            
│   └── predict.py          
├── tests/                     
│   ├── __init__.py
│   ├── conftest.py               
│   ├── test_dataset.py         
│   ├── test_model.py         
│   └── test_training.py         
├── notebooks/                      
│   └── exploratory_analysis.ipynb 
├── data/                          
│   └── flowers102/                
│       ├── jpg/                 
│       ├── imagelabels.mat     
│       └── setid.mat             
└── outputs/                       
    ├── checkpoints/          
    ├── logs/                     
    └── results/                  
```

## Results

The model achieves **95%** accuracy on the test set. Detailed performance metrics and visualizations can be found in the `outputs/results/` directory after training.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
