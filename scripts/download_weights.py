#!/usr/bin/env python
"""
Script to download pre-trained model weights manually.
"""

import os
import sys
import ssl
import torch
import urllib.request
from pathlib import Path

# Create the checkpoints directory
checkpoint_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# ResNet50 weights URL
resnet50_url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
output_file = checkpoint_dir / "resnet50-19c8e357.pth"

print(f"Downloading ResNet50 weights to {output_file}...")

# Create an unverified context to bypass SSL certificate verification
ssl_context = ssl._create_unverified_context()

try:
    # Download with progress indicator
    with urllib.request.urlopen(resnet50_url, context=ssl_context) as response, open(output_file, 'wb') as out_file:
        content_length = int(response.info().get('Content-Length', 0))
        downloaded = 0
        block_size = 1024 * 8  # 8KB

        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            
            downloaded += len(buffer)
            out_file.write(buffer)
            
            # Print progress
            progress = downloaded / content_length * 100
            sys.stdout.write(f"\rDownloaded: {downloaded / (1024 * 1024):.1f}MB / {content_length / (1024 * 1024):.1f}MB ({progress:.1f}%)")
            sys.stdout.flush()
    
    print("\nDownload completed!")
    print(f"Weights saved to: {output_file}")
    
except Exception as e:
    print(f"Error downloading weights: {e}")
    sys.exit(1)