#!/usr/bin/env python3
"""
Advanced fix for model loading issues with direction predictor.

This script rebuilds the old direction predictor to match the new model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vectorfin.src.prediction_interpretation.prediction import DirectionPredictionHead

def rebuild_direction_predictor():
    """Create a new direction predictor that's compatible with the old weights."""
    print("Rebuilding direction predictor model...")
    
    # Define paths
    model_path = "./trained_models/interpreter_direction.pt"
    backup_path = model_path + ".backup"
    temp_model_path = model_path + ".temp"
    
    # Make sure we have a backup
    if os.path.exists(model_path) and not os.path.exists(backup_path):
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"Created backup at {backup_path}")
    
    # Load the original model to get its structure
    try:
        old_state_dict = torch.load(backup_path, map_location='cpu')
        print("Loaded original state dict")
    except Exception as e:
        print(f"Error loading original state dict: {e}")
        return False
    
    # Check if this is a simple model with just weight and bias
    if 'weight' in old_state_dict and 'bias' in old_state_dict:
        input_dim = old_state_dict['weight'].shape[1]  # Get input dimension (128)
        output_dim = old_state_dict['weight'].shape[0]  # Get output dimension (1)
        
        print(f"Found simple model with input_dim={input_dim}, output_dim={output_dim}")
        
        # Create a custom model class that exactly matches the old model
        class SimpleDirectionHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(output_dim, input_dim))
                self.bias = nn.Parameter(torch.zeros(output_dim))
            
            def forward(self, x):
                return torch.sigmoid(F.linear(x, self.weight, self.bias))
        
        # Create an instance of the simple model
        simple_model = SimpleDirectionHead()
        
        # Set the weights directly
        simple_model.weight.data = old_state_dict['weight']
        simple_model.bias.data = old_state_dict['bias']
        
        # Save the model with same structure as original
        torch.save({
            'weight': simple_model.weight,
            'bias': simple_model.bias
        }, model_path)
        
        print(f"Rebuilt simple direction predictor model and saved to {model_path}")
        return True
    else:
        print("Original model doesn't have the expected simple structure")
        return False

if __name__ == "__main__":
    success = rebuild_direction_predictor()
    if success:
        print("Direction predictor model successfully rebuilt!")
    else:
        print("Failed to rebuild direction predictor model")
