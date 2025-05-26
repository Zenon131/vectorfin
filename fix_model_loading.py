#!/usr/bin/env python3
"""
Fix for model loading issues with direction predictor.

This script creates a compatibility layer for loading older model state dictionaries
into the new DirectionPredictionHead architecture.
"""

import torch
import torch.nn as nn
import os
from vectorfin.src.prediction_interpretation.prediction import DirectionPredictionHead

def convert_old_model_format(model_path):
    """
    Convert old model format to be compatible with the new DirectionPredictionHead.
    
    The old model format has 'weight' and 'bias' at the top level,
    while the new model expects layers for MLP structure.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Path to the new model file
    """
    print(f"Converting model: {model_path}")
    
    # Load the old state dict
    try:
        old_state_dict = torch.load(model_path, map_location='cpu')
        print("Loaded old state dict successfully")
    except Exception as e:
        print(f"Error loading old state dict: {e}")
        return model_path
    
    if 'weight' not in old_state_dict or 'bias' not in old_state_dict:
        print("Model doesn't seem to be in old format (missing 'weight' or 'bias')")
        return model_path
    
    # Get input dimension from weight matrix
    input_dim = old_state_dict['weight'].shape[1]  # 128 in our case
    output_dim = old_state_dict['weight'].shape[0]  # 1 in our case
    
    print(f"Found model with dimensions: input_dim={input_dim}, output_dim={output_dim}")
    
    # Create a custom DirectionPredictionHead with a fixed MLP structure
    # No hidden layers, just a direct input to output mapping
    new_model = DirectionPredictionHead(
        input_dim=input_dim, 
        hidden_dims=[]  # No hidden layers, direct input to output mapping
    )
    
    # Create a new state dict with the expected structure
    # The old model was likely a simple linear layer, so we'll map it to the output layer
    new_state_dict = {}
    new_state_dict['output_layer.weight'] = old_state_dict['weight']
    new_state_dict['output_layer.bias'] = old_state_dict['bias']
    
    # Initialize the rest of the layers with random values
    # We'll use the current values in the new model
    current_state_dict = new_model.state_dict()
    
    print("Current state dict keys:", list(current_state_dict.keys()))
    print("New state dict keys:", list(new_state_dict.keys()))
    
    # Copy existing weight to output_layer and initialize layers
    for key in current_state_dict:
        if key not in new_state_dict:
            new_state_dict[key] = current_state_dict[key]
    
    # Save the new state dict
    new_model_path = model_path.replace('.pt', '_converted.pt')
    torch.save(new_state_dict, new_model_path)
    print(f"Saved converted model to {new_model_path}")
    
    return new_model_path

def fix_direction_predictor():
    """Fix the direction predictor model."""
    # Define the path to the direction predictor model
    model_path = "./trained_models/interpreter_direction.pt"
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Restore from backup if it exists and try again
    backup_path = model_path + ".backup"
    if os.path.exists(backup_path):
        if os.path.exists(model_path):
            os.remove(model_path)
        os.rename(backup_path, model_path)
        print(f"Restored original model from {backup_path}")
    
    # Convert the model
    new_model_path = convert_old_model_format(model_path)
    
    # Backup the original model
    if not os.path.exists(backup_path) and os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"Backed up original model to {backup_path}")
    
    # Replace the original model with the converted one if it's not the same
    if new_model_path != model_path:
        if os.path.exists(new_model_path):
            os.replace(new_model_path, model_path)
            print(f"Replaced original model with converted model")
    
    print("Direction predictor model fixed!")

if __name__ == "__main__":
    fix_direction_predictor()
