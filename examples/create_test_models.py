#!/usr/bin/env python3
"""
Create synthetic models for testing VectorFin interactions.

This script creates placeholder model files in the trained_models directory
to enable testing of the model interaction scripts.
"""

import os
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem
from vectorfin.src.text_vectorization.model import FinancialTextVectorizer
from vectorfin.src.numerical_data.model import NumericalVectorizer
from vectorfin.src.alignment_integration.model import AlignmentIntegrationLayer
from vectorfin.src.prediction_interpretation.prediction import PredictionInterpreter


def create_test_models(output_dir="./trained_models", vector_dim=128, sentiment_dim=16, fusion_dim=128):
    """
    Create synthetic models for testing.
    
    Args:
        output_dir: Directory to save the models
        vector_dim: Dimension of vectors in the shared space
        sentiment_dim: Dimension of sentiment features
        fusion_dim: Dimension of fusion vectors
    """
    print(f"Creating test models in {output_dir}...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create text vectorizer
    text_vectorizer = FinancialTextVectorizer(
        pretrained_model_name="yiyanghkust/finbert-tone",  # Default model
        vector_dim=vector_dim,
        sentiment_dim=sentiment_dim,
        device="cpu"
    )
    torch.save(text_vectorizer.state_dict(), output_path / "text_vectorizer.pt")
    
    # Create numerical vectorizer
    num_vectorizer = NumericalVectorizer(
        input_dim=10,  # Example dimension for price, volume, etc.
        vector_dim=vector_dim,
        device="cpu"
    )
    torch.save(num_vectorizer.state_dict(), output_path / "num_vectorizer.pt")
    
    # Create alignment layer
    alignment = AlignmentIntegrationLayer(
        vector_dim=vector_dim,
        device="cpu"
    )
    # Save alignment layer directly with torch.save
    torch.save(alignment.state_dict(), output_path / "alignment.pt")
    
    # Create prediction interpreter files
    # The format is {path}_{prediction_type}.pt
    interpreter_base_path = str(output_path / "interpreter")
    
    # Create dummy interpreter files
    for pred_type in ["direction", "magnitude", "volatility", "timing"]:
        # Create a simple linear layer as a predictor
        predictor = torch.nn.Linear(fusion_dim, 1)
        torch.save(predictor.state_dict(), f"{interpreter_base_path}_{pred_type}.pt")
    
    print(f"Test models created successfully in {output_dir}!")


if __name__ == "__main__":
    create_test_models()
