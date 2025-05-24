"""
Prediction and Interpretation Module for VectorFin

This module converts unified vectors into actionable market insights through
multiple prediction heads and explainability mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional


class MarketPredictionHead(nn.Module):
    """
    Base class for market prediction heads.
    
    This class provides common functionality for prediction heads
    that take unified vectors as input and produce market predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize the prediction head.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dims: Dimensions of hidden layers
            output_dim: Dimension of output predictions
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            
        # Output layer
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)
        
        # Store configuration
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions from input vectors.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        features = self.layers(x)
        predictions = self.output_layer(features)
        return predictions


class DirectionPredictionHead(MarketPredictionHead):
    """
    Prediction head for market direction (binary classification).
    
    This head predicts whether the market will go up or down.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize the direction prediction head.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        # Default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,  # Binary output
            dropout=dropout,
            device=device
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict market direction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, 1) with values between 0 and 1
            representing the probability of an upward movement
        """
        logits = super().forward(x)
        probs = torch.sigmoid(logits)
        return probs


class MagnitudePredictionHead(MarketPredictionHead):
    """
    Prediction head for price magnitude (regression).
    
    This head predicts the magnitude of price movements.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize the magnitude prediction head.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        # Default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,  # Regression output
            dropout=dropout,
            device=device
        )


class VolatilityPredictionHead(MarketPredictionHead):
    """
    Prediction head for market volatility (regression).
    
    This head predicts the expected volatility of price movements.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize the volatility prediction head.
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        # Default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,  # Regression output
            dropout=dropout,
            device=device
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict market volatility.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, 1) with positive values
        """
        # Ensure volatility predictions are positive
        return F.softplus(super().forward(x))


class TimingPredictionHead(MarketPredictionHead):
    """
    Prediction head for event timing (regression).
    
    This head predicts when a market event is likely to occur.
    """
    
    def __init__(
        self,
        input_dim: int,
        max_horizon: int = 30,  # Maximum prediction horizon in days
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize the timing prediction head.
        
        Args:
            input_dim: Dimension of input vectors
            max_horizon: Maximum prediction horizon in days
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        # Default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=max_horizon,  # Output for each day in horizon
            dropout=dropout,
            device=device
        )
        
        self.max_horizon = max_horizon
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict event timing probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, max_horizon) with values
            representing the probability of an event occurring on each day
        """
        logits = super().forward(x)
        probs = F.softmax(logits, dim=1)
        return probs


class AttentionExplainer:
    """
    Attention-based explainability system.
    
    This class provides methods to interpret and visualize which inputs
    most influenced the model's predictions.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the attention explainer.
        
        Args:
            model: The model to explain, should have attention mechanism
        """
        self.model = model
        self.attention_weights = {}
        
        # Register hooks to capture attention weights
        for name, module in model.named_modules():
            if "cross_attn" in name:
                module.register_forward_hook(self._capture_attention)
    
    def _capture_attention(self, module, input, output):
        """
        Hook function to capture attention weights.
        """
        # Assuming the first output element is text attends to num
        # and the second is num attends to text
        if isinstance(output, tuple) and len(output) == 2:
            self.attention_weights["text_to_num"] = module.text_num_attn.detach()
            self.attention_weights["num_to_text"] = module.num_text_attn.detach()
    
    def explain(
        self,
        text_inputs: List[str],
        num_features: List[str],
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor
    ) -> Dict:
        """
        Generate explanation for a prediction.
        
        Args:
            text_inputs: List of text inputs
            num_features: List of numerical feature names
            text_vectors: Text vectors of shape (batch_size, vector_dim)
            num_vectors: Numerical vectors of shape (batch_size, vector_dim)
            
        Returns:
            Dictionary of explanation data
        """
        # Clear previous attention weights
        self.attention_weights = {}
        
        # Forward pass to capture attention weights
        _ = self.model(text_vectors, num_vectors)
        
        explanations = {}
        
        # Process text to num attention
        if "text_to_num" in self.attention_weights:
            # Average across attention heads
            text_to_num_attn = self.attention_weights["text_to_num"].mean(dim=1)
            
            # Create explanation for each sample in batch
            for i in range(text_to_num_attn.shape[0]):
                if i < len(text_inputs):
                    text_input = text_inputs[i]
                    
                    # Get top numerical features for this text
                    num_attention = text_to_num_attn[i, 0, :].cpu().numpy()
                    top_indices = num_attention.argsort()[-5:][::-1]  # Top 5 features
                    
                    top_features = []
                    for idx in top_indices:
                        if idx < len(num_features):
                            top_features.append({
                                "feature": num_features[idx],
                                "weight": float(num_attention[idx])
                            })
                    
                    explanations[f"sample_{i}_text_attends_to"] = top_features
        
        # Process num to text attention
        if "num_to_text" in self.attention_weights:
            # Average across attention heads
            num_to_text_attn = self.attention_weights["num_to_text"].mean(dim=1)
            
            # Create explanation for numerical features
            for i in range(num_to_text_attn.shape[0]):
                # Get top text inputs for numerical features
                text_attention = num_to_text_attn[i, 0, :].cpu().numpy()
                top_indices = text_attention.argsort()[-5:][::-1]  # Top 5 texts
                
                top_texts = []
                for idx in top_indices:
                    if idx < len(text_inputs):
                        top_texts.append({
                            "text": text_inputs[idx],
                            "weight": float(text_attention[idx])
                        })
                
                explanations[f"sample_{i}_num_attends_to"] = top_texts
        
        return explanations
    
    def visualize_attention(
        self,
        text_inputs: List[str],
        num_features: List[str],
        sample_idx: int = 0
    ) -> plt.Figure:
        """
        Visualize attention weights.
        
        Args:
            text_inputs: List of text inputs
            num_features: List of numerical feature names
            sample_idx: Index of sample to visualize
            
        Returns:
            Matplotlib figure with attention visualization
        """
        if not self.attention_weights:
            raise ValueError("No attention weights captured. Run explain() first.")
            
        # Get attention weights for the sample
        if "text_to_num" in self.attention_weights:
            text_to_num_attn = self.attention_weights["text_to_num"][sample_idx].mean(dim=0)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Limit to available inputs and features
            max_text = min(len(text_inputs), text_to_num_attn.shape[0])
            max_num = min(len(num_features), text_to_num_attn.shape[1])
            
            # Plot heatmap
            sns.heatmap(
                text_to_num_attn[:max_text, :max_num].cpu().numpy(),
                annot=True,
                fmt=".2f",
                cmap="viridis",
                xticklabels=num_features[:max_num],
                yticklabels=text_inputs[:max_text],
                ax=ax
            )
            
            ax.set_title("Text to Numerical Feature Attention")
            ax.set_ylabel("Text Inputs")
            ax.set_xlabel("Numerical Features")
            
            plt.tight_layout()
            
            return fig
            
        return None


class PredictionInterpreter:
    """
    Main module for prediction and interpretation.
    
    This class combines multiple prediction heads and explainability
    systems to provide comprehensive market insights.
    """
    
    def __init__(
        self,
        vector_dim: int,
        direction_head: Optional[DirectionPredictionHead] = None,
        magnitude_head: Optional[MagnitudePredictionHead] = None,
        volatility_head: Optional[VolatilityPredictionHead] = None,
        timing_head: Optional[TimingPredictionHead] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the prediction interpreter.
        
        Args:
            vector_dim: Dimension of input vectors
            direction_head: Head for direction prediction
            magnitude_head: Head for magnitude prediction
            volatility_head: Head for volatility prediction
            timing_head: Head for timing prediction
            device: Device to use (cpu or cuda)
        """
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create prediction heads if not provided
        self.direction_head = direction_head or DirectionPredictionHead(vector_dim, device=self.device)
        self.magnitude_head = magnitude_head or MagnitudePredictionHead(vector_dim, device=self.device)
        self.volatility_head = volatility_head or VolatilityPredictionHead(vector_dim, device=self.device)
        self.timing_head = timing_head or TimingPredictionHead(vector_dim, device=self.device)
        
        # Store configuration
        self.vector_dim = vector_dim
    
    def predict_direction(self, unified_vectors: torch.Tensor) -> torch.Tensor:
        """
        Predict market direction.
        
        Args:
            unified_vectors: Unified vectors of shape (batch_size, vector_dim)
            
        Returns:
            Direction probabilities of shape (batch_size, 1)
        """
        return self.direction_head(unified_vectors)
    
    def predict_magnitude(self, unified_vectors: torch.Tensor) -> torch.Tensor:
        """
        Predict price movement magnitude.
        
        Args:
            unified_vectors: Unified vectors of shape (batch_size, vector_dim)
            
        Returns:
            Magnitude predictions of shape (batch_size, 1)
        """
        return self.magnitude_head(unified_vectors)
    
    def predict_volatility(self, unified_vectors: torch.Tensor) -> torch.Tensor:
        """
        Predict market volatility.
        
        Args:
            unified_vectors: Unified vectors of shape (batch_size, vector_dim)
            
        Returns:
            Volatility predictions of shape (batch_size, 1)
        """
        return self.volatility_head(unified_vectors)
    
    def predict_timing(self, unified_vectors: torch.Tensor) -> torch.Tensor:
        """
        Predict event timing.
        
        Args:
            unified_vectors: Unified vectors of shape (batch_size, vector_dim)
            
        Returns:
            Timing probabilities of shape (batch_size, max_horizon)
        """
        return self.timing_head(unified_vectors)
    
    def predict_all(self, unified_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make all predictions.
        
        Args:
            unified_vectors: Unified vectors of shape (batch_size, vector_dim)
            
        Returns:
            Dictionary of prediction results
        """
        return {
            "direction": self.predict_direction(unified_vectors),
            "magnitude": self.predict_magnitude(unified_vectors),
            "volatility": self.predict_volatility(unified_vectors),
            "timing": self.predict_timing(unified_vectors)
        }
    
    def interpret_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Interpret prediction results.
        
        Args:
            predictions: Dictionary of prediction results
            threshold: Threshold for direction prediction
            
        Returns:
            List of interpreted prediction results
        """
        batch_size = predictions["direction"].shape[0]
        interpretations = []
        
        for i in range(batch_size):
            # Extract predictions for this sample
            direction = predictions["direction"][i].item()
            magnitude = predictions["magnitude"][i].item()
            volatility = predictions["volatility"][i].item()
            timing = predictions["timing"][i].detach().cpu().numpy()
            
            # Interpret direction
            direction_label = "up" if direction > threshold else "down"
            
            # Find most likely timing
            most_likely_day = timing.argmax().item()
            timing_confidence = timing[most_likely_day].item()
            
            # Create interpretation
            interpretation = {
                "direction": {
                    "prediction": direction_label,
                    "probability": direction,
                    "confidence": abs(direction - 0.5) * 2  # Scale to 0-1
                },
                "magnitude": {
                    "prediction": magnitude,
                    "percentile": None  # Would need historical context
                },
                "volatility": {
                    "prediction": volatility,
                    "percentile": None  # Would need historical context
                },
                "timing": {
                    "prediction": most_likely_day,
                    "confidence": timing_confidence
                }
            }
            
            # Overall assessment
            if direction > 0.7:
                strength = "strong"
            elif direction > 0.6:
                strength = "moderate"
            else:
                strength = "weak"
                
            interpretation["summary"] = (
                f"{strength.capitalize()} {direction_label} signal with "
                f"expected magnitude of {magnitude:.2%} and "
                f"volatility of {volatility:.2%}, "
                f"most likely in {most_likely_day} days."
            )
            
            interpretations.append(interpretation)
            
        return interpretations
    
    def save(self, path: str) -> None:
        """
        Save all prediction heads.
        
        Args:
            path: Base path to save the models
        """
        torch.save(self.direction_head.state_dict(), f"{path}_direction.pt")
        torch.save(self.magnitude_head.state_dict(), f"{path}_magnitude.pt")
        torch.save(self.volatility_head.state_dict(), f"{path}_volatility.pt")
        torch.save(self.timing_head.state_dict(), f"{path}_timing.pt")
    
    @classmethod
    def load(cls, path: str, vector_dim: int, device: Optional[str] = None) -> 'PredictionInterpreter':
        """
        Load prediction heads from files.
        
        Args:
            path: Base path to load the models from
            vector_dim: Dimension of input vectors
            device: Device to use (cpu or cuda)
            
        Returns:
            Loaded PredictionInterpreter
        """
        # Create new instance
        interpreter = cls(vector_dim=vector_dim, device=device)
        
        # Load state dictionaries
        interpreter.direction_head.load_state_dict(
            torch.load(f"{path}_direction.pt", map_location=interpreter.device)
        )
        interpreter.magnitude_head.load_state_dict(
            torch.load(f"{path}_magnitude.pt", map_location=interpreter.device)
        )
        interpreter.volatility_head.load_state_dict(
            torch.load(f"{path}_volatility.pt", map_location=interpreter.device)
        )
        interpreter.timing_head.load_state_dict(
            torch.load(f"{path}_timing.pt", map_location=interpreter.device)
        )
        
        return interpreter


# Example usage
if __name__ == "__main__":
    # Create sample unified vectors
    unified_vectors = torch.randn(8, 128)
    unified_vectors = F.normalize(unified_vectors, p=2, dim=1)
    
    # Create prediction interpreter
    interpreter = PredictionInterpreter(vector_dim=128)
    
    # Make predictions
    predictions = interpreter.predict_all(unified_vectors)
    print(f"Direction shape: {predictions['direction'].shape}")
    print(f"Magnitude shape: {predictions['magnitude'].shape}")
    print(f"Volatility shape: {predictions['volatility'].shape}")
    print(f"Timing shape: {predictions['timing'].shape}")
    
    # Interpret predictions
    interpretations = interpreter.interpret_predictions(predictions)
    for i, interp in enumerate(interpretations[:2]):  # Show first 2
        print(f"\nSample {i}:")
        print(interp["summary"])
