"""
VectorFin: A Multimodal Financial Analysis System

This module integrates all components of the VectorFin system to provide
holistic market insights by combining numerical market data and textual
sentiment analysis in a unified vector space.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path

# Import VectorFin components
from vectorfin.src.text_vectorization import FinancialTextVectorizer, FinancialTextProcessor
from vectorfin.src.numerical_data import NumericalVectorizer, MarketDataProcessor, NumericalFeatureConfig
from vectorfin.src.alignment_integration import AlignmentIntegrationLayer, VectorFusionModule, SemanticNavigator
from vectorfin.src.prediction_interpretation import PredictionInterpreter, AttentionExplainer
from vectorfin.src.data import FinancialTextData, MarketData, AlignedFinancialDataset


class VectorFinSystem:
    """
    Main class for the VectorFin system.
    
    This class integrates all components of the system and provides a high-level
    API for financial analysis using the unified vector space.
    """
    
    def __init__(
        self,
        vector_dim: int = 128,
        sentiment_dim: int = 16,
        fusion_dim: int = 128,
        device: Optional[str] = None,
        models_dir: Optional[str] = None
    ):
        """
        Initialize the VectorFin system.
        
        Args:
            vector_dim: Dimension of vectors in the shared space
            sentiment_dim: Dimension of sentiment features
            fusion_dim: Dimension of fused vectors
            device: Device to use (cpu or cuda)
            models_dir: Directory to load/save models
        """
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set models directory
        if models_dir:
            self.models_dir = Path(models_dir)
            self.models_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.models_dir = None
        
        # Initialize components
        self.text_vectorizer = FinancialTextVectorizer(
            vector_dim=vector_dim,
            sentiment_dim=sentiment_dim,
            device=self.device
        )
        
        self.num_processor = MarketDataProcessor(
            config=NumericalFeatureConfig()
        )
        
        # Numerical vectorizer will be initialized later when we know the input dimension
        self.num_vectorizer = None
        
        self.alignment = AlignmentIntegrationLayer(
            vector_dim=vector_dim,
            device=self.device
        )
        
        self.interpreter = PredictionInterpreter(
            vector_dim=fusion_dim,
            device=self.device
        )
        
        self.explainer = None  # Will be initialized when needed
        self.navigator = SemanticNavigator(vector_dim=fusion_dim)
        
        # Store configuration
        self.vector_dim = vector_dim
        self.sentiment_dim = sentiment_dim
        self.fusion_dim = fusion_dim
    
    def _ensure_num_vectorizer(self, input_dim: int) -> None:
        """
        Ensure the numerical vectorizer is initialized.
        
        Args:
            input_dim: Input dimension for the numerical vectorizer
        """
        if self.num_vectorizer is None:
            self.num_vectorizer = NumericalVectorizer(
                input_dim=input_dim,
                vector_dim=self.vector_dim,
                device=self.device
            )
    
    def process_text(self, texts: List[str]) -> torch.Tensor:
        """
        Process and vectorize text data.
        
        Args:
            texts: List of financial texts
            
        Returns:
            Tensor of text vectors
        """
        # Clean texts
        cleaned_texts = FinancialTextProcessor.batch_clean(texts)
        
        # Vectorize texts
        with torch.no_grad():
            vectors = self.text_vectorizer(cleaned_texts)
        
        return vectors
    
    def process_market_data(
        self,
        market_data: pd.DataFrame
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Process and vectorize market data.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Tuple of (numerical vectors, feature names)
        """
        # Process market data
        features_df = self.num_processor.process_market_data(market_data)
        
        # Get feature names and values
        feature_names = features_df.columns.tolist()
        feature_values = features_df.values
        
        # Ensure numerical vectorizer is initialized
        self._ensure_num_vectorizer(input_dim=feature_values.shape[1])
        
        # Convert to tensor
        features_tensor = torch.tensor(feature_values, dtype=torch.float32).to(self.device)
        
        # Vectorize market data
        with torch.no_grad():
            vectors = self.num_vectorizer(features_tensor)
        
        return vectors, feature_names
    
    def combine_vectors(
        self,
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine text and numerical vectors.
        
        Args:
            text_vectors: Tensor of text vectors
            num_vectors: Tensor of numerical vectors
            
        Returns:
            Tensor of unified vectors
        """
        with torch.no_grad():
            unified_vectors = self.alignment(text_vectors, num_vectors)
        
        return unified_vectors
    
    def predict(self, unified_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions from unified vectors.
        
        Args:
            unified_vectors: Tensor of unified vectors
            
        Returns:
            Dictionary of prediction results
        """
        with torch.no_grad():
            predictions = self.interpreter.predict_all(unified_vectors)
        
        return predictions
    
    def analyze_text_and_market(
        self,
        texts: List[str],
        market_data: pd.DataFrame
    ) -> Dict:
        """
        Analyze text and market data to generate insights.
        
        Args:
            texts: List of financial texts
            market_data: DataFrame with market data
            
        Returns:
            Dictionary of analysis results
        """
        # Process text data
        text_vectors = self.process_text(texts)
        
        # Process market data
        num_vectors, feature_names = self.process_market_data(market_data)
        
        # Ensure same batch size
        batch_size = min(text_vectors.shape[0], num_vectors.shape[0])
        text_vectors = text_vectors[:batch_size]
        num_vectors = num_vectors[:batch_size]
        
        # Combine vectors
        unified_vectors = self.combine_vectors(text_vectors, num_vectors)
        
        # Make predictions
        predictions = self.predict(unified_vectors)
        
        # Interpret predictions
        interpretations = self.interpreter.interpret_predictions(predictions)
        
        # Initialize explainer if needed
        if self.explainer is None:
            self.explainer = AttentionExplainer(self.alignment)
        
        # Generate explanations
        explanations = self.explainer.explain(
            texts[:batch_size],
            feature_names,
            text_vectors,
            num_vectors
        )
        
        # Create result dictionary
        results = {
            "predictions": predictions,
            "interpretations": interpretations,
            "explanations": explanations,
            "vectors": {
                "text": text_vectors.detach().cpu().numpy(),
                "numerical": num_vectors.detach().cpu().numpy(),
                "unified": unified_vectors.detach().cpu().numpy()
            }
        }
        
        return results
    
    def visualize_attention(
        self,
        texts: List[str],
        market_data: pd.DataFrame,
        sample_idx: int = 0
    ) -> plt.Figure:
        """
        Visualize attention between text and market data.
        
        Args:
            texts: List of financial texts
            market_data: DataFrame with market data
            sample_idx: Index of sample to visualize
            
        Returns:
            Matplotlib figure with attention visualization
        """
        # Process text and market data
        text_vectors = self.process_text(texts)
        num_vectors, feature_names = self.process_market_data(market_data)
        
        # Ensure same batch size
        batch_size = min(text_vectors.shape[0], num_vectors.shape[0])
        text_vectors = text_vectors[:batch_size]
        num_vectors = num_vectors[:batch_size]
        
        # Initialize explainer if needed
        if self.explainer is None:
            self.explainer = AttentionExplainer(self.alignment)
        
        # Forward pass to capture attention weights
        _ = self.alignment(text_vectors, num_vectors)
        
        # Visualize attention
        fig = self.explainer.visualize_attention(
            texts[:batch_size],
            feature_names,
            sample_idx
        )
        
        return fig
    
    def semantic_search(
        self,
        query_vector: torch.Tensor,
        index_texts: List[str] = None,
        index_market_data: pd.DataFrame = None,
        k: int = 10
    ) -> Dict:
        """
        Search for similar items in the vector space.
        
        Args:
            query_vector: Vector to search for
            index_texts: Texts to search in (optional)
            index_market_data: Market data to search in (optional)
            k: Number of results to return
            
        Returns:
            Dictionary of search results
        """
        # Build index if needed
        if not hasattr(self.navigator, 'index') or self.navigator.index is None:
            if index_texts is not None and index_market_data is not None:
                # Process text data
                text_vectors = self.process_text(index_texts)
                
                # Process market data
                num_vectors, _ = self.process_market_data(index_market_data)
                
                # Ensure same batch size
                batch_size = min(text_vectors.shape[0], num_vectors.shape[0])
                text_vectors = text_vectors[:batch_size]
                num_vectors = num_vectors[:batch_size]
                index_texts = index_texts[:batch_size]
                
                # Combine vectors
                unified_vectors = self.combine_vectors(text_vectors, num_vectors)
                
                # Create metadata
                metadata = [{"text": text} for text in index_texts]
                
                # Build index
                self.navigator.build_index(unified_vectors, metadata)
            else:
                raise ValueError("No index available. Provide index_texts and index_market_data.")
        
        # Search for similar items
        distances, indices, metadata = self.navigator.search(query_vector, k)
        
        # Create result dictionary
        results = {
            "distances": distances,
            "indices": indices,
            "metadata": metadata
        }
        
        return results
    
    def save_models(self, directory: Optional[str] = None) -> None:
        """
        Save all models to files.
        
        Args:
            directory: Directory to save models (defaults to self.models_dir)
        """
        # Use provided directory or default
        save_dir = Path(directory) if directory else self.models_dir
        
        if save_dir is None:
            raise ValueError("No models directory specified.")
        
        # Create directory if it doesn't exist
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save text vectorizer
        torch.save(
            self.text_vectorizer.state_dict(),
            save_dir / "text_vectorizer.pt"
        )
        
        # Save numerical vectorizer if initialized
        if self.num_vectorizer is not None:
            torch.save(
                self.num_vectorizer.state_dict(),
                save_dir / "num_vectorizer.pt"
            )
        
        # Save alignment layer
        self.alignment.save(str(save_dir / "alignment.pt"))
        
        # Save prediction interpreter
        self.interpreter.save(str(save_dir / "interpreter"))
        
        print(f"Models saved to {save_dir}")
    
    @classmethod
    def load_models(
        cls,
        directory: str,
        vector_dim: int = 128,
        sentiment_dim: int = 16,
        fusion_dim: int = 128,
        device: Optional[str] = None
    ) -> 'VectorFinSystem':
        """
        Load system from saved models.
        
        Args:
            directory: Directory containing saved models
            vector_dim: Dimension of vectors in the shared space
            sentiment_dim: Dimension of sentiment features
            fusion_dim: Dimension of fused vectors
            device: Device to use (cpu or cuda)
            
        Returns:
            Loaded VectorFinSystem
        """
        # Create new instance
        system = cls(
            vector_dim=vector_dim,
            sentiment_dim=sentiment_dim,
            fusion_dim=fusion_dim,
            device=device,
            models_dir=directory
        )
        
        # Load models
        models_dir = Path(directory)
        
        # Load text vectorizer
        text_vectorizer_path = models_dir / "text_vectorizer.pt"
        if text_vectorizer_path.exists():
            system.text_vectorizer.load_state_dict(
                torch.load(text_vectorizer_path, map_location=system.device)
            )
        
        # Load numerical vectorizer if available
        num_vectorizer_path = models_dir / "num_vectorizer.pt"
        if num_vectorizer_path.exists():
            # Initialize with a temporary dimension, will be updated when used
            system.num_vectorizer = NumericalVectorizer(
                input_dim=10,  # Temporary
                vector_dim=vector_dim,
                device=system.device
            )
            system.num_vectorizer.load_state_dict(
                torch.load(num_vectorizer_path, map_location=system.device)
            )
        
        # Load alignment layer
        alignment_path = models_dir / "alignment.pt"
        if alignment_path.exists():
            # Create a new alignment layer
            system.alignment = AlignmentIntegrationLayer(
                vector_dim=vector_dim,
                device=system.device
            )
            # Load the saved state dict
            system.alignment.load_state_dict(
                torch.load(alignment_path, map_location=system.device)
            )
        
        # Load prediction interpreter
        interpreter_base_path = str(models_dir / "interpreter")
        
        # Create a new interpreter instance
        system.interpreter = PredictionInterpreter(
            vector_dim=fusion_dim,
            device=system.device
        )
        
        # Try to load each predictor file
        for pred_type in ["direction", "magnitude", "volatility", "timing"]:
            try:
                pred_path = f"{interpreter_base_path}_{pred_type}.pt"
                pred_attr = f"{pred_type}_head"
                if hasattr(system.interpreter, pred_attr) and os.path.exists(pred_path):
                    # Load model state
                    state_dict = torch.load(pred_path, map_location=system.device)
                    
                    # Check if this is a simple linear layer (just weight and bias)
                    if 'weight' in state_dict and 'bias' in state_dict and len(state_dict) == 2:
                        # Create a simple linear layer that matches the saved weights
                        pred_head = nn.Linear(
                            state_dict['weight'].shape[1], 
                            state_dict['weight'].shape[0]
                        ).to(system.device)
                        
                        # Load the weights
                        pred_head.load_state_dict(state_dict)
                        
                        # Replace the existing head with our compatible one
                        setattr(system.interpreter, pred_attr, pred_head)
                        print(f"Successfully loaded {pred_type} predictor with simplified model structure")
                    else:
                        # Try original approach
                        getattr(system.interpreter, pred_attr).load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load {pred_type} predictor: {e}")
        
        print(f"Models loaded from {directory}")
        return system


class VectorFinTrainer:
    """
    Trainer for the VectorFin system.
    
    This class provides methods for training the different components
    of the VectorFin system using financial data.
    """
    
    def __init__(
        self,
        system: VectorFinSystem,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize the trainer.
        
        Args:
            system: VectorFinSystem to train
            learning_rate: Learning rate for optimizers
            weight_decay: Weight decay for optimizers
        """
        self.system = system
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizers
        self.text_optimizer = torch.optim.AdamW(
            self.system.text_vectorizer.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Numerical vectorizer optimizer will be initialized when needed
        self.num_optimizer = None
        
        self.alignment_optimizer = torch.optim.AdamW(
            self.system.alignment.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create optimizers for prediction heads
        self.prediction_optimizers = {
            "direction": torch.optim.AdamW(
                self.system.interpreter.direction_head.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            ),
            "magnitude": torch.optim.AdamW(
                self.system.interpreter.magnitude_head.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            ),
            "volatility": torch.optim.AdamW(
                self.system.interpreter.volatility_head.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            ),
            "timing": torch.optim.AdamW(
                self.system.interpreter.timing_head.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        }
        
        # Create loss functions
        self.direction_loss_fn = nn.BCELoss()
        self.magnitude_loss_fn = nn.MSELoss()
        self.volatility_loss_fn = nn.MSELoss()
        self.timing_loss_fn = nn.CrossEntropyLoss()
        
        # For contrastive learning
        self.contrastive_loss_fn = None  # Will be initialized when needed
        
    def _ensure_num_optimizer(self) -> None:
        """
        Ensure the numerical vectorizer optimizer is initialized.
        """
        if self.num_optimizer is None and self.system.num_vectorizer is not None:
            self.num_optimizer = torch.optim.AdamW(
                self.system.num_vectorizer.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
    
    def train_text_vectorizer(
        self,
        texts: List[str],
        labels: List[int],
        num_epochs: int = 3,
        batch_size: int = 16
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the text vectorizer with labeled sentiment data.
        
        Args:
            texts: List of financial texts
            labels: Sentiment labels (0: negative, 1: neutral, 2: positive)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training history
        """
        return self.system.text_vectorizer.fine_tune(
            texts=texts,
            labels=labels,
            learning_rate=self.learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
    
    def train_num_vectorizer(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the numerical vectorizer.
        
        Args:
            dataloader: DataLoader with market data features
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary of training history
        """
        self._ensure_num_optimizer()
        
        return self.system.num_vectorizer.train_autoencoder(
            dataloader=dataloader,
            learning_rate=self.learning_rate,
            num_epochs=num_epochs,
            weight_decay=self.weight_decay
        )
    
    def train_alignment_layer(
        self,
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor,
        num_epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the alignment layer with contrastive learning.
        
        Args:
            text_vectors: Tensor of text vectors
            num_vectors: Tensor of numerical vectors
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training history
        """
        # Initialize contrastive loss if needed
        if self.contrastive_loss_fn is None:
            from vectorfin.src.alignment_integration import ContrastiveLoss
            self.contrastive_loss_fn = ContrastiveLoss(margin=0.5, temperature=0.07)
        
        # Training history
        history = {"loss": []}
        
        # Set model to training mode
        self.system.alignment.train()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Process in batches
            for i in range(0, len(text_vectors), batch_size):
                end_idx = min(i + batch_size, len(text_vectors))
                
                batch_text_vectors = text_vectors[i:end_idx]
                batch_num_vectors = num_vectors[i:end_idx]
                
                # Forward pass through alignment layer
                unified_vectors = self.system.alignment(batch_text_vectors, batch_num_vectors)
                
                # Calculate contrastive loss
                loss = self.contrastive_loss_fn(batch_text_vectors, batch_num_vectors)
                
                # Backward pass and optimize
                self.alignment_optimizer.zero_grad()
                loss.backward()
                self.alignment_optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
            
            # Record average epoch loss
            avg_epoch_loss = epoch_loss / (len(text_vectors) / batch_size)
            history["loss"].append(avg_epoch_loss)
            
        # Set model back to evaluation mode
        self.system.alignment.eval()
        
        return history
    
    def train_prediction_heads(
        self,
        unified_vectors: torch.Tensor,
        labels: Dict[str, torch.Tensor],
        num_epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the prediction heads.
        
        Args:
            unified_vectors: Tensor of unified vectors
            labels: Dictionary of label tensors for each prediction head
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training history
        """
        # Training history
        history = {
            "direction_loss": [],
            "magnitude_loss": [],
            "volatility_loss": [],
            "timing_loss": []
        }
        
        # Set models to training mode
        self.system.interpreter.direction_head.train()
        self.system.interpreter.magnitude_head.train()
        self.system.interpreter.volatility_head.train()
        self.system.interpreter.timing_head.train()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_losses = {key: 0.0 for key in history.keys()}
            
            # Process in batches
            for i in range(0, len(unified_vectors), batch_size):
                end_idx = min(i + batch_size, len(unified_vectors))
                
                batch_vectors = unified_vectors[i:end_idx]
                
                # Direction prediction
                if "direction" in labels:
                    batch_direction = labels["direction"][i:end_idx]
                    direction_pred = self.system.interpreter.predict_direction(batch_vectors)
                    direction_loss = self.direction_loss_fn(direction_pred, batch_direction)
                    
                    self.prediction_optimizers["direction"].zero_grad()
                    direction_loss.backward(retain_graph=True)
                    self.prediction_optimizers["direction"].step()
                    
                    epoch_losses["direction_loss"] += direction_loss.item()
                
                # Magnitude prediction
                if "magnitude" in labels:
                    batch_magnitude = labels["magnitude"][i:end_idx]
                    magnitude_pred = self.system.interpreter.predict_magnitude(batch_vectors)
                    magnitude_loss = self.magnitude_loss_fn(magnitude_pred, batch_magnitude)
                    
                    self.prediction_optimizers["magnitude"].zero_grad()
                    magnitude_loss.backward(retain_graph=True)
                    self.prediction_optimizers["magnitude"].step()
                    
                    epoch_losses["magnitude_loss"] += magnitude_loss.item()
                
                # Volatility prediction
                if "volatility" in labels:
                    batch_volatility = labels["volatility"][i:end_idx]
                    volatility_pred = self.system.interpreter.predict_volatility(batch_vectors)
                    volatility_loss = self.volatility_loss_fn(volatility_pred, batch_volatility)
                    
                    self.prediction_optimizers["volatility"].zero_grad()
                    volatility_loss.backward(retain_graph=True)
                    self.prediction_optimizers["volatility"].step()
                    
                    epoch_losses["volatility_loss"] += volatility_loss.item()
                
                # Timing prediction
                if "timing" in labels:
                    batch_timing = labels["timing"][i:end_idx]
                    timing_pred = self.system.interpreter.predict_timing(batch_vectors)
                    timing_loss = self.timing_loss_fn(timing_pred, batch_timing)
                    
                    self.prediction_optimizers["timing"].zero_grad()
                    timing_loss.backward()
                    self.prediction_optimizers["timing"].step()
                    
                    epoch_losses["timing_loss"] += timing_loss.item()
            
            # Record average epoch losses
            num_batches = (len(unified_vectors) - 1) // batch_size + 1
            for key in epoch_losses:
                if epoch_losses[key] > 0:  # Only record if loss was calculated
                    history[key].append(epoch_losses[key] / num_batches)
            
        # Set models back to evaluation mode
        self.system.interpreter.direction_head.eval()
        self.system.interpreter.magnitude_head.eval()
        self.system.interpreter.volatility_head.eval()
        self.system.interpreter.timing_head.eval()
        
        return history
    
    def train_end_to_end(
        self,
        dataset: AlignedFinancialDataset,
        num_epochs: int = 5,
        batch_size: int = 16,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the entire system end-to-end.
        
        Args:
            dataset: Dataset with aligned financial text and market data
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            save_dir: Directory to save models (optional)
            
        Returns:
            Dictionary of training history
        """
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Training history
        history = {
            "total_loss": [],
            "text_loss": [],
            "num_loss": [],
            "alignment_loss": [],
            "prediction_loss": []
        }
        
        # Set models to training mode
        self.system.text_vectorizer.train()
        if self.system.num_vectorizer:
            self.system.num_vectorizer.train()
        self.system.alignment.train()
        self.system.interpreter.direction_head.train()
        self.system.interpreter.magnitude_head.train()
        self.system.interpreter.volatility_head.train()
        self.system.interpreter.timing_head.train()
        
        # Initialize contrastive loss if needed
        if self.contrastive_loss_fn is None:
            from vectorfin.src.alignment_integration import ContrastiveLoss
            self.contrastive_loss_fn = ContrastiveLoss(margin=0.5, temperature=0.07)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_losses = {key: 0.0 for key in history.keys()}
            
            for batch in dataloader:
                # Process texts
                texts = batch["texts"]
                cleaned_texts = FinancialTextProcessor.batch_clean(texts)
                text_vectors = self.system.text_vectorizer(cleaned_texts)
                
                # Process market data
                market_data = torch.tensor(
                    batch["market_data"], dtype=torch.float32
                ).to(self.system.device)
                
                # Ensure numerical vectorizer is initialized
                self._ensure_num_optimizer()
                num_vectors = self.system.num_vectorizer(market_data)
                
                # Combine vectors
                unified_vectors = self.system.alignment(text_vectors, num_vectors)
                
                # Make predictions
                direction_pred = self.system.interpreter.predict_direction(unified_vectors)
                magnitude_pred = self.system.interpreter.predict_magnitude(unified_vectors)
                volatility_pred = self.system.interpreter.predict_volatility(unified_vectors)
                
                # Calculate losses
                # For simplicity, assuming labels are available in the batch
                # In a real implementation, you would extract these from the target data
                
                # Text reconstruction loss
                text_loss = 0  # Placeholder
                
                # Numerical reconstruction loss
                reconstructed_num = self.system.num_vectorizer.reconstruct(market_data)
                num_loss = nn.MSELoss()(reconstructed_num, market_data)
                
                # Alignment loss
                alignment_loss = self.contrastive_loss_fn(text_vectors, num_vectors)
                
                # Prediction losses - placeholders
                # In a real implementation, you would use actual labels
                direction_loss = 0
                magnitude_loss = 0
                volatility_loss = 0
                prediction_loss = direction_loss + magnitude_loss + volatility_loss
                
                # Total loss
                total_loss = text_loss + num_loss + alignment_loss + prediction_loss
                
                # Backward pass and optimize
                self.text_optimizer.zero_grad()
                self.num_optimizer.zero_grad()
                self.alignment_optimizer.zero_grad()
                for optimizer in self.prediction_optimizers.values():
                    optimizer.zero_grad()
                
                total_loss.backward()
                
                self.text_optimizer.step()
                self.num_optimizer.step()
                self.alignment_optimizer.step()
                for optimizer in self.prediction_optimizers.values():
                    optimizer.step()
                
                # Update statistics
                epoch_losses["total_loss"] += total_loss.item()
                epoch_losses["text_loss"] += text_loss if isinstance(text_loss, float) else text_loss.item() if hasattr(text_loss, 'item') else 0
                epoch_losses["num_loss"] += num_loss.item()
                epoch_losses["alignment_loss"] += alignment_loss.item()
                epoch_losses["prediction_loss"] += prediction_loss if isinstance(prediction_loss, float) else prediction_loss.item() if hasattr(prediction_loss, 'item') else 0
            
            # Record average epoch losses
            num_batches = len(dataloader)
            for key in epoch_losses:
                history[key].append(epoch_losses[key] / num_batches)
                
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {history['total_loss'][-1]:.4f}")
            
            # Save models if directory is provided
            if save_dir and (epoch + 1) % 5 == 0:  # Save every 5 epochs
                self.system.save_models(save_dir)
        
        # Set models back to evaluation mode
        self.system.text_vectorizer.eval()
        if self.system.num_vectorizer:
            self.system.num_vectorizer.eval()
        self.system.alignment.eval()
        self.system.interpreter.direction_head.eval()
        self.system.interpreter.magnitude_head.eval()
        self.system.interpreter.volatility_head.eval()
        self.system.interpreter.timing_head.eval()
        
        # Save final models if directory is provided
        if save_dir:
            self.system.save_models(save_dir)
        
        return history


# Example usage
if __name__ == "__main__":
    # Create VectorFin system
    system = VectorFinSystem(vector_dim=128)
    
    # Example text and market data
    texts = [
        "The company reported better than expected earnings, raising their guidance for the next quarter.",
        "The stock plummeted after the CEO announced his resignation amid fraud allegations."
    ]
    
    # Create sample market data
    market_data = pd.DataFrame({
        'open': [150.0, 152.0],
        'high': [155.0, 153.0],
        'low': [149.0, 145.0],
        'close': [153.0, 146.0],
        'volume': [1000000, 1500000]
    })
    
    # Analyze text and market data
    analysis = system.analyze_text_and_market(texts, market_data)
    
    # Print summary
    for i, interp in enumerate(analysis["interpretations"]):
        print(f"\nSample {i}:")
        print(interp["summary"])
