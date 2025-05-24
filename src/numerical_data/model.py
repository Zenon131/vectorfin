"""
Numerical Data Module for VectorFin

This module transforms market metrics into meaningful vector representations
that capture their financial significance and can be used in the shared vector space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Technical analysis library
import ta


@dataclass
class NumericalFeatureConfig:
    """Configuration for numerical feature processing."""
    use_ma: bool = True  # Use moving averages
    ma_periods: List[int] = None  # Periods for moving averages
    use_rsi: bool = True  # Use Relative Strength Index
    use_macd: bool = True  # Use Moving Average Convergence Divergence
    use_bbands: bool = True  # Use Bollinger Bands
    use_volatility: bool = True  # Use volatility measures
    use_volume: bool = True  # Use volume features
    use_returns: bool = True  # Use returns (daily, weekly, monthly)
    normalize: bool = True  # Normalize features
    
    def __post_init__(self):
        # Default moving average periods if none provided
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 200]


class MarketDataProcessor:
    """
    Processes and prepares market data for the numerical vectorizer.
    
    This class handles preprocessing steps like normalization, calculating
    technical indicators, and creating temporal features from market data.
    """
    
    def __init__(self, config: NumericalFeatureConfig = None):
        """
        Initialize the market data processor.
        
        Args:
            config: Configuration for feature processing
        """
        self.config = config or NumericalFeatureConfig()
    
    def process_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process market data by calculating technical indicators and features.
        
        Args:
            df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
                Must have columns: 'open', 'high', 'low', 'close', 'volume'
                
        Returns:
            DataFrame with calculated features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Basic sanity check
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Calculate returns
        if self.config.use_returns:
            # Daily returns
            result['return_1d'] = df['close'].pct_change(1)
            # Weekly returns
            result['return_5d'] = df['close'].pct_change(5)
            # Monthly returns
            result['return_20d'] = df['close'].pct_change(20)
        
        # Calculate moving averages
        if self.config.use_ma:
            for period in self.config.ma_periods:
                result[f'ma_{period}'] = df['close'].rolling(window=period).mean()
                # Relative to current price
                result[f'ma_{period}_rel'] = df['close'] / result[f'ma_{period}'] - 1
        
        # Calculate RSI
        if self.config.use_rsi:
            for period in [14, 28]:  # Standard periods for RSI
                result[f'rsi_{period}'] = ta.momentum.RSIIndicator(
                    close=df['close'], window=period
                ).rsi()
        
        # Calculate MACD
        if self.config.use_macd:
            macd = ta.trend.MACD(close=df['close'])
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()
        
        # Calculate Bollinger Bands
        if self.config.use_bbands:
            bb = ta.volatility.BollingerBands(close=df['close'])
            result['bb_high'] = bb.bollinger_hband()
            result['bb_low'] = bb.bollinger_lband()
            result['bb_mid'] = bb.bollinger_mavg()
            # Position within bands
            result['bb_position'] = (df['close'] - result['bb_low']) / (result['bb_high'] - result['bb_low'])
        
        # Calculate volatility measures
        if self.config.use_volatility:
            # Daily price range relative to close
            result['daily_range'] = (df['high'] - df['low']) / df['close']
            # 20-day volatility
            result['volatility_20d'] = result['return_1d'].rolling(window=20).std()
        
        # Volume features
        if self.config.use_volume and 'volume' in df.columns:
            # Normalized volume
            result['volume_norm'] = df['volume'] / df['volume'].rolling(window=20).mean()
            # Volume change
            result['volume_change'] = df['volume'].pct_change(1)
            
            # Money flow indicators
            result['mfi_14'] = ta.volume.MFIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=14
            ).money_flow_index()
            
            # On-balance volume
            result['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            ).on_balance_volume()
        
        # Handle NaN values
        result = result.fillna(0)
        
        # Normalize features if configured
        if self.config.normalize:
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # Skip non-feature columns (like dates)
                if col not in ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    # Z-score normalization
                    mean = result[col].mean()
                    std = result[col].std()
                    if std > 0:  # Avoid division by zero
                        result[col] = (result[col] - mean) / std
                    
        return result


class NumericalVectorizer(nn.Module):
    """
    Transforms market metrics into vector representations.
    
    This module uses an autoencoder architecture to encode numerical market data
    into a latent space, and then projects it into the shared vector space.
    """
    
    def __init__(
        self,
        input_dim: int,
        vector_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        device: Optional[str] = None
    ):
        """
        Initialize the numerical vectorizer.
        
        Args:
            input_dim: Dimension of input features
            vector_dim: Dimension of output vectors
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        # Default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create encoder layers
        encoder_layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, hidden_dim))
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Final encoder layer to vector dimension
        encoder_layers.append(nn.Linear(current_dim, vector_dim))
        
        # Create encoder
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Create decoder for autoencoder reconstruction
        decoder_layers = []
        current_dim = vector_dim
        
        # Reverse the hidden dimensions for the decoder
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, hidden_dim))
            decoder_layers.append(nn.LayerNorm(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Final decoder layer to input dimension
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        
        # Create decoder
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store configuration
        self.input_dim = input_dim
        self.vector_dim = vector_dim
        
        # Move model to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode numeric features into vectors.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) with market features
            
        Returns:
            Tensor of shape (batch_size, vector_dim) with encoded vectors
        """
        # Encode inputs to latent space
        vectors = self.encoder(x)
        
        # Normalize vectors
        normalized_vectors = F.normalize(vectors, p=2, dim=1)
        
        return normalized_vectors
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from encoded representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        # Encode inputs
        encoded = self.encoder(x)
        
        # Decode back to input space
        reconstructed = self.decoder(encoded)
        
        return reconstructed
    
    def train_autoencoder(
        self,
        dataloader: torch.utils.data.DataLoader,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        weight_decay: float = 1e-5
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder.
        
        Args:
            dataloader: DataLoader with market data features
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            weight_decay: Weight decay for optimizer
            
        Returns:
            Dictionary of training history
        """
        # Create optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Training history
        history = {"loss": []}
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Process each batch
            for batch in dataloader:
                # Move to device
                batch = batch.to(self.device)
                
                # Forward pass - reconstruct inputs
                reconstructed = self.reconstruct(batch)
                
                # Calculate loss
                loss = criterion(reconstructed, batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                num_batches += 1
            
            # Record average epoch loss
            avg_epoch_loss = epoch_loss / num_batches
            history["loss"].append(avg_epoch_loss)
            
        return history


class MarketRegimeDetector:
    """
    Detects market regimes from numerical data.
    
    This class uses unsupervised learning to identify different market regimes,
    which can be used to provide context for the numerical vectorization.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize the market regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
        """
        self.n_regimes = n_regimes
        self.regime_model = None
        
    def fit(self, features: np.ndarray) -> None:
        """
        Fit the regime detection model to market data.
        
        Args:
            features: Array of market features
        """
        from sklearn.cluster import KMeans
        
        # Create and fit K-means model
        self.regime_model = KMeans(n_clusters=self.n_regimes, random_state=42)
        self.regime_model.fit(features)
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict market regimes for given features.
        
        Args:
            features: Array of market features
            
        Returns:
            Array of regime labels
        """
        if self.regime_model is None:
            raise ValueError("Model has not been fit yet.")
            
        return self.regime_model.predict(features)


# Example usage
if __name__ == "__main__":
    import torch.utils.data
    from sklearn.datasets import make_classification
    
    # Create synthetic market data
    X, _ = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Create vectorizer
    vectorizer = NumericalVectorizer(input_dim=20)
    
    # Train autoencoder
    history = vectorizer.train_autoencoder(dataloader, num_epochs=10)
    print(f"Final loss: {history['loss'][-1]}")
    
    # Get vector representations
    vectors = vectorizer(X)
    print(f"Numeric vectors shape: {vectors.shape}")