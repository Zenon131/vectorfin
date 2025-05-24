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
            config: Configuration for feature engineering
        """
        self.config = config or NumericalFeatureConfig()
    
    def process(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process market data by calculating technical indicators and other features.
        
        Args:
            market_data: DataFrame with market data (must contain OHLCV columns)
            
        Returns:
            DataFrame with processed features
        """
        # Verify that dataframe has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in market_data.columns:
                raise ValueError(f"Market data must contain '{col}' column")
        
        # Make a copy to avoid modifying the original DataFrame
        df = market_data.copy()
        
        # Calculate and add features based on the configuration
        processed_df = self._add_features(df)
        
        # Handle missing values (NaN) that may be introduced by indicators
        processed_df = processed_df.fillna(method='ffill').fillna(0)
        
        # Normalize features if configured to do so
        if self.config.normalize:
            processed_df = self._normalize_features(processed_df)
        
        return processed_df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and other features to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        # Moving Averages
        if self.config.use_ma:
            for period in self.config.ma_periods:
                df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
                # Distance from MA as percentage
                df[f'ma_{period}_dist'] = (df['close'] - df[f'ma_{period}']) / df[f'ma_{period}']
        
        # RSI (Relative Strength Index)
        if self.config.use_rsi:
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        
        # MACD (Moving Average Convergence Divergence)
        if self.config.use_macd:
            macd_indicator = ta.trend.MACD(df['close'])
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_diff'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        if self.config.use_bbands:
            bb_indicator = ta.volatility.BollingerBands(df['close'])
            df['bb_high'] = bb_indicator.bollinger_hband()
            df['bb_low'] = bb_indicator.bollinger_lband()
            df['bb_mid'] = bb_indicator.bollinger_mavg()
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            df['bb_pct'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Volatility
        if self.config.use_volatility:
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['daily_range'] = (df['high'] - df['low']) / df['close']
            df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
        
        # Volume features
        if self.config.use_volume:
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['money_flow'] = df['volume'] * (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
        
        # Returns
        if self.config.use_returns:
            df['return_1d'] = df['close'].pct_change(1)
            df['return_5d'] = df['close'].pct_change(5)
            df['return_20d'] = df['close'].pct_change(20)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        # Columns to exclude from normalization
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date']
        
        # Columns to normalize
        norm_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create a copy
        norm_df = df.copy()
        
        # Z-score normalization for each feature
        for col in norm_cols:
            mean = norm_df[col].mean()
            std = norm_df[col].std()
            if std > 0:  # Avoid division by zero
                norm_df[col] = (norm_df[col] - mean) / std
            else:
                norm_df[col] = 0  # If std is 0, feature has no variance
        
        return norm_df


class NumericalVectorizer(nn.Module):
    """
    Transforms market metrics into meaningful vector representations.
    
    This module takes processed market data features and transforms them into
    vector representations that capture their financial significance and can be
    used in the shared vector space with text vectors.
    """
    
    def __init__(
        self,
        input_dim: int,
        vector_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the numerical vectorizer.
        
        Args:
            input_dim: Dimension of the input features
            vector_dim: Dimension of the output vector
            hidden_dims: List of hidden dimensions for the network
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            device: The device to use (cpu or cuda)
        """
        super().__init__()
        
        # Use default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input dimension and output dimension
        self.input_dim = input_dim
        self.vector_dim = vector_dim
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], vector_dim))
        
        # Sequential module
        self.network = nn.Sequential(*layers)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform market data features into vector representations.
        
        Args:
            x: Tensor of shape (batch_size, input_dim) containing market data features
            
        Returns:
            Tensor of shape (batch_size, vector_dim) containing vector representations
        """
        # Forward pass through the network
        vectors = self.network(x)
        
        # Normalize vectors to have unit norm (L2 normalization)
        normalized_vectors = F.normalize(vectors, p=2, dim=1)
        
        return normalized_vectors
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[int, float]:
        """
        Calculate feature importance using integrated gradients.
        
        Args:
            x: Tensor of shape (batch_size, input_dim) containing market data features
            
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        # Set model to evaluation mode
        self.eval()
        
        # Require gradients for input
        x_with_grad = x.clone().detach().requires_grad_(True)
        
        # Get output for this input
        output = self.forward(x_with_grad)
        
        # Calculate gradient of mean output with respect to input
        output_mean = output.mean()
        output_mean.backward()
        
        # Get gradients
        gradients = x_with_grad.grad.abs()
        
        # Average across batch dimension
        avg_gradients = gradients.mean(dim=0)
        
        # Convert to dictionary of feature index to importance
        importance_dict = {i: float(avg_gradients[i].item()) for i in range(self.input_dim)}
        
        # Sort by importance (descending)
        importance_dict = {k: v for k, v in 
                          sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}
        
        return importance_dict


# Example usage
if __name__ == "__main__":
    # Create a dummy market data DataFrame
    dates = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
    market_data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 5, len(dates)),
        'high': np.random.normal(105, 5, len(dates)),
        'low': np.random.normal(95, 5, len(dates)),
        'close': np.random.normal(102, 5, len(dates)),
        'volume': np.random.normal(1000000, 200000, len(dates))
    })
    
    # Process market data
    processor = MarketDataProcessor()
    processed_data = processor.process(market_data)
    
    # Print the first few rows
    print(processed_data.head())
    
    # Create a vectorizer (input dimension will be the number of features in processed data)
    num_features = len(processed_data.columns) - 1  # Exclude 'date' column
    vectorizer = NumericalVectorizer(input_dim=num_features)
    
    # Convert a batch of processed data to tensors
    # Exclude the date column and convert to a tensor
    batch = processed_data.drop('date', axis=1).iloc[:5].values
    batch_tensor = torch.tensor(batch, dtype=torch.float32)
    
    # Get vectors
    vectors = vectorizer(batch_tensor)
    print(f"Numerical vectors shape: {vectors.shape}")
