"""
Data Module for VectorFin

This module handles data acquisition, preprocessing, and loading
for both text and numerical financial data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import os
from typing import Dict, List, Tuple, Union, Optional
import datetime
import json
from pathlib import Path


class FinancialTextData:
    """
    Class for loading and processing financial text data.
    """
    
    @staticmethod
    def load_news_data(
        filepath: str,
        date_column: str = 'date',
        text_column: str = 'headline',
        source_column: Optional[str] = 'source'
    ) -> pd.DataFrame:
        """
        Load financial news data from a file.
        
        Args:
            filepath: Path to the data file (CSV or JSON)
            date_column: Name of the date column
            text_column: Name of the text column
            source_column: Name of the source column (optional)
            
        Returns:
            DataFrame with processed news data
        """
        # Check file extension
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext == '.json':
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Validate required columns
        required_columns = [date_column, text_column]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values(by=date_column)
        
        return df
    
    @staticmethod
    def fetch_news_api(
        api_key: str,
        tickers: List[str],
        start_date: str,
        end_date: str,
        api_url: str = "https://api.example.com/news"  # Replace with actual API URL
    ) -> pd.DataFrame:
        """
        Fetch financial news from an API.
        
        Args:
            api_key: API key for authentication
            tickers: List of ticker symbols to fetch news for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            api_url: URL of the news API
            
        Returns:
            DataFrame with news data
        """
        import requests
        
        # Format date range
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        # Initialize results list
        all_news = []
        
        # Fetch news for each ticker
        for ticker in tickers:
            params = {
                'apiKey': api_key,
                'ticker': ticker,
                'from': start_date,
                'to': end_date
            }
            
            response = requests.get(api_url, params=params)
            
            if response.status_code == 200:
                news_data = response.json()
                
                # Process and add to results
                for item in news_data.get('articles', []):
                    all_news.append({
                        'date': item.get('publishedAt'),
                        'headline': item.get('title'),
                        'content': item.get('content'),
                        'source': item.get('source', {}).get('name'),
                        'url': item.get('url'),
                        'ticker': ticker
                    })
            else:
                print(f"Error fetching news for {ticker}: {response.status_code}")
        
        # Convert to DataFrame
        if all_news:
            df = pd.DataFrame(all_news)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date')
            return df
        else:
            return pd.DataFrame(columns=['date', 'headline', 'content', 'source', 'url', 'ticker'])
    
    @staticmethod
    def preprocess_text_data(
        df: pd.DataFrame,
        text_column: str = 'headline',
        max_length: int = 128
    ) -> pd.DataFrame:
        """
        Preprocess text data for vectorization.
        
        Args:
            df: DataFrame with text data
            text_column: Name of the text column
            max_length: Maximum length for truncation
            
        Returns:
            DataFrame with preprocessed text data
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Basic cleaning
        result[text_column] = result[text_column].astype(str)
        result[text_column] = result[text_column].str.strip()
        
        # Truncate if needed
        result[text_column] = result[text_column].apply(
            lambda x: x[:max_length] if len(x) > max_length else x
        )
        
        return result


class MarketData:
    """
    Class for loading and processing market data.
    """
    
    @staticmethod
    def fetch_market_data(
        tickers: Union[str, List[str]],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data from Yahoo Finance.
        
        Args:
            tickers: Ticker symbol or list of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo, etc.)
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames with market data
        """
        # Convert to list if a single ticker is provided
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Format date range
        start_date_obj = pd.to_datetime(start_date)
        end_date_obj = pd.to_datetime(end_date)
        
        # Initialize results dictionary
        results = {}
        
        # Fetch data for each ticker
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date_obj,
                    end=end_date_obj,
                    interval=interval,
                    progress=False
                )
                
                # Skip if no data found
                if df.empty:
                    print(f"No data found for {ticker}")
                    continue
                
                # Reset index to make Date a column
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Handle MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    # Extract first level for Date column, second level for other columns
                    new_cols = []
                    for col in df.columns:
                        if col[0] == 'Date':
                            new_cols.append('date')
                        else:
                            new_cols.append(col[0].lower())
                    df.columns = new_cols
                else:
                    # For single-level columns
                    df.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in df.columns]
                
                # Ensure 'date' column exists
                if 'date' not in df.columns:
                    if 'datetime' in df.columns:
                        df = df.rename(columns={'datetime': 'date'})
                    elif 'index' in df.columns:
                        df = df.rename(columns={'index': 'date'})
                
                # Store in results dictionary
                results[ticker] = df
                
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        return results
    
    @staticmethod
    def load_market_data(filepath: str) -> Dict[str, pd.DataFrame]:
        """
        Load market data from a file.
        
        Args:
            filepath: Path to the data file or directory
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames with market data
        """
        # Check if it's a file or directory
        path = Path(filepath)
        
        if path.is_file():
            # Single file
            ext = path.suffix.lower()
            
            if ext == '.csv':
                df = pd.read_csv(filepath)
                
                # Try to determine ticker from filename
                ticker = path.stem.split('_')[0].upper()
                
                return {ticker: df}
                
            elif ext == '.json':
                # Load JSON into a dictionary of DataFrames
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                results = {}
                for ticker, ticker_data in data.items():
                    results[ticker] = pd.DataFrame(ticker_data)
                
                return results
                
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        elif path.is_dir():
            # Directory of files
            results = {}
            
            # Process each CSV file
            for csv_file in path.glob('*.csv'):
                ticker = csv_file.stem.split('_')[0].upper()
                df = pd.read_csv(csv_file)
                results[ticker] = df
            
            return results
            
        else:
            raise ValueError(f"Path not found: {filepath}")
    
    @staticmethod
    def align_market_data(market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align market data from multiple tickers to a common date range.
        
        Args:
            market_data: Dictionary mapping ticker symbols to DataFrames
            
        Returns:
            DataFrame with aligned market data, with multi-level columns
        """
        if not market_data:
            return pd.DataFrame()
        
        # Initialize with the first DataFrame
        tickers = list(market_data.keys())
        first_ticker = tickers[0]
        
        # Create a copy of the first DataFrame with date as index
        aligned_data = market_data[first_ticker].copy()
        aligned_data = aligned_data.set_index('date')
        
        # Rename columns to include ticker
        aligned_data.columns = pd.MultiIndex.from_product(
            [[first_ticker], aligned_data.columns],
            names=['ticker', 'feature']
        )
        
        # Align other tickers
        for ticker in tickers[1:]:
            df = market_data[ticker].copy()
            
            # Set date as index
            df = df.set_index('date')
            
            # Rename columns to include ticker
            df.columns = pd.MultiIndex.from_product(
                [[ticker], df.columns],
                names=['ticker', 'feature']
            )
            
            # Join with the aligned data
            aligned_data = aligned_data.join(df, how='outer')
        
        # Sort by date
        aligned_data = aligned_data.sort_index()
        
        return aligned_data


class AlignedFinancialDataset(Dataset):
    """
    Dataset for aligned financial text and market data.
    
    This dataset combines text and market data with temporal alignment,
    ensuring that text data precedes the market movements it might influence.
    """
    
    def __init__(
        self,
        text_data: pd.DataFrame,
        market_data: pd.DataFrame,
        text_column: str = 'headline',
        date_column: str = 'date',
        prediction_horizon: int = 1,
        max_texts_per_day: int = 10,
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            text_data: DataFrame with text data
            market_data: DataFrame with market data
            text_column: Name of the text column
            date_column: Name of the date column
            prediction_horizon: Number of days ahead to predict
            max_texts_per_day: Maximum number of texts per day
            transform: Optional transform to apply to the data
        """
        self.text_data = text_data
        self.market_data = market_data
        self.text_column = text_column
        self.date_column = date_column
        self.prediction_horizon = prediction_horizon
        self.max_texts_per_day = max_texts_per_day
        self.transform = transform
        
        # Group text data by date
        self.text_by_date = self._group_text_by_date()
        
        # Create aligned samples
        self.samples = self._create_samples()
    
    def _group_text_by_date(self) -> Dict[str, List[str]]:
        """
        Group text data by date.
        
        Returns:
            Dictionary mapping date strings (YYYY-MM-DD) to lists of texts
        """
        text_by_date = {}
        
        # Ensure date column is datetime
        self.text_data[self.date_column] = pd.to_datetime(self.text_data[self.date_column])
        
        # Group by date
        date_groups = self.text_data.groupby(self.date_column)
        
        for date, group in date_groups:
            # Convert date to string format for consistent keys
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            
            # Get texts for this date
            texts = group[self.text_column].tolist()
            
            # Limit number of texts if needed
            if len(texts) > self.max_texts_per_day:
                texts = texts[:self.max_texts_per_day]
            
            text_by_date[date_str] = texts
        
        return text_by_date
    
    def _create_samples(self) -> List[Dict]:
        """
        Create samples with aligned text and market data.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # Reset index to make date a column if it's not already
        if isinstance(self.market_data.index, pd.DatetimeIndex):
            market_data = self.market_data.reset_index()
        else:
            market_data = self.market_data.copy()
        
        # Convert date column to timestamp for consistent comparison
        if 'date' in market_data.columns:
            market_data['date'] = pd.to_datetime(market_data['date'])
            
        # Create a set of text_by_date keys for fast lookup
        available_dates = set(pd.to_datetime(date) for date in self.text_by_date.keys())
        
        # Iterate through market data rows
        for i in range(len(market_data) - self.prediction_horizon):
            # Get current date from the market data
            try:
                if 'date' in market_data.columns:
                    current_date_val = market_data.iloc[i]['date']
                    # Handle Series object or other unexpected types
                    if isinstance(current_date_val, pd.Series):
                        print(f"Debug: date is a Series with values: {current_date_val.values}")
                        current_date = pd.to_datetime(current_date_val.iloc[0])
                    else:
                        current_date = pd.to_datetime(current_date_val)
                else:
                    # If date is the index
                    current_date = pd.to_datetime(market_data.index[i])
                    
                target_idx = i + self.prediction_horizon
                if 'date' in market_data.columns:
                    target_date_val = market_data.iloc[target_idx]['date']
                    if isinstance(target_date_val, pd.Series):
                        target_date = pd.to_datetime(target_date_val.iloc[0])
                    else:
                        target_date = pd.to_datetime(target_date_val)
                else:
                    target_date = pd.to_datetime(market_data.index[target_idx])
                
                # Skip if current_date is not a valid date
                if not isinstance(current_date, (pd.Timestamp, datetime.datetime)):
                    print(f"Debug: Skipping invalid date type: {type(current_date)}")
                    continue
                    
                # Skip if no text data for this date
                current_date_str = current_date.strftime('%Y-%m-%d')
                if current_date_str not in [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in available_dates]:
                    continue
                
                # Convert current_date to string format for lookup
                current_date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
                
                # Check if we have text data for this date
                if current_date_str not in self.text_by_date:
                    continue
                
                # Get text data for the current date
                texts = self.text_by_date[current_date_str]
                    
            except Exception as e:
                print(f"Debug: Error processing date at index {i}: {e}")
                continue
            
            # Get market data for the current date
            if 'date' in market_data.columns:
                current_market_data = market_data.iloc[i].drop('date')
            else:
                current_market_data = market_data.iloc[i]
            
            # Get target market data
            if 'date' in market_data.columns:
                target_market_data = market_data.iloc[target_idx].drop('date')
            else:
                target_market_data = market_data.iloc[target_idx]
            
            # Create sample
            sample = {
                'date': current_date,
                'texts': texts,
                'market_data': current_market_data.values,
                'target_date': target_date,
                'target': target_market_data.values
            }
            
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample by index.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Sample dictionary
        """
        sample = self.samples[idx]
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample


# Example usage
if __name__ == "__main__":
    # Example market data fetching
    market_data = MarketData.fetch_market_data(
        tickers=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    
    print(f"Fetched data for {len(market_data)} tickers")
    
    # Create sample news data
    news_data = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", end="2023-01-31"),
        'headline': [
            f"Sample news headline {i}" for i in range(31)
        ],
        'source': ['News Source'] * 31
    })
    
    # Preprocess news data
    processed_news = FinancialTextData.preprocess_text_data(news_data)
    print(f"Processed {len(processed_news)} news items")
