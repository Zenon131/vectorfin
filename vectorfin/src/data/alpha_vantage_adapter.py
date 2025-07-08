"""
Alpha Vantage API adapter for VectorFin.

This module provides a compatibility layer to use Alpha Vantage's API
as a drop-in replacement for NewsAPI in VectorFin.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any


def fetch_news(tickers: List[str], days: int = 10) -> pd.DataFrame:
    """
    Fetch news for the given tickers using Alpha Vantage's News API.
    
    Args:
        tickers: List of ticker symbols to fetch news for
        days: Number of days of news to fetch
        
    Returns:
        DataFrame with news data in the same format expected by VectorFin
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the ALPHA_VANTAGE_API_KEY environment variable to fetch news.")

    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # For Alpha Vantage, we need to combine tickers into a comma-separated list
    ticker_string = ",".join(tickers)
    
    # Fetch news from Alpha Vantage
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker_string,
        'time_from': start_date.strftime('%Y%m%dT%H%M'),  # Format: YYYYMMDDTHHMM
        'limit': 1000,  # Get more results to ensure coverage
        'apikey': api_key
    }
    
    response = requests.get(url, params=params)
    
    # Check for API errors
    if response.status_code != 200:
        raise RuntimeError(f"Alpha Vantage API error: {response.status_code} - {response.text}")
    
    data = response.json()
    news_items = []
    
    # Process the data into the format expected by VectorFin
    for item in data.get('feed', []):
        # For each ticker mentioned in the article
        for ticker_sentiment in item.get('ticker_sentiment', []):
            ticker = ticker_sentiment.get('ticker')
            # Only include if it's one of our requested tickers
            if ticker in tickers:
                news_items.append({
                    'date': item.get('time_published', ''),
                    'headline': item.get('title', ''),
                    'content': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'ticker': ticker,
                    # Additional fields that might be useful
                    'relevance_score': ticker_sentiment.get('relevance_score'),
                    'sentiment': ticker_sentiment.get('ticker_sentiment_label')
                })
    
    # Convert to DataFrame
    if news_items:
        df = pd.DataFrame(news_items)
        
        # Format date properly
        # Alpha Vantage format is typically: YYYYMMDDTHHMM
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M', errors='coerce')
        
        # Filter out any rows with invalid dates
        df = df[~df['date'].isna()]
        
        # Sort by date
        df = df.sort_values(by='date', ascending=False)
        
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'date', 'headline', 'content', 'source', 'url', 'ticker',
            'relevance_score', 'sentiment'
        ])


def setup_alpha_vantage_adapter():
    """
    Set up the Alpha Vantage adapter as a drop-in replacement for NewsAPI.
    
    This function checks for the ALPHA_VANTAGE_API_KEY environment variable
    and sets it as NEWS_API_KEY for compatibility with existing code.
    """
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_vantage_key:
        os.environ["NEWS_API_KEY"] = alpha_vantage_key
        return True
    return False
