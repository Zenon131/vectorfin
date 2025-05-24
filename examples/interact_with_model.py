"""
Interact with a trained VectorFin model and use LLM to interpret results.

This script demonstrates how to load a trained model, make predictions,
and use an LLM to interpret the financial predictions.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
load_dotenv()

# Import save_predictions from the same directory
try:
    from save_predictions import save_prediction_results
except ImportError:
    # Define a fallback function in case import fails
    def save_prediction_results(prediction_results, interpretation, tickers):
        print("Warning: save_predictions module not found. Saving disabled.")


# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem
from vectorfin.src.data.data_loader import MarketData, FinancialTextData


def load_trained_model(models_dir="./trained_models", device=None):
    """Load a trained VectorFin system."""
    print(f"Loading model from {models_dir}...")
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create an empty system (will be filled with loaded models)
    system = VectorFinSystem.load_models(
        directory=models_dir,
        vector_dim=128,
        sentiment_dim=16,
        fusion_dim=128,
        device=device
    )
    
    print("Model loaded successfully!")
    return system


def fetch_recent_news(tickers, days=5):
    """Fetch real financial news for the given tickers using NewsAPI."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the NEWS_API_KEY environment variable to fetch real news.")

    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    news_items = []
    url = "https://newsapi.org/v2/everything"
    for ticker in tickers:
        params = {
            'q': ticker,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,
            'apiKey': api_key
        }
        response = requests.get(url, params=params)
        data = response.json().get('articles', [])
        for article in data:
            news_items.append({
                'date': article.get('publishedAt'),
                'headline': article.get('title'),
                'ticker': ticker,
                'source': article.get('source', {}).get('name')
            })
    
    news_data = pd.DataFrame(news_items)
    news_data['date'] = pd.to_datetime(news_data['date'])
    return news_data


def fetch_market_data(tickers, days=30):
    """Fetch recent market data for the given tickers."""
    print(f"Fetching market data for {tickers}...")
    
    # Calculate date range (we get more historical data for context)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Fetch data using VectorFin's MarketData class
    market_data = MarketData.fetch_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    return market_data


def prepare_data_for_prediction(news_data, market_data):
    """Prepare data for model prediction."""
    # Preprocess news data
    processed_news = FinancialTextData.preprocess_text_data(news_data)
    
    # Align market data
    aligned_market_data = MarketData.align_market_data(market_data)
    
    return processed_news, aligned_market_data


def query_llm_for_interpretation(prediction_results, market_data, news_data, prediction_horizon):
    """
    Send prediction results to an LLM API for interpretation.
    
    You can use OpenAI, Anthropic, or other providers.
    This function is a template - you'll need to add your API key and endpoint.
    """
    # For demonstration - in a real implementation, you'd send this to an API
    prompt = f"""
    Based on financial data and news analysis, interpret the following prediction:
    
    Prediction Horizon: {prediction_horizon} days
    
    Market Data Summary:
    {market_data_summary(market_data)}
    
    Recent News Headlines:
    {format_recent_news(news_data, 5)}
    
    Model Prediction Results:
    {json.dumps(prediction_results, indent=2)}
    
    Please provide:
    1. A concise interpretation of the prediction
    2. Key factors that might be influencing this prediction
    3. Potential risks or uncertainties to consider
    4. A recommendation based on this prediction (buy, hold, sell, etc.)
    """
    
    print("\n--- LLM Interpretation Prompt ---")
    print(prompt)
    
    # In a real implementation, you would make an API call like this:
    response = requests.post(
        "http://192.168.68.122:6223/v1/chat/completions",  # Or your preferred LLM API
        json={
            "model": "gemma-3-4b-it-qat",  # Or your preferred model
            "messages": [
                {"role": "system", "content": "You are a financial analysis assistant that interprets market predictions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
    )
    interpretation = response.json()["choices"][0]["message"]["content"]
    
    # Add fallback in case API call fails
    if not interpretation:
        print("Warning: API call failed or returned empty response. Using fallback interpretation.")
        interpretation = """
        ## Financial Prediction Interpretation (FALLBACK)
        
        The LLM API request failed. This is a fallback interpretation.
        
        Based on the prediction results, the model suggests a possible market movement
        in the coming days. Please check the numerical prediction values directly
        and consider consulting another source for interpretation.
        """
    
    return interpretation


def market_data_summary(market_data):
    """Create a summary of the market data for the LLM."""
    summary = []
    
    # Handle both original dictionary format and aligned DataFrame format
    if isinstance(market_data, dict):
        # Original market_data dict format
        for ticker, data in market_data.items():
            latest_data = data.iloc[-5:]  # Last 5 days
            
            # Calculate basic stats
            price_change = (latest_data['close'].iloc[-1] - latest_data['close'].iloc[0]) / latest_data['close'].iloc[0] * 100
            avg_volume = latest_data['volume'].mean()
            
            summary.append(f"{ticker}: {price_change:.2f}% price change over last 5 days, average volume: {avg_volume:.0f}")
    elif isinstance(market_data, pd.DataFrame):
        # Aligned market_data DataFrame format (from MarketData.align_market_data)
        if isinstance(market_data.columns, pd.MultiIndex) and market_data.columns.nlevels >= 2:
            # Handle each ticker in the MultiIndex columns
            for ticker in market_data.columns.get_level_values(0).unique():
                try:
                    # Get the latest 5 days of data for this ticker
                    ticker_data = market_data.iloc[-5:][ticker]
                    
                    # Check if 'close' and 'volume' exist for this ticker
                    if 'close' in ticker_data.columns and 'volume' in ticker_data.columns:
                        # Calculate price change and average volume
                        close_prices = ticker_data['close']
                        price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100
                        avg_volume = ticker_data['volume'].mean()
                        
                        summary.append(f"{ticker}: {price_change:.2f}% price change over last 5 days, average volume: {avg_volume:.0f}")
                except (KeyError, IndexError) as e:
                    summary.append(f"{ticker}: Unable to calculate stats - {str(e)}")
        else:
            # Single ticker aligned data or unknown format
            summary.append("Market data available but in an unexpected format.")
    else:
        summary.append("No market data available.")
    
    return "\n".join(summary)


def format_recent_news(news_data, num_items=5):
    """Format the most recent news for the LLM."""
    recent_news = news_data.sort_values('date', ascending=False).head(num_items)
    
    formatted = []
    for _, news in recent_news.iterrows():
        formatted.append(f"{news['date'].strftime('%Y-%m-%d')}: {news['headline']} (Sentiment: {news.get('sentiment', 'unknown')})")
    
    return "\n".join(formatted)


def make_prediction(system, news_data, market_data, prediction_horizon=5):
    """Use the model to make a prediction."""
    print(f"Making prediction with horizon of {prediction_horizon} days...")
    
    # Get the most recent date in the market data
    try:
        # Handle different market_data formats:
        if isinstance(market_data, dict):
            # Original market_data dict format
            latest_date = max(df['date'].max() for df in market_data.values() if 'date' in df.columns and not df.empty)
        elif isinstance(market_data, pd.DataFrame):
            # Aligned market_data DataFrame format (from MarketData.align_market_data)
            if market_data.index.name == 'date':
                latest_date = market_data.index.max()
            else:
                latest_date = datetime.now()
        else:
            # Fallback
            latest_date = datetime.now()
    except (ValueError, KeyError):
        print("Warning: Could not find date column in market data or market data is empty.")
        # Fallback to current date
        latest_date = datetime.now()
    
    # Filter news data for the past week
    recent_date = latest_date - timedelta(days=7)
    
    # Handle timezone-aware datetime if needed
    # Convert timezone-aware datetime to timezone-naive datetime for comparison
    if news_data.empty:
        recent_news = news_data
    else:
        # Check if 'date' column has timezone info
        date_dtype = news_data['date'].dtype
        has_tzinfo = hasattr(date_dtype, 'tz') and date_dtype.tz is not None
        
        if has_tzinfo:
            # If news_data dates have timezone, handle comparison accordingly
            if not hasattr(recent_date, 'tzinfo') or recent_date.tzinfo is None:
                # Convert recent_date to match timezone
                recent_date = pd.Timestamp(recent_date).tz_localize(date_dtype.tz)
        else:
            # If news_data dates are timezone-naive but recent_date has tz
            if hasattr(recent_date, 'tzinfo') and recent_date.tzinfo is not None:
                # Make recent_date timezone-naive
                recent_date = recent_date.replace(tzinfo=None)
        
        # Now filter
        recent_news = news_data[news_data['date'] >= recent_date]
    
    # Get texts from the recent news
    texts = recent_news['headline'].tolist() if not recent_news.empty else []
    
    # Since we're working with test models, we'll generate random predictions
    # This is just a demonstration and will be replaced with actual model predictions
    # when real trained models are available
    
    # Format results
    results = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'prediction_horizon': prediction_horizon,
        'predictions': {
            'direction': float(np.random.random()),  # Random probability of upward movement
            'magnitude': float(np.random.normal(0, 2)),  # Random percentage change
            'volatility': float(np.random.gamma(2, 1))  # Random volatility
        }
    }
    
    return results
    

def main():
    # Define parameters
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    models_dir = "./trained_models"
    prediction_horizon = 5  # days
    
    # Load trained model
    system = load_trained_model(models_dir)
    
    # Fetch raw data
    news_data = fetch_recent_news(tickers, days=10)
    market_data = fetch_market_data(tickers, days=30)
    
    # Prepare data for prediction (preprocess news & align market)
    processed_news, aligned_market_data = prepare_data_for_prediction(news_data, market_data)
    
    # Make prediction using processed data
    prediction_results = make_prediction(system, processed_news, aligned_market_data, prediction_horizon)
    print("\n--- Prediction Results ---")
    print(json.dumps(prediction_results, indent=2))
    
    # Get LLM interpretation using aligned market and processed news
    interpretation = query_llm_for_interpretation(prediction_results, aligned_market_data, processed_news, prediction_horizon)
    print("\n--- LLM Interpretation ---")
    print(interpretation)
    
    # Save prediction results and interpretation to files
    try:
        save_prediction_results(prediction_results, interpretation, tickers)
    except Exception as e:
        print(f"\nWarning: Could not save prediction results: {str(e)}")


if __name__ == "__main__":
    main()
