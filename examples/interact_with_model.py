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
    """Fetch real financial news for the given tickers using Alpha Vantage."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the ALPHA_VANTAGE_API_KEY environment variable to fetch news.")

    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # For Alpha Vantage, we combine tickers into a comma-separated list
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
    
    news_items = []
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code} - {response.text}")
        return pd.DataFrame(columns=['date', 'headline', 'ticker', 'source'])
    
    data = response.json()
    
    # Process the data from Alpha Vantage format
    for item in data.get('feed', []):
        # For each ticker mentioned in the article
        ticker_sentiments = item.get('ticker_sentiment', [])
        
        # If no specific tickers, associate with all requested tickers
        if not ticker_sentiments:
            for ticker in tickers:
                news_items.append({
                    'date': item.get('time_published', ''),
                    'headline': item.get('title', ''),
                    'ticker': ticker,
                    'source': item.get('source', '')
                })
        else:
            # Otherwise, associate with each mentioned ticker
            for ticker_sentiment in ticker_sentiments:
                ticker = ticker_sentiment.get('ticker')
                # Only include if it's one of our requested tickers
                if ticker in tickers:
                    news_items.append({
                        'date': item.get('time_published', ''),
                        'headline': item.get('title', ''),
                        'ticker': ticker,
                        'source': item.get('source', '')
                    })
    
    # Convert to DataFrame
    if news_items:
        news_data = pd.DataFrame(news_items)
        
        # Format date properly - Alpha Vantage format is YYYYMMDDTHHMM
        news_data['date'] = pd.to_datetime(news_data['date'], format='%Y%m%dT%H%M', errors='coerce')
        
        # Filter out any rows with invalid dates
        news_data = news_data[~news_data['date'].isna()]
        
        # Sort by date
        news_data = news_data.sort_values(by='date', ascending=False)
        
        return news_data
    else:
        return pd.DataFrame(columns=['date', 'headline', 'ticker', 'source'])


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


def query_llm_for_interpretation(prediction_results, market_data, news_data, prediction_horizon, 
                              llm_config=None, system_prompt=None):
    """
    Send prediction results to an LLM API for interpretation.
    
    Args:
        prediction_results: The prediction results to interpret
        market_data: Market data for context
        news_data: News data for context
        prediction_horizon: Prediction horizon in days
        llm_config: Optional configuration for the LLM API
        system_prompt: Optional custom system prompt
        
    Returns:
        LLM interpretation of the prediction results
    """
    # Import config from utils if available
    try:
        from vectorfin.src.utils.config import config
        # Use config if available
        default_llm_config = {
            "api_url": config.get("llm.api_url", "http://10.102.138.33:6223/v1/chat/completions"),
            "model_name": config.get("llm.model_name", "gemma-3-4b-it-qat"),
            "api_key": config.get("llm.api_key"),
            "connect_timeout": config.get("llm.connect_timeout", 10),
            "read_timeout": config.get("llm.read_timeout", 120),
            "max_retries": config.get("llm.max_retries", 2),
            "temperature": config.get("llm.temperature", 0.3)
        }
    except ImportError:
        # Fallback to environment variables
        default_llm_config = {
            "api_url": os.getenv("LLM_API_URL", "http://10.102.138.33:6223/v1/chat/completions"),
            "model_name": os.getenv("INTERPRETATION_MODEL", "gemma-3-4b-it-qat"),
            "api_key": os.getenv("LLM_API_KEY"),
            "connect_timeout": int(os.getenv("LLM_CONNECT_TIMEOUT", "10")),
            "read_timeout": int(os.getenv("LLM_READ_TIMEOUT", "120")),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "2")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3"))
        }
    
    # Use provided config or fall back to defaults
    if llm_config is None:
        llm_config = default_llm_config
    else:
        # Merge with defaults
        for key, value in default_llm_config.items():
            if key not in llm_config:
                llm_config[key] = value
    
    # Use provided system prompt or use default
    if system_prompt is None:
        system_prompt = "You are a financial analysis assistant that interprets market predictions."
    
    # Create the user prompt
    user_prompt = f"""
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
    print(user_prompt)
    
    # Maximum number of retry attempts
    max_retries = llm_config["max_retries"]
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Get API configuration
            llm_api_url = llm_config["api_url"]
            model_name = llm_config["model_name"]
            
            session = requests.Session()
            # Use a shorter timeout just for connection establishment
            adapter = requests.adapters.HTTPAdapter(max_retries=1)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            print(f"Connecting to LLM API at {llm_api_url}... (Attempt {retry_count + 1}/{max_retries + 1})")
            
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if llm_config.get("api_key"):
                headers["Authorization"] = f"Bearer {llm_config['api_key']}"
            
            # Prepare request data
            request_data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": llm_config["temperature"]
            }
            
            # Separate connection timeout (shorter) from read timeout (longer)
            response = session.post(
                llm_api_url,
                headers=headers,
                json=request_data,
                timeout=(llm_config["connect_timeout"], llm_config["read_timeout"])
            )
            
            # Check if request was successful
            if response.status_code == 200:
                try:
                    interpretation = response.json()["choices"][0]["message"]["content"]
                    print(f"Successfully received interpretation from {model_name}")
                    return interpretation
                except (KeyError, IndexError) as e:
                    print(f"Error parsing LLM response: {str(e)}")
                    print(f"Response content: {response.text[:500]}...")  # Print first 500 chars of response
                    # Continue to retry or fallback if parsing failed
            else:
                print(f"LLM API returned status code: {response.status_code}")
                print(f"Response content: {response.text[:500]}...")  # Print first 500 chars of response
                # If we get a 5xx error, retry; for 4xx errors, don't bother retrying
                if response.status_code >= 500:
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"Retrying request (attempt {retry_count + 1}/{max_retries + 1})...")
                        continue
                # If we've exhausted retries or it's a 4xx error, fall back
                break
        
        except requests.RequestException as e:
            print(f"\n--- LLM Connection Failed: {str(e)} ---")
            if retry_count < max_retries and "timeout" in str(e).lower():  # Only retry timeouts
                retry_count += 1
                print(f"Request timed out. Retrying (attempt {retry_count + 1}/{max_retries + 1})...")
                continue
            break  # Exit loop if it's not a timeout or if we've exhausted retries
        except (KeyError, ValueError) as e:
            print(f"\n--- LLM Processing Error: {str(e)} ---")
            break  # Don't retry for parsing errors
    
    # If we get here, all retries failed or we encountered a non-retryable error
    print("Generating fallback interpretation...")
    interpretation = generate_fallback_interpretation(prediction_results, market_data, news_data)
    
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


def make_prediction(system, news_data, market_data, prediction_horizon=5, use_real_model=True):
    """
    Use the model to make a prediction.
    
    Args:
        system: The trained VectorFin system
        news_data: Preprocessed news data
        market_data: Market data in aligned format
        prediction_horizon: Number of days to predict into the future
        use_real_model: Whether to use the real model or generate random predictions for testing
        
    Returns:
        Prediction results dictionary
    """
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
    
    if use_real_model and system is not None:
        try:
            # Use the actual model to make predictions
            print("Using trained model for prediction...")
            
            # Prepare inputs for the model
            if isinstance(market_data, pd.DataFrame) and market_data.shape[0] > 0:
                # Get features from the market data
                market_features = system.prepare_market_features(market_data)
                
                # Get text features from news data if available
                text_features = None
                if texts:
                    text_features = system.prepare_text_features(texts)
                
                # Make predictions using the model components
                direction_prob = system.predict_direction(market_features, text_features)
                magnitude = system.predict_magnitude(market_features, text_features)
                volatility = system.predict_volatility(market_features, text_features)
                
                # Format results
                results = {
                    'date': latest_date.strftime('%Y-%m-%d'),
                    'prediction_horizon': prediction_horizon,
                    'predictions': {
                        'direction': float(direction_prob),
                        'magnitude': float(magnitude),
                        'volatility': float(volatility)
                    }
                }
                return results
        except Exception as e:
            print(f"Error using trained model: {str(e)}")
            print("Falling back to random predictions for demonstration.")
    
    # Fallback: Generate random predictions for demonstration
    # This is useful for testing the API without a trained model
    print("Using random predictions for demonstration.")
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
    

def generate_fallback_interpretation(prediction_results, market_data, news_data):
    """Generate a fallback interpretation when LLM is not available."""
    direction = prediction_results["predictions"]["direction"]
    magnitude = prediction_results["predictions"]["magnitude"]
    volatility = prediction_results["predictions"]["volatility"]
    
    # Determine direction strength
    if direction > 0.75:
        direction_desc = "strong bullish"
    elif direction > 0.60:
        direction_desc = "moderately bullish"
    elif direction > 0.50:
        direction_desc = "slightly bullish"
    elif direction > 0.40:
        direction_desc = "slightly bearish"
    elif direction > 0.25:
        direction_desc = "moderately bearish"
    else:
        direction_desc = "strong bearish"
    
    # Categorize magnitude
    if magnitude > 5:
        magnitude_desc = "very large"
    elif magnitude > 3:
        magnitude_desc = "large"
    elif magnitude > 1.5:
        magnitude_desc = "moderate"
    else:
        magnitude_desc = "small"
    
    # Categorize volatility
    if volatility > 4:
        volatility_desc = "extremely high"
    elif volatility > 2.5:
        volatility_desc = "high"
    elif volatility > 1.5:
        volatility_desc = "moderate"
    else:
        volatility_desc = "low"
    
    # Recent price movement
    ticker_movements = []
    if isinstance(market_data, dict):
        for ticker, df in market_data.items():
            try:
                recent_change = ((df['close'].iloc[-1] / df['close'].iloc[-6]) - 1) * 100
                ticker_movements.append(f"{ticker} ({recent_change:.1f}%)")
            except:
                pass
    ticker_movement_text = ", ".join(ticker_movements) if ticker_movements else "recent market movement"
    
    # Generate interpretation
    interpretation = f"""## Financial Prediction Interpretation (FALLBACK)

### Summary
The model predicts a {direction_desc} signal with {magnitude_desc} expected price movement ({magnitude:.2f}%) and {volatility_desc} volatility ({volatility:.2f}%) over the next {prediction_results['prediction_horizon']} days.

### Key Factors
- Current direction probability: {direction:.2f} (>0.5 indicates upward movement)
- Recent market performance: {ticker_movement_text}
- News sentiment may be affecting the prediction

### Potential Risks
- Higher than normal volatility indicates increased uncertainty
- Market conditions could change rapidly
- External factors not captured in the data may affect outcomes

### Recommendation
Based on the {direction_desc} signal with {volatility_desc} volatility, consider a {get_recommendation(direction, volatility)} approach to this market.
"""
    
    return interpretation


def get_recommendation(direction, volatility):
    """Generate a recommendation based on direction and volatility."""
    if direction > 0.75:
        if volatility < 2.0:
            return "strong buy"
        else:
            return "cautious buy with position sizing to account for volatility"
    elif direction > 0.60:
        if volatility < 2.5:
            return "moderate buy"
        else:
            return "small position buy with tight risk controls"
    elif direction > 0.50:
        return "hold with potential to add on dips"
    elif direction > 0.40:
        return "hold, but avoid adding to positions"
    elif direction > 0.25:
        if volatility < 2.5:
            return "reduce positions"
        else:
            return "consider protective options or reduce positions"
    else:
        return "sell or consider short positions with strict risk management"


def main(args=None):
    """
    Main function to run VectorFin prediction.
    
    Args:
        args: Optional command-line arguments
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="VectorFin Market Prediction")
    parser.add_argument("--tickers", type=str, nargs="+", default=['AAPL', 'MSFT', 'GOOGL'],
                        help="List of stock tickers to analyze")
    parser.add_argument("--models-dir", type=str, default="./trained_models",
                        help="Directory containing trained models")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Prediction horizon in days")
    parser.add_argument("--news-days", type=int, default=10,
                        help="Days of news to fetch")
    parser.add_argument("--market-days", type=int, default=30,
                        help="Days of market data to fetch")
    parser.add_argument("--llm-api-url", type=str, default=None,
                        help="LLM API URL for interpretation")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM model name for interpretation")
    parser.add_argument("--llm-timeout", type=int, default=None,
                        help="LLM API read timeout in seconds")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save prediction results to file")
    parser.add_argument("--random", action="store_true",
                        help="Use random predictions (for testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for prediction results (JSON)")
    
    # Parse arguments
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    # Try to import config if available
    try:
        from vectorfin.src.utils.config import config
        # Use config for defaults if not specified in arguments
        if args.llm_api_url is None:
            args.llm_api_url = config.get("llm.api_url")
        if args.llm_model is None:
            args.llm_model = config.get("llm.model_name")
        if args.llm_timeout is None:
            args.llm_timeout = config.get("llm.read_timeout")
    except ImportError:
        pass
    
    # Set up LLM configuration
    llm_config = {
        "api_url": args.llm_api_url if args.llm_api_url else os.getenv("LLM_API_URL", "http://10.102.138.33:6223/v1/chat/completions"),
        "model_name": args.llm_model if args.llm_model else os.getenv("INTERPRETATION_MODEL", "gemma-3-4b-it-qat"),
        "api_key": os.getenv("LLM_API_KEY"),
        "read_timeout": args.llm_timeout if args.llm_timeout else int(os.getenv("LLM_READ_TIMEOUT", "120")),
        "connect_timeout": int(os.getenv("LLM_CONNECT_TIMEOUT", "10")),
        "max_retries": int(os.getenv("LLM_MAX_RETRIES", "2")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3"))
    }
    
    # Load trained model
    system = load_trained_model(args.models_dir)
    
    # Fetch raw data
    news_data = fetch_recent_news(args.tickers, days=args.news_days)
    market_data = fetch_market_data(args.tickers, days=args.market_days)
    
    # Prepare data for prediction (preprocess news & align market)
    processed_news, aligned_market_data = prepare_data_for_prediction(news_data, market_data)
    
    # Make prediction using processed data
    prediction_results = make_prediction(
        system, 
        processed_news, 
        aligned_market_data, 
        args.horizon, 
        use_real_model=not args.random
    )
    print("\n--- Prediction Results ---")
    print(json.dumps(prediction_results, indent=2))
    
    # Get LLM interpretation using aligned market and processed news
    interpretation = query_llm_for_interpretation(
        prediction_results, 
        aligned_market_data, 
        processed_news, 
        args.horizon,
        llm_config=llm_config
    )
    print("\n--- LLM Interpretation ---")
    print(interpretation)
    
    # Save prediction results and interpretation to files
    if not args.no_save:
        try:
            save_prediction_results(prediction_results, interpretation, args.tickers)
        except Exception as e:
            print(f"\nWarning: Could not save prediction results: {str(e)}")
    
    # Save to custom output file if specified
    if args.output:
        try:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(args.output, "w") as f:
                json.dump({
                    "prediction": prediction_results,
                    "interpretation": interpretation,
                    "tickers": args.tickers,
                    "generated_at": datetime.now().isoformat()
                }, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"\nWarning: Could not save to output file: {str(e)}")
    
    return {
        "prediction_results": prediction_results,
        "interpretation": interpretation
    }


if __name__ == "__main__":
    main()
