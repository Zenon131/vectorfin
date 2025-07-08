"""
Example script to interact with the VectorFin model using Alpha Vantage.

This version replaces the NewsAPI with Alpha Vantage for fetching financial news.
"""

import os
import sys
import requests
import pandas as pd
import json
import argparse
from datetime import datetime, timedelta
import traceback

# Add the parent directory to the path so we can import from the vectorfin package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Alpha Vantage adapter
from vectorfin.src.data.alpha_vantage_adapter import fetch_news

# Import needed modules from vectorfin
from vectorfin.src.models.vectorfin import VectorFinSystem
from vectorfin.src.data.data_loader import FinancialTextData, FinancialMarketData
from vectorfin.src.prediction_interpretation.prediction import InterpretationSystem
from vectorfin.src.utils.config import Config
from vectorfin.src.utils.common import setup_logging


def fetch_market_data(tickers, days=30):
    """Fetch recent market data for the given tickers."""
    print(f"Fetching market data for {tickers}...")
    
    # Calculate date range (we get more historical data for context)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # For this example, we'll generate synthetic data
    # In a real app, you'd use a market data API like Alpha Vantage's TIME_SERIES_DAILY
    print("Note: Using synthetic market data for demonstration")
    data = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end:
        if current_date.weekday() < 5:  # Only weekdays
            for ticker in tickers:
                # Generate random but somewhat realistic price data
                base_price = 100 + hash(ticker) % 400  # Different base price for each ticker
                price = base_price + (hash(str(current_date) + ticker) % 20) - 10
                data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'open': price - 1,
                    'high': price + 2,
                    'low': price - 2,
                    'close': price,
                    'volume': 1000000 + hash(str(current_date) + ticker) % 5000000
                })
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_data_for_prediction(news_data, market_data):
    """Prepare data for prediction by preprocessing and aligning."""
    # Preprocess text data
    processed_news = FinancialTextData.preprocess_text_data(news_data)
    
    # For market data, we need to ensure the data is properly aligned
    # We'll use the original market_data for this example
    aligned_market_data = market_data
    
    return processed_news, aligned_market_data


def query_llm_for_interpretation(prediction_results, market_data, news_data, prediction_horizon, 
                                tickers, confidence_scores=None):
    """
    Query an LLM for human-readable interpretation of prediction results.
    
    Args:
        prediction_results: Dictionary of prediction results for each ticker
        market_data: Recent market data for context
        news_data: News data for context
        prediction_horizon: Number of days for the prediction
        tickers: List of ticker symbols
        confidence_scores: Optional confidence scores for predictions
        
    Returns:
        String containing the LLM's interpretation
    """
    config = Config()
    interpreter = InterpretationSystem(config)
    
    # Prepare market context
    market_context = ""
    for ticker in tickers:
        ticker_data = market_data[market_data['ticker'] == ticker].sort_values('date', ascending=False).head(5)
        if not ticker_data.empty:
            market_context += f"\n{ticker} recent prices (last 5 days):\n"
            for _, row in ticker_data.iterrows():
                market_context += f"  {row['date'].strftime('%Y-%m-%d')}: Open: ${row['open']:.2f}, Close: ${row['close']:.2f}, Volume: {row['volume']:,}\n"
    
    # Format the prediction results for the prompt
    predictions_text = ""
    for ticker, direction in prediction_results.items():
        confidence = ""
        if confidence_scores and ticker in confidence_scores:
            confidence = f" (Confidence: {confidence_scores[ticker]:.2f})"
        predictions_text += f"{ticker}: {direction}{confidence}\n"
    
    # Build the prompt for the LLM
    prompt = f"""
You are a financial analyst assistant. I need you to interpret the following stock prediction results.

Prediction horizon: {prediction_horizon} days

Predicted price movements:
{predictions_text}

Market context:
{market_context}

Recent relevant news:
{format_recent_news(news_data, 5)}

Based on this information, please provide:
1. A brief overall market outlook for these stocks
2. Specific insights for each ticker, connecting the predictions to recent news and market trends
3. Potential risks or factors that could affect these predictions
4. Any recommended actions based on these predictions

Limit your response to 300-400 words.
"""

    try:
        interpretation = interpreter.generate_interpretation(prompt)
        return interpretation
    except Exception as e:
        print(f"Error querying LLM: {str(e)}")
        traceback.print_exc()
        # Fallback to a simple interpretation
        return generate_fallback_interpretation(prediction_results, market_data, news_data)


def generate_fallback_interpretation(prediction_results, market_data, news_data):
    """Generate a simple fallback interpretation when LLM query fails."""
    interpretation = "Prediction Interpretation (generated without LLM):\n\n"
    
    # Count directions
    up_count = sum(1 for d in prediction_results.values() if d == "UP")
    down_count = sum(1 for d in prediction_results.values() if d == "DOWN")
    
    # Add overall sentiment
    if up_count > down_count:
        interpretation += "Overall market sentiment appears BULLISH.\n"
    elif down_count > up_count:
        interpretation += "Overall market sentiment appears BEARISH.\n"
    else:
        interpretation += "Overall market sentiment appears MIXED.\n"
    
    # Add ticker-specific interpretations
    interpretation += "\nStock-specific predictions:\n"
    for ticker, direction in prediction_results.items():
        interpretation += f"- {ticker}: Predicted to go {direction}\n"
    
    # Add news summary
    if not news_data.empty:
        interpretation += "\nRecent news that may impact these predictions:\n"
        recent_news = news_data.sort_values('date', ascending=False).head(3)
        for _, news in recent_news.iterrows():
            interpretation += f"- {news['headline']}\n"
    
    return interpretation


def format_recent_news(news_data, num_items=5):
    """Format recent news into a string for the prompt."""
    if news_data.empty:
        return "No recent news available."
    
    recent_news = news_data.sort_values('date', ascending=False).head(num_items)
    news_text = ""
    for _, news in recent_news.iterrows():
        date_str = news['date'].strftime('%Y-%m-%d')
        news_text += f"[{date_str}] {news['headline']} (Ticker: {news['ticker']})\n"
    
    return news_text


def make_prediction(system, news_data, market_data, prediction_horizon=5, use_real_model=True):
    """
    Make prediction using the VectorFin system.
    
    Args:
        system: VectorFin system instance
        news_data: Preprocessed news data
        market_data: Market data for context
        prediction_horizon: Number of days for prediction
        use_real_model: Whether to use the real model or a demo mode
        
    Returns:
        Dictionary with predictions for each ticker
    """
    print(f"Making predictions for horizon of {prediction_horizon} days...")
    
    # Get unique tickers
    tickers = market_data['ticker'].unique().tolist()
    
    # For demo mode without real models, return random predictions
    if not use_real_model:
        import random
        predictions = {}
        for ticker in tickers:
            predictions[ticker] = random.choice(["UP", "DOWN"])
        return predictions
    
    # Filter news data to only include recent and relevant news
    # Get the most recent market data date
    if market_data.empty:
        recent_date = datetime.now() - timedelta(days=30)
    else:
        recent_date = market_data['date'].max() - timedelta(days=30)
    
    # Handle potential timezone issues
    if news_data.empty:
        recent_news = news_data
    else:
        # Check the date dtype
        date_dtype = news_data['date'].dtype
        
        if pd.api.types.is_datetime64tz_dtype(date_dtype):
            # If news_data dates have timezone, handle comparison accordingly
            if not pd.api.types.is_datetime64tz_dtype(pd.Series([recent_date]).dtype):
                recent_date = pd.Timestamp(recent_date).tz_localize('UTC')
        elif pd.api.types.is_datetime64tz_dtype(pd.Series([recent_date]).dtype):
            # If news_data dates are timezone-naive but recent_date has tz
            recent_date = pd.Timestamp(recent_date).tz_localize(None)
        
        # Filter news to only include recent items
        recent_news = news_data[news_data['date'] >= recent_date]
    
    # Make predictions
    predictions = {}
    confidence_scores = {}
    
    for ticker in tickers:
        # Filter data for this ticker
        ticker_news = recent_news[recent_news['ticker'] == ticker]
        ticker_market = market_data[market_data['ticker'] == ticker]
        
        # Skip if we don't have enough data
        if ticker_news.empty or ticker_market.empty:
            print(f"Skipping {ticker}: Insufficient data")
            continue
        
        try:
            result = system.predict(
                ticker_news, 
                ticker_market, 
                prediction_horizon=prediction_horizon
            )
            
            direction = "UP" if result.get('direction', 0) > 0.5 else "DOWN"
            confidence = result.get('confidence', 0.5)
            
            predictions[ticker] = direction
            confidence_scores[ticker] = confidence
            
            print(f"Prediction for {ticker}: {direction} (Confidence: {confidence:.2f})")
        except Exception as e:
            print(f"Error predicting for {ticker}: {str(e)}")
    
    return predictions, confidence_scores


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Interact with the VectorFin model")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                      help='Ticker symbols to analyze')
    parser.add_argument('--days', type=int, default=10,
                      help='Number of days of news to fetch')
    parser.add_argument('--horizon', type=int, default=5,
                      help='Prediction horizon in days')
    parser.add_argument('--demo', action='store_true',
                      help='Run in demo mode without real models')
    parser.add_argument('--model-dir', type=str, default=None,
                      help='Directory containing trained models')
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    config = Config()
    if args.model_dir:
        config.set('models_dir', args.model_dir)
    
    try:
        # Initialize the system
        system = VectorFinSystem(config)
        
        # Fetch data
        print(f"Fetching news for {args.tickers} over the past {args.days} days...")
        news_data = fetch_news(args.tickers, days=args.days)
        market_data = fetch_market_data(args.tickers, days=30)
        
        # Prepare data
        processed_news, aligned_market_data = prepare_data_for_prediction(news_data, market_data)
        
        # Make predictions
        predictions, confidence_scores = make_prediction(
            system, 
            processed_news, 
            aligned_market_data, 
            prediction_horizon=args.horizon,
            use_real_model=not args.demo
        )
        
        # Generate interpretation
        interpretation = query_llm_for_interpretation(
            predictions,
            market_data,
            news_data,
            args.horizon,
            args.tickers,
            confidence_scores
        )
        
        # Print results
        print("\n" + "="*80)
        print("PREDICTION RESULTS")
        print("="*80)
        for ticker, direction in predictions.items():
            confidence = confidence_scores.get(ticker, 0.5)
            print(f"{ticker}: {direction} (Confidence: {confidence:.2f})")
        
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        print(interpretation)
        
        # Save results to file
        output_dir = os.path.join(os.getcwd(), "prediction_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_file = os.path.join(output_dir, f"prediction_{timestamp}.json")
        interpretation_file = os.path.join(output_dir, f"interpretation_{timestamp}.txt")
        
        # Save prediction results
        with open(prediction_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "tickers": args.tickers,
                "prediction_horizon": args.horizon,
                "predictions": {t: {"direction": d, "confidence": confidence_scores.get(t, 0.5)} 
                              for t, d in predictions.items()}
            }, f, indent=2)
        
        # Save interpretation
        with open(interpretation_file, 'w') as f:
            f.write(interpretation)
        
        print("\nResults saved to:")
        print(f"- Predictions: {prediction_file}")
        print(f"- Interpretation: {interpretation_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
