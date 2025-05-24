"""
VectorFin Inference Script

This script demonstrates how to use a trained VectorFin model to make predictions
on new financial data.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import sys

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem
from vectorfin.src.data.data_loader import MarketData


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with a trained VectorFin system.')
    
    # Model parameters
    parser.add_argument('--models_dir', type=str, default='./trained_models',
                        help='Directory with trained models')
    parser.add_argument('--vector_dim', type=int, default=128,
                        help='Dimension of vectors in the shared space')
    parser.add_argument('--sentiment_dim', type=int, default=16,
                        help='Dimension of sentiment features')
    parser.add_argument('--fusion_dim', type=int, default=128,
                        help='Dimension of fused vectors')
    
    # Data parameters
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Ticker symbol to analyze')
    parser.add_argument('--days', type=int, default=10,
                        help='Number of recent days to analyze')
    
    # Visualization parameters
    parser.add_argument('--save_plots', action='store_true',
                        help='Save visualization plots')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save outputs')
                        
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for inference (cpu, cuda, or None for auto)')
    
    return parser.parse_args()


def fetch_recent_market_data(ticker, days):
    """Fetch recent market data for the specified ticker."""
    print(f"Fetching recent market data for {ticker}...")
    
    # Calculate date range
    end_date = datetime.now()
    # Add buffer days to account for weekends and holidays
    start_date = end_date - timedelta(days=days*2)
    
    # Format dates
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # Fetch data
    market_data = MarketData.fetch_market_data(
        tickers=[ticker],
        start_date=start_date_str,
        end_date=end_date_str
    )
    
    # Get the last N days of data
    if ticker in market_data:
        data = market_data[ticker].tail(days)
        print(f"Retrieved {len(data)} days of market data")
        return data
    else:
        print(f"No data found for {ticker}")
        return None


def get_recent_news(ticker, days):
    """
    Get recent financial news for the ticker.
    
    In a real application, this would fetch news from an API or database.
    For this example, we'll generate synthetic news.
    """
    print(f"Generating synthetic news for {ticker}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Templates for synthetic news
    news_templates = [
        "{ticker} stock shows promising momentum in market trading",
        "Investors closely watching {ticker} ahead of earnings report",
        "Analyst upgrades outlook for {ticker} citing strong fundamentals",
        "{ticker} introduces new product line, market response mixed",
        "Economic indicators suggest positive outlook for {ticker}",
        "Market volatility impacts {ticker} trading patterns",
        "{ticker} announces strategic partnership to expand market reach",
        "Industry report highlights {ticker}'s competitive position"
    ]
    
    # Generate news items (1-3 per day)
    news_items = []
    for date in dates:
        # Randomly decide number of news items for this day
        num_news = np.random.randint(1, 4)
        
        for _ in range(num_news):
            # Select random template
            template = np.random.choice(news_templates)
            
            # Create headline
            headline = template.format(ticker=ticker)
            
            # Add to news items
            news_items.append({
                'date': date,
                'headline': headline
            })
    
    # Convert to DataFrame
    news_data = pd.DataFrame(news_items)
    
    print(f"Generated {len(news_data)} synthetic news items")
    return news_data


def load_vectorfin_system(args):
    """Load a trained VectorFin system."""
    print(f"Loading VectorFin system from {args.models_dir}...")
    
    try:
        # Check if models directory exists
        if not Path(args.models_dir).exists():
            print(f"Models directory not found: {args.models_dir}")
            print("Creating a new VectorFin system with default parameters...")
            system = VectorFinSystem(
                vector_dim=args.vector_dim,
                sentiment_dim=args.sentiment_dim,
                fusion_dim=args.fusion_dim,
                device=args.device
            )
        else:
            # Load trained system
            system = VectorFinSystem.load_models(
                directory=args.models_dir,
                vector_dim=args.vector_dim,
                sentiment_dim=args.sentiment_dim,
                fusion_dim=args.fusion_dim,
                device=args.device
            )
        
        print("VectorFin system loaded successfully")
        return system
        
    except Exception as e:
        print(f"Error loading VectorFin system: {e}")
        print("Creating a new VectorFin system with default parameters...")
        system = VectorFinSystem(
            vector_dim=args.vector_dim,
            sentiment_dim=args.sentiment_dim,
            fusion_dim=args.fusion_dim,
            device=args.device
        )
        return system


def run_inference(system, market_data, news_data):
    """Run inference with the VectorFin system."""
    print("Running inference...")
    
    # Get news headlines for each market data date
    results = []
    
    # Process each day
    for index, row in market_data.iterrows():
        date = pd.Timestamp(index) if isinstance(index, pd.Timestamp) else pd.Timestamp(row['date'])
        
        # Get news for this date
        day_news = news_data[news_data['date'].dt.date == date.date()]
        
        if len(day_news) > 0:
            # Extract headlines
            headlines = day_news['headline'].tolist()
            
            # Create market data frame for this day
            day_market_data = pd.DataFrame({
                'open': [row['open']],
                'high': [row['high']],
                'low': [row['low']],
                'close': [row['close']],
                'volume': [row['volume']] if 'volume' in row else [0]
            })
            
            # Run analysis
            analysis = system.analyze_text_and_market(headlines, day_market_data)
            
            # Extract interpretations
            interpretations = analysis["interpretations"]
            
            # Store results
            results.append({
                'date': date,
                'headlines': headlines,
                'market_data': row.to_dict(),
                'interpretations': interpretations,
                'predictions': {
                    'direction': analysis["predictions"]["direction"].item(),
                    'magnitude': analysis["predictions"]["magnitude"].item(),
                    'volatility': analysis["predictions"]["volatility"].item(),
                }
            })
    
    print(f"Generated predictions for {len(results)} days")
    return results


def visualize_predictions(results, ticker, args):
    """Visualize the predictions."""
    if not results:
        print("No results to visualize")
        return
    
    print("Visualizing predictions...")
    
    # Extract dates and predictions
    dates = [r['date'] for r in results]
    actual_prices = [r['market_data']['close'] for r in results]
    direction_predictions = [r['predictions']['direction'] for r in results]
    magnitude_predictions = [r['predictions']['magnitude'] for r in results]
    volatility_predictions = [r['predictions']['volatility'] for r in results]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot actual prices
    ax1.plot(dates, actual_prices, 'b-', label='Actual Price')
    ax1.set_title(f'{ticker} Price and Predictions')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Plot direction predictions as a heatmap
    cmap = plt.cm.RdYlGn
    direction_colors = [cmap(dp) for dp in direction_predictions]
    
    # Plot prediction confidence
    ax2.bar(dates, direction_predictions, color=direction_colors, alpha=0.7, label='Direction Confidence')
    ax2.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Confidence')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    # Add text annotations for predictions
    for i, result in enumerate(results):
        summary = result['interpretations'][0]['summary']
        ax1.annotate(f"{i+1}", 
                    (dates[i], actual_prices[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Add legend for numbers
    legend_text = "Prediction details:\n"
    for i, result in enumerate(results):
        summary = result['interpretations'][0]['summary']
        legend_text += f"{i+1}: {summary[:60]}...\n"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, legend_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save or show the plot
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{ticker}_prediction_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def print_predictions(results, ticker):
    """Print the predictions in a readable format."""
    print(f"\n===== {ticker} Predictions =====")
    print("Date\t\tPred.\tConf.\tSummary")
    print("-" * 80)
    
    for result in results:
        date = result['date'].strftime('%Y-%m-%d')
        direction = "UP" if result['predictions']['direction'] > 0.5 else "DOWN"
        confidence = result['predictions']['direction'] if direction == "UP" else 1 - result['predictions']['direction']
        confidence = f"{confidence:.2f}"
        summary = result['interpretations'][0]['summary']
        
        print(f"{date}\t{direction}\t{confidence}\t{summary[:50]}...")
    
    print("-" * 80)


def main():
    """Main function to run inference."""
    print("VectorFin Inference")
    print("==================")
    
    # Parse arguments
    args = parse_args()
    
    # Fetch recent market data
    market_data = fetch_recent_market_data(args.ticker, args.days)
    if market_data is None or len(market_data) == 0:
        print("No market data available. Exiting.")
        return
    
    # Get recent news (synthetic for this example)
    news_data = get_recent_news(args.ticker, args.days)
    
    # Load VectorFin system
    system = load_vectorfin_system(args)
    
    # Run inference
    results = run_inference(system, market_data, news_data)
    
    # Print predictions
    print_predictions(results, args.ticker)
    
    # Visualize predictions
    visualize_predictions(results, args.ticker, args)
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
