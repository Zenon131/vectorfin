"""
Example usage of the VectorFin system.

This script demonstrates how to use the VectorFin system for financial analysis
by combining textual and numerical market data.
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yfinance as yf
import sys

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem, VectorFinTrainer
from vectorfin.src.data.data_loader import FinancialTextData, MarketData


def download_market_data(tickers, start_date, end_date):
    """
    Download market data for demonstration.
    """
    print(f"Downloading market data for {tickers}...")
    market_data = MarketData.fetch_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"Downloaded data for {len(market_data)} tickers")
    
    # Preview the data
    for ticker, data in market_data.items():
        print(f"\n{ticker} data shape: {data.shape}")
        print(data.head(3))
    
    return market_data


def create_sample_news():
    """
    Create sample financial news for demonstration.
    """
    news_data = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=10),
        'headline': [
            "Company XYZ reports strong quarterly earnings, beating analyst expectations",
            "Federal Reserve signals possible interest rate cut later this year",
            "Market volatility increases amid global economic uncertainty",
            "Tech stocks plunge as inflation fears rattle investors",
            "Oil prices surge following supply disruptions in major producing regions",
            "Retail sales data shows stronger than expected consumer spending",
            "Company ABC announces major acquisition, shares jump 15%",
            "Cryptocurrency market experiences significant correction",
            "Housing market shows signs of cooling as mortgage rates climb",
            "Manufacturing sector expansion slows according to latest PMI data"
        ],
        'source': ['Financial Times', 'Wall Street Journal', 'Bloomberg', 'CNBC', 
                  'Reuters', 'Financial Times', 'Wall Street Journal', 'Bloomberg', 
                  'CNBC', 'Reuters']
    })
    
    print("\nSample news data:")
    print(news_data.head(3))
    
    return news_data


def run_sample_analysis(system, news_data, market_data):
    """
    Run a sample analysis with the VectorFin system.
    """
    # Select a ticker for analysis
    ticker = list(market_data.keys())[0]
    ticker_data = market_data[ticker]
    
    print(f"\nRunning analysis for {ticker}...")
    
    # Get news headlines
    texts = news_data['headline'].tolist()
    
    # Prepare market data for the latest date
    latest_market_data = ticker_data.tail(5)
    
    # Analyze text and market data
    analysis = system.analyze_text_and_market(texts, latest_market_data)
    
    # Print interpretations
    print("\nAnalysis results:")
    for i, interp in enumerate(analysis["interpretations"][:3]):  # Show first 3
        print(f"\nSample {i}:")
        print(interp["summary"])
    
    # Return the analysis for further processing
    return analysis


def visualize_results(system, news_data, market_data, analysis):
    """
    Visualize the results of the analysis.
    """
    # Select a ticker for visualization
    ticker = list(market_data.keys())[0]
    ticker_data = market_data[ticker]
    
    # Get news headlines
    texts = news_data['headline'].tolist()[:5]  # Use first 5 for visualization
    
    # Prepare market data
    latest_market_data = ticker_data.tail(5)
    
    # Visualize attention between text and market data
    try:
        print("\nGenerating attention visualization...")
        fig = system.visualize_attention(texts, latest_market_data)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating visualization: {e}")


def main():
    """
    Main function to demonstrate VectorFin usage.
    """
    print("VectorFin Demo")
    print("-------------")
    
    # Create VectorFin system
    print("\nInitializing VectorFin system...")
    system = VectorFinSystem(vector_dim=128)
    
    # Download sample market data
    tickers = ["AAPL", "MSFT"]
    market_data = download_market_data(
        tickers=tickers,
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    
    # Create sample news data
    news_data = create_sample_news()
    
    # Run analysis
    analysis = run_sample_analysis(system, news_data, market_data)
    
    # Visualize results
    visualize_results(system, news_data, market_data, analysis)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
