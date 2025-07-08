"""
VectorFin API Client Example

This example demonstrates how to use the VectorFin API with custom LLM configuration.
"""

import requests
import json
import os
from typing import Dict, List, Optional, Union, Any

def get_prediction(
    api_key: str,
    api_url: str = "http://localhost:8000",
    tickers: List[str] = ["AAPL", "MSFT", "GOOGL"],
    prediction_horizon: int = 5,
    llm_config: Optional[Dict[str, Any]] = None,
    news_api_key: Optional[str] = None
):
    """
    Get a prediction from the VectorFin API.
    
    Args:
        api_key: API key for VectorFin API
        api_url: URL of the VectorFin API
        tickers: List of stock tickers to analyze
        prediction_horizon: Number of days to predict into the future
        llm_config: Optional custom LLM configuration
        news_api_key: Optional NewsAPI key
        
    Returns:
        API response as dictionary
    """
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "tickers": tickers,
        "prediction_horizon": prediction_horizon
    }
    
    # Add optional configurations
    if llm_config:
        data["llm_config"] = llm_config
    
    if news_api_key:
        data["news_api_key"] = news_api_key
    
    # Make the request
    response = requests.post(
        f"{api_url}/predict",
        headers=headers,
        json=data
    )
    
    # Handle response
    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API request failed with status code {response.status_code}: {response.text}"
        raise Exception(error_msg)


def main():
    """Run the example API client."""
    # Get API key from environment or user input
    api_key = os.getenv("VECTORFIN_API_KEY")
    if not api_key:
        api_key = input("Enter your VectorFin API key: ")
    
    # Example with default LLM configuration
    print("Making prediction with default LLM configuration...")
    try:
        result = get_prediction(api_key)
        print(f"Prediction for {result['tickers']} ({result['date']}):")
        print(f"Direction: {result['predictions']['direction']:.2f}")
        print(f"Magnitude: {result['predictions']['magnitude']:.2f}%")
        print(f"Volatility: {result['predictions']['volatility']:.2f}")
        print("\nInterpretation:")
        print(result["interpretation"][:500] + "...")  # First 500 chars
    except Exception as e:
        print(f"Error with default configuration: {str(e)}")
    
    # Example with custom LLM configuration
    print("\nMaking prediction with custom LLM configuration...")
    try:
        # Replace these with your own LLM API details
        custom_llm_config = {
            "api_url": "https://api.openai.com/v1/chat/completions",
            "model_name": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.2,
            "connect_timeout": 15,
            "read_timeout": 180
        }
        
        # Only proceed if we have an OpenAI API key
        if not custom_llm_config["api_key"]:
            print("Skipping custom LLM example (no OPENAI_API_KEY environment variable found)")
            return
        
        result = get_prediction(
            api_key, 
            llm_config=custom_llm_config,
            tickers=["TSLA", "NVDA"]  # Different tickers for variety
        )
        
        print(f"Prediction for {result['tickers']} ({result['date']}):")
        print(f"Direction: {result['predictions']['direction']:.2f}")
        print(f"Magnitude: {result['predictions']['magnitude']:.2f}%")
        print(f"Volatility: {result['predictions']['volatility']:.2f}")
        print("\nInterpretation:")
        print(result["interpretation"][:500] + "...")  # First 500 chars
    except Exception as e:
        print(f"Error with custom configuration: {str(e)}")


if __name__ == "__main__":
    main()
