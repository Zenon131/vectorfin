#!/usr/bin/env python
# filepath: /Users/jonathanwallace/vectorfin/examples/save_predictions.py
"""
Script to save VectorFin predictions and interpretations to files for later analysis.
This extends the interact_with_model.py functionality to store prediction history.
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def save_prediction_results(prediction_results, interpretation, tickers):
    """
    Save prediction results and LLM interpretation to dated files.
    
    Args:
        prediction_results: Dictionary containing prediction data
        interpretation: String containing LLM interpretation
        tickers: List of tickers that were analyzed
    """
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).resolve().parent.parent / "prediction_outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get current date for filename
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Create filenames
    prediction_file = output_dir / f"prediction_{current_date}.json"
    interpretation_file = output_dir / f"interpretation_{current_date}.txt"
    
    # Save prediction results as JSON
    combined_results = {
        "date": current_date,
        "tickers": tickers,
        "prediction_results": prediction_results,
    }
    
    with open(prediction_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # Save interpretation as text
    with open(interpretation_file, 'w') as f:
        f.write(f"VectorFin Prediction Interpretation\n")
        f.write(f"Date: {current_date}\n")
        f.write(f"Tickers: {', '.join(tickers)}\n")
        f.write(f"{'='*50}\n\n")
        f.write(interpretation)
    
    print(f"Prediction results saved to {prediction_file}")
    print(f"Interpretation saved to {interpretation_file}")
    
    # Append to prediction history CSV for tracking over time
    history_file = output_dir / "prediction_history.csv"
    
    # Extract key metrics from prediction
    history_entry = {
        "date": current_date,
        "tickers": ",".join(tickers),
        "prediction_horizon": prediction_results.get("prediction_horizon", "N/A"),
        "direction": prediction_results.get("predictions", {}).get("direction", "N/A"),
        "magnitude": prediction_results.get("predictions", {}).get("magnitude", "N/A"),
        "volatility": prediction_results.get("predictions", {}).get("volatility", "N/A")
    }
    
    # Check if history file exists
    if history_file.exists():
        # Append to existing history
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, pd.DataFrame([history_entry])], ignore_index=True)
    else:
        # Create new history file
        history_df = pd.DataFrame([history_entry])
    
    # Save updated history
    history_df.to_csv(history_file, index=False)
    print(f"Prediction history updated in {history_file}")


if __name__ == "__main__":
    print("This is a utility module to be imported by interact_with_model.py")
    print("To run predictions, use: python interact_with_model.py")
