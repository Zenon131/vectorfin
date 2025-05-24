#!/usr/bin/env python
# filepath: /Users/jonathanwallace/vectorfin/examples/visualize_predictions.py
"""
Script to visualize VectorFin prediction history.

This script loads the prediction history from the CSV file and plots
the direction, magnitude, and volatility predictions over time.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def visualize_prediction_history():
    """
    Visualize the prediction history from the CSV file.
    """
    # Find the prediction history file
    history_file = Path(__file__).resolve().parent.parent / "prediction_outputs" / "prediction_history.csv"
    
    if not history_file.exists():
        print(f"Prediction history file not found at {history_file}")
        return
    
    # Load the prediction history
    history_df = pd.read_csv(history_file)
    
    # Convert date column to datetime
    history_df['date'] = pd.to_datetime(history_df['date'], format='%Y%m%d')
    
    # Sort by date
    history_df = history_df.sort_values('date')
    
    # Create visualization directory if it doesn't exist
    viz_dir = Path(__file__).resolve().parent.parent / "visualization"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up the figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot direction probability
    ax1.plot(history_df['date'], history_df['direction'], marker='o', linestyle='-', color='blue')
    ax1.set_ylabel('Direction Probability')
    ax1.set_title('Probability of Upward Movement')
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)  # Add a reference line at 0.5
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot magnitude prediction
    ax2.plot(history_df['date'], history_df['magnitude'], marker='o', linestyle='-', color='green')
    ax2.set_ylabel('Magnitude (%)')
    ax2.set_title('Predicted Price Change Magnitude')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Add a reference line at 0
    ax2.grid(True, alpha=0.3)
    
    # Plot volatility prediction
    ax3.plot(history_df['date'], history_df['volatility'], marker='o', linestyle='-', color='purple')
    ax3.set_ylabel('Volatility')
    ax3.set_title('Predicted Market Volatility')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)  # Volatility is always positive
    
    # Format x-axis
    ax3.set_xlabel('Date')
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = viz_dir / "prediction_history.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    
    # Also save as PDF for higher quality
    pdf_path = viz_dir / "prediction_history.pdf"
    plt.savefig(pdf_path)
    print(f"PDF visualization saved to {pdf_path}")
    
    # Try to display the plot (if running in an interactive environment)
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    visualize_prediction_history()
