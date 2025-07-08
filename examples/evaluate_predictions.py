#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VectorFin Prediction Evaluation

This script evaluates VectorFin predictions by comparing them with actual market outcomes.
It helps verify if the model's statistics are meaningful by calculating various accuracy metrics
and running statistical tests to assess significance.

Usage:
    python evaluate_predictions.py --days_back 30
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from datetime import datetime, timedelta
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
import yfinance as yf

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components if needed
from vectorfin.src.data.data_loader import MarketData


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VectorFin predictions against actual market outcomes.")
    
    parser.add_argument("--predictions_file", type=str, 
                        default=None,
                        help="Path to predictions CSV file. Default is prediction_history.csv in prediction_outputs folder")
    
    parser.add_argument("--days_back", type=int, default=None,
                        help="Number of days to go back for evaluation (default: all available data)")
    
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (default: prediction_outputs/evaluation)")
    
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable generation of plots")
    
    return parser.parse_args()


def load_predictions(file_path, days_back=None):
    """
    Load prediction history from CSV file.
    
    Args:
        file_path: Path to the predictions CSV file
        days_back: Optional limit to only load predictions from the last N days
        
    Returns:
        DataFrame containing prediction history
    """
    print(f"Loading predictions from {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # Sort by date
    df = df.sort_values('date')
    
    # Filter by days_back if specified
    if days_back is not None:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df = df[df['date'] >= cutoff_date]
        print(f"Filtered to {len(df)} predictions in the last {days_back} days")
    
    print(f"Loaded {len(df)} predictions")
    return df


def fetch_actual_outcomes(predictions_df):
    """
    Fetch actual market data for the prediction periods.
    
    Args:
        predictions_df: DataFrame with prediction history
        
    Returns:
        DataFrame with predictions and actual outcomes
    """
    results_df = predictions_df.copy()
    
    # Add columns for actual outcomes
    results_df['actual_direction'] = np.nan
    results_df['actual_magnitude'] = np.nan
    results_df['actual_volatility'] = np.nan
    
    # Process each prediction
    for idx, row in results_df.iterrows():
        tickers = row['tickers'].split(',')
        prediction_date = row['date']
        horizon = row['prediction_horizon']
        target_date = prediction_date + timedelta(days=horizon)
        
        # Account for weekends and holidays by extending the search range
        extended_date = target_date + timedelta(days=5)
        
        print(f"\nProcessing prediction from {prediction_date.strftime('%Y-%m-%d')} "
              f"for {horizon} days ahead (target: {target_date.strftime('%Y-%m-%d')})")
        
        try:
            # Use yfinance to fetch historical data
            market_data = {}
            for ticker in tickers:
                # Fetch a window of data around the target date to handle market closures
                ticker_data = yf.download(ticker, 
                                         start=prediction_date - timedelta(days=5),
                                         end=extended_date + timedelta(days=5),
                                         progress=False)
                
                if not ticker_data.empty:
                    market_data[ticker] = ticker_data
            
            if not market_data:
                print(f"  Warning: No market data available for {tickers}")
                continue
                
            # Get data points for prediction date and target date
            start_prices = {}
            end_prices = {}
            volatilities = {}
            
            for ticker, data in market_data.items():
                # Find closest trading day to prediction date
                start_date = find_closest_trading_day(data, prediction_date)
                if start_date is None:
                    continue
                    
                # Find closest trading day to target date
                end_date = find_closest_trading_day(data, target_date)
                if end_date is None:
                    continue
                
                # Store prices
                start_prices[ticker] = data.loc[start_date, 'Close']
                end_prices[ticker] = data.loc[end_date, 'Close']
                
                # Calculate realized volatility over prediction period
                period_data = data[data.index >= start_date]
                period_data = period_data[period_data.index <= end_date]
                
                if len(period_data) >= 2:
                    # Calculate realized volatility as std dev of returns
                    returns = period_data['Close'].pct_change().dropna()
                    volatilities[ticker] = returns.std() * 100  # Convert to percentage
            
            if not start_prices or not end_prices:
                print(f"  Warning: Could not find matching trading days for {prediction_date} or {target_date}")
                continue
                
            # Calculate average outcomes across tickers
            avg_start_price = np.mean(list(start_prices.values()))
            avg_end_price = np.mean(list(end_prices.values()))
            
            # Calculate actual direction (1 if up, 0 if down)
            actual_direction = 1.0 if avg_end_price > avg_start_price else 0.0
            
            # Calculate actual magnitude (percentage change)
            actual_magnitude = (avg_end_price - avg_start_price) / avg_start_price * 100
            
            # Calculate actual volatility (average across tickers)
            actual_volatility = np.mean(list(volatilities.values())) if volatilities else np.nan
            
            # Store results
            results_df.at[idx, 'actual_direction'] = actual_direction
            results_df.at[idx, 'actual_magnitude'] = actual_magnitude
            results_df.at[idx, 'actual_volatility'] = actual_volatility
            
            print(f"  Results: Direction {'UP' if actual_direction else 'DOWN'}, "
                  f"Magnitude {actual_magnitude:.2f}%, Volatility {actual_volatility:.2f}%")
            
        except Exception as e:
            print(f"  Error processing prediction: {str(e)}")
    
    return results_df


def find_closest_trading_day(data, target_date):
    """Find the closest trading day to the target date."""
    # Convert target_date to date only if it's a datetime
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # Look for exact match first
    available_dates = [d.date() if isinstance(d, datetime) else d for d in data.index]
    
    if target_date in available_dates:
        return target_date
    
    # Look for closest date within a window (max 10 days forward)
    for i in range(1, 10):
        next_date = target_date + timedelta(days=i)
        if next_date in available_dates:
            return next_date
    
    # If not found, return None
    return None


def calculate_metrics(results_df):
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        results_df: DataFrame with predictions and actual outcomes
        
    Returns:
        Dictionary of metrics
    """
    # Filter out rows with missing actual values
    valid_df = results_df.dropna(subset=['actual_direction', 'actual_magnitude', 'actual_volatility'])
    
    if len(valid_df) == 0:
        print("Warning: No valid prediction-outcome pairs found for evaluation")
        return {}
        
    print(f"\nCalculating metrics based on {len(valid_df)} valid predictions")
    
    # Convert direction probabilities to binary predictions using 0.5 threshold
    valid_df['pred_direction_binary'] = (valid_df['direction'] > 0.5).astype(float)
    
    metrics = {}
    
    # Direction metrics
    metrics['direction_accuracy'] = accuracy_score(valid_df['actual_direction'], valid_df['pred_direction_binary'])
    
    try:
        metrics['direction_precision'] = precision_score(valid_df['actual_direction'], valid_df['pred_direction_binary'])
    except:
        metrics['direction_precision'] = np.nan
        
    try:
        metrics['direction_recall'] = recall_score(valid_df['actual_direction'], valid_df['pred_direction_binary'])
    except:
        metrics['direction_recall'] = np.nan
    
    # Magnitude metrics
    metrics['magnitude_mse'] = mean_squared_error(valid_df['actual_magnitude'], valid_df['magnitude'])
    metrics['magnitude_rmse'] = np.sqrt(metrics['magnitude_mse'])
    metrics['magnitude_mae'] = np.mean(np.abs(valid_df['actual_magnitude'] - valid_df['magnitude']))
    
    try:
        metrics['magnitude_r2'] = r2_score(valid_df['actual_magnitude'], valid_df['magnitude'])
    except:
        metrics['magnitude_r2'] = np.nan
    
    # Volatility metrics
    metrics['volatility_mse'] = mean_squared_error(valid_df['actual_volatility'], valid_df['volatility'])
    metrics['volatility_rmse'] = np.sqrt(metrics['volatility_mse'])
    metrics['volatility_mae'] = np.mean(np.abs(valid_df['actual_volatility'] - valid_df['volatility']))
    
    try:
        metrics['volatility_r2'] = r2_score(valid_df['actual_volatility'], valid_df['volatility'])
    except:
        metrics['volatility_r2'] = np.nan
    
    # Direction correlation (point-biserial)
    metrics['direction_correlation'], metrics['direction_pvalue'] = stats.pointbiserialr(
        valid_df['actual_direction'], valid_df['direction'])
    
    # Magnitude correlation
    metrics['magnitude_correlation'], metrics['magnitude_pvalue'] = stats.pearsonr(
        valid_df['actual_magnitude'], valid_df['magnitude'])
    
    # Volatility correlation
    metrics['volatility_correlation'], metrics['volatility_pvalue'] = stats.pearsonr(
        valid_df['actual_volatility'], valid_df['volatility'])
    
    return metrics, valid_df


def visualize_results(results_df, metrics, output_dir):
    """
    Create visualizations of prediction evaluation results.
    
    Args:
        results_df: DataFrame with predictions and actual outcomes
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\nGenerating visualizations in {output_path}")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('deep')
    
    # 1. Direction Prediction Accuracy
    plt.figure(figsize=(12, 8))
    
    # Create a confusion matrix for direction predictions
    valid_df = results_df.dropna(subset=['actual_direction'])
    if len(valid_df) > 0:
        cm = pd.crosstab(valid_df['actual_direction'], (valid_df['direction'] > 0.5), 
                          rownames=['Actual'], colnames=['Predicted'],
                          normalize='all')
        
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
        plt.title(f'Direction Prediction Confusion Matrix\nAccuracy: {metrics["direction_accuracy"]:.2f}', 
                  fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'direction_confusion_matrix.png', dpi=300)
        plt.close()
    
    # 2. Direction Probability vs Actual Outcome
    plt.figure(figsize=(12, 8))
    
    # Group predictions by probability bins
    valid_df = results_df.dropna(subset=['actual_direction'])
    if len(valid_df) > 0:
        valid_df['prob_bin'] = pd.cut(valid_df['direction'], bins=np.arange(0, 1.1, 0.1))
        bin_stats = valid_df.groupby('prob_bin')['actual_direction'].agg(['mean', 'count'])
        bin_stats = bin_stats.reset_index()
        
        # Plot calibration curve
        ax = plt.subplot(1, 1, 1)
        bin_centers = [(interval.left + interval.right) / 2 for interval in bin_stats['prob_bin']]
        
        # Actual outcome rate by predicted probability
        sns.barplot(x=bin_centers, y=bin_stats['mean'], ax=ax)
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Add count labels
        for i, (_, row) in enumerate(bin_stats.iterrows()):
            if not np.isnan(row['count']):
                plt.text(i, row['mean'] + 0.05, f'n={int(row["count"])}', 
                         ha='center', va='center', fontsize=9)
        
        plt.xlabel('Predicted probability of upward movement')
        plt.ylabel('Fraction of upward movements')
        plt.title('Direction Prediction Calibration\n(How well predicted probabilities match actual frequencies)',
                  fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'direction_calibration.png', dpi=300)
        plt.close()
    
    # 3. Magnitude Prediction vs Actual
    plt.figure(figsize=(12, 8))
    
    valid_df = results_df.dropna(subset=['actual_magnitude'])
    if len(valid_df) > 0:
        plt.scatter(valid_df['magnitude'], valid_df['actual_magnitude'], alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(valid_df['magnitude'].min(), valid_df['actual_magnitude'].min())
        max_val = max(valid_df['magnitude'].max(), valid_df['actual_magnitude'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
        
        # Add regression line
        if len(valid_df) > 1:  # Need at least 2 points for regression
            z = np.polyfit(valid_df['magnitude'], valid_df['actual_magnitude'], 1)
            p = np.poly1d(z)
            plt.plot(sorted(valid_df['magnitude']), p(sorted(valid_df['magnitude'])), 
                     'r-', label=f'Regression line (r={metrics["magnitude_correlation"]:.2f})')
        
        plt.xlabel('Predicted Magnitude (%)')
        plt.ylabel('Actual Magnitude (%)')
        plt.title(f'Magnitude Prediction vs Actual\nRMSE: {metrics["magnitude_rmse"]:.2f}%, '
                  f'Correlation: {metrics["magnitude_correlation"]:.2f}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'magnitude_prediction.png', dpi=300)
        plt.close()
    
    # 4. Volatility Prediction vs Actual
    plt.figure(figsize=(12, 8))
    
    valid_df = results_df.dropna(subset=['actual_volatility'])
    if len(valid_df) > 0:
        plt.scatter(valid_df['volatility'], valid_df['actual_volatility'], alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(valid_df['volatility'].min(), valid_df['actual_volatility'].min())
        max_val = max(valid_df['volatility'].max(), valid_df['actual_volatility'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
        
        # Add regression line
        if len(valid_df) > 1:  # Need at least 2 points for regression
            z = np.polyfit(valid_df['volatility'], valid_df['actual_volatility'], 1)
            p = np.poly1d(z)
            plt.plot(sorted(valid_df['volatility']), p(sorted(valid_df['volatility'])), 
                     'r-', label=f'Regression line (r={metrics["volatility_correlation"]:.2f})')
        
        plt.xlabel('Predicted Volatility (%)')
        plt.ylabel('Actual Volatility (%)')
        plt.title(f'Volatility Prediction vs Actual\nRMSE: {metrics["volatility_rmse"]:.2f}%, '
                  f'Correlation: {metrics["volatility_correlation"]:.2f}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'volatility_prediction.png', dpi=300)
        plt.close()
    
    # 5. Combined metrics dashboard
    plt.figure(figsize=(15, 10))
    
    # Create a table of metrics
    metric_names = [
        'Direction Accuracy', 'Direction Precision', 'Direction Recall',
        'Direction Correlation', 'Direction p-value',
        'Magnitude RMSE (%)', 'Magnitude MAE (%)', 'Magnitude R²', 
        'Magnitude Correlation', 'Magnitude p-value',
        'Volatility RMSE (%)', 'Volatility MAE (%)', 'Volatility R²',
        'Volatility Correlation', 'Volatility p-value'
    ]
    
    metric_values = [
        metrics.get('direction_accuracy', np.nan),
        metrics.get('direction_precision', np.nan),
        metrics.get('direction_recall', np.nan),
        metrics.get('direction_correlation', np.nan),
        metrics.get('direction_pvalue', np.nan),
        metrics.get('magnitude_rmse', np.nan),
        metrics.get('magnitude_mae', np.nan),
        metrics.get('magnitude_r2', np.nan),
        metrics.get('magnitude_correlation', np.nan),
        metrics.get('magnitude_pvalue', np.nan),
        metrics.get('volatility_rmse', np.nan),
        metrics.get('volatility_mae', np.nan),
        metrics.get('volatility_r2', np.nan),
        metrics.get('volatility_correlation', np.nan),
        metrics.get('volatility_pvalue', np.nan)
    ]
    
    # Format values
    metric_values_formatted = []
    for i, val in enumerate(metric_values):
        if np.isnan(val):
            metric_values_formatted.append('N/A')
        elif 'p-value' in metric_names[i]:
            metric_values_formatted.append(f"{val:.4f}")
        else:
            metric_values_formatted.append(f"{val:.2f}")
    
    # Create color coding for p-values
    colors = []
    for i, name in enumerate(metric_names):
        if 'p-value' in name and not np.isnan(metric_values[i]):
            if metric_values[i] < 0.01:
                colors.append('darkgreen')  # Highly significant
            elif metric_values[i] < 0.05:
                colors.append('green')  # Significant
            elif metric_values[i] < 0.1:
                colors.append('orange')  # Marginally significant
            else:
                colors.append('red')  # Not significant
        else:
            colors.append('black')
    
    # Create the table
    plt.axis('off')
    table = plt.table(
        cellText=[metric_values_formatted],
        rowLabels=['Value'],
        colLabels=metric_names,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.6, 1, 0.3]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Apply colors to p-value cells
    for i, color in enumerate(colors):
        table[(1, i)].get_text().set_color(color)
    
    plt.figtext(0.5, 0.9, 'VectorFin Prediction Evaluation Metrics', 
                ha='center', fontsize=18, weight='bold')
    
    # Add statistical significance explanation
    significance_text = (
        "Statistical Significance:\n"
        "• p < 0.01: Strong evidence against null hypothesis (dark green)\n"
        "• p < 0.05: Significant evidence against null hypothesis (green)\n"
        "• p < 0.10: Weak evidence against null hypothesis (orange)\n"
        "• p ≥ 0.10: No evidence against null hypothesis (red)\n\n"
        "Null hypothesis: There is no relationship between predictions and actual outcomes."
    )
    plt.figtext(0.5, 0.25, significance_text, ha='center', va='center', fontsize=10, 
                bbox=dict(facecolor='whitesmoke', alpha=0.5, boxstyle='round,pad=1'))
    
    # Add interpretation
    interpretation = interpret_metrics(metrics)
    plt.figtext(0.5, 0.45, interpretation, ha='center', va='center', fontsize=11,
                bbox=dict(facecolor='aliceblue', alpha=0.5, boxstyle='round,pad=1'))
    
    plt.tight_layout()
    plt.savefig(output_path / 'evaluation_summary.png', dpi=300)
    plt.close()
    
    print(f"Generated visualization files in {output_path}")


def interpret_metrics(metrics):
    """Generate an interpretation of the metrics."""
    if not metrics:
        return "Insufficient data to evaluate model performance."
    
    parts = []
    
    # Direction interpretation
    dir_acc = metrics.get('direction_accuracy', np.nan)
    dir_pval = metrics.get('direction_pvalue', np.nan)
    
    if not np.isnan(dir_acc):
        if dir_acc > 0.6:
            parts.append(f"Direction prediction is GOOD (accuracy: {dir_acc:.2f}).")
        elif dir_acc > 0.5:
            parts.append(f"Direction prediction is FAIR (accuracy: {dir_acc:.2f}).")
        else:
            parts.append(f"Direction prediction is POOR (accuracy: {dir_acc:.2f}).")
            
        if not np.isnan(dir_pval):
            if dir_pval < 0.05:
                parts.append(f"Direction predictions are statistically significant (p={dir_pval:.4f}).")
            else:
                parts.append(f"Direction predictions lack statistical significance (p={dir_pval:.4f}).")
    
    # Magnitude interpretation
    mag_corr = metrics.get('magnitude_correlation', np.nan)
    mag_pval = metrics.get('magnitude_pvalue', np.nan)
    
    if not np.isnan(mag_corr):
        if abs(mag_corr) > 0.5:
            parts.append(f"Magnitude predictions show STRONG correlation (r={mag_corr:.2f}).")
        elif abs(mag_corr) > 0.3:
            parts.append(f"Magnitude predictions show MODERATE correlation (r={mag_corr:.2f}).")
        elif abs(mag_corr) > 0.1:
            parts.append(f"Magnitude predictions show WEAK correlation (r={mag_corr:.2f}).")
        else:
            parts.append(f"Magnitude predictions show NO meaningful correlation (r={mag_corr:.2f}).")
            
        if not np.isnan(mag_pval):
            if mag_pval < 0.05:
                parts.append(f"Magnitude correlation is statistically significant (p={mag_pval:.4f}).")
            else:
                parts.append(f"Magnitude correlation lacks statistical significance (p={mag_pval:.4f}).")
    
    # Volatility interpretation
    vol_corr = metrics.get('volatility_correlation', np.nan)
    vol_pval = metrics.get('volatility_pvalue', np.nan)
    
    if not np.isnan(vol_corr):
        if abs(vol_corr) > 0.5:
            parts.append(f"Volatility predictions show STRONG correlation (r={vol_corr:.2f}).")
        elif abs(vol_corr) > 0.3:
            parts.append(f"Volatility predictions show MODERATE correlation (r={vol_corr:.2f}).")
        elif abs(vol_corr) > 0.1:
            parts.append(f"Volatility predictions show WEAK correlation (r={vol_corr:.2f}).")
        else:
            parts.append(f"Volatility predictions show NO meaningful correlation (r={vol_corr:.2f}).")
            
        if not np.isnan(vol_pval):
            if vol_pval < 0.05:
                parts.append(f"Volatility correlation is statistically significant (p={vol_pval:.4f}).")
            else:
                parts.append(f"Volatility correlation lacks statistical significance (p={vol_pval:.4f}).")
    
    # Overall assessment
    significant_components = 0
    if not np.isnan(dir_pval) and dir_pval < 0.05:
        significant_components += 1
    if not np.isnan(mag_pval) and mag_pval < 0.05:
        significant_components += 1
    if not np.isnan(vol_pval) and vol_pval < 0.05:
        significant_components += 1
    
    if significant_components == 3:
        overall = "STRONG evidence that improved stats are real."
    elif significant_components == 2:
        overall = "MODERATE evidence that improved stats are real."
    elif significant_components == 1:
        overall = "WEAK evidence that improved stats are real."
    else:
        overall = "NO statistical evidence that improved stats are real."
    
    parts.append(f"\nOVERALL ASSESSMENT: {overall}")
    
    return "\n".join(parts)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set default paths if not provided
    if args.predictions_file is None:
        args.predictions_file = str(Path(__file__).resolve().parent.parent / 
                                  "prediction_outputs" / "prediction_history.csv")
    
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).resolve().parent.parent / 
                             "prediction_outputs" / "evaluation")
    
    # Load predictions
    predictions_df = load_predictions(args.predictions_file, args.days_back)
    
    if predictions_df.empty:
        print("Error: No predictions found or prediction file is empty.")
        return
    
    # Fetch actual outcomes
    results_df = fetch_actual_outcomes(predictions_df)
    
    # Calculate metrics
    metrics, valid_df = calculate_metrics(results_df)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print("-------------------")
    if metrics:
        print(f"Direction Prediction:")
        print(f"  - Accuracy: {metrics.get('direction_accuracy', 'N/A'):.4f}")
        print(f"  - Precision: {metrics.get('direction_precision', 'N/A'):.4f}")
        print(f"  - Recall: {metrics.get('direction_recall', 'N/A'):.4f}")
        print(f"  - Correlation: {metrics.get('direction_correlation', 'N/A'):.4f}")
        print(f"  - p-value: {metrics.get('direction_pvalue', 'N/A'):.4f}")
        
        print(f"\nMagnitude Prediction:")
        print(f"  - RMSE: {metrics.get('magnitude_rmse', 'N/A'):.4f}")
        print(f"  - MAE: {metrics.get('magnitude_mae', 'N/A'):.4f}")
        print(f"  - R²: {metrics.get('magnitude_r2', 'N/A'):.4f}")
        print(f"  - Correlation: {metrics.get('magnitude_correlation', 'N/A'):.4f}")
        print(f"  - p-value: {metrics.get('magnitude_pvalue', 'N/A'):.4f}")
        
        print(f"\nVolatility Prediction:")
        print(f"  - RMSE: {metrics.get('volatility_rmse', 'N/A'):.4f}")
        print(f"  - MAE: {metrics.get('volatility_mae', 'N/A'):.4f}")
        print(f"  - R²: {metrics.get('volatility_r2', 'N/A'):.4f}")
        print(f"  - Correlation: {metrics.get('volatility_correlation', 'N/A'):.4f}")
        print(f"  - p-value: {metrics.get('volatility_pvalue', 'N/A'):.4f}")
        
        # Add interpretation
        print("\nInterpretation:")
        print("--------------")
        print(interpret_metrics(metrics))
    else:
        print("No metrics available. Check that predictions file contains valid data.")
    
    # Generate visualizations if not disabled
    if not args.no_plots and metrics:
        visualize_results(results_df, metrics, args.output_dir)
    
    # Save evaluation results
    if metrics:
        # Save full results DataFrame
        results_path = Path(args.output_dir) / "evaluation_results.csv"
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved detailed evaluation results to {results_path}")
        
        # Save metrics summary
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()})
        metrics_path = Path(args.output_dir) / "evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics summary to {metrics_path}")


if __name__ == "__main__":
    main()
