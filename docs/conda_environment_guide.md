# Conda Environment Setup for VectorFin

This document explains how to set up and use the conda environment for the VectorFin project.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download/)

## Creating the Environment

The environment has already been created with all necessary dependencies. If you need to recreate it, follow these steps:

```bash
# Create a new conda environment named "vectorfin" with Python 3.10
conda create -n vectorfin python=3.10 -y

# Activate the environment
conda activate vectorfin

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Using the Environment

We've created a convenience script called `use_conda_env.sh` that activates the conda environment and runs your specified command:

```bash
# To run any Python script in the conda environment
./use_conda_env.sh python examples/basic_usage.py

# To run the daily prediction script
./use_conda_env.sh ./run_daily_prediction.sh

# To evaluate predictions
./use_conda_env.sh python examples/evaluate_predictions.py
```

## Manual Activation

If you prefer to manually activate the environment:

```bash
# Activate the environment
conda activate vectorfin

# Now you can run commands directly
python examples/basic_usage.py
```

## Environment Details

- Python version: 3.10
- Key packages:
  - PyTorch 2.6.0
  - Transformers 4.35.0+
  - Pandas, NumPy, Scikit-learn
  - yfinance for market data retrieval
  - Various NLP and ML libraries

## Verifying Model Statistics

To verify if the improved model statistics are real:

```bash
# Run the evaluation script
./use_conda_env.sh python examples/evaluate_predictions.py

# The script will:
# 1. Load historical predictions from prediction_history.csv
# 2. Fetch actual market data for the same periods
# 3. Calculate accuracy, RMSE, and other performance metrics
# 4. Generate visualizations and a comprehensive HTML report
```

The evaluation results will be saved in the `evaluation_results` directory, including an HTML report with visualizations and statistical analysis.
