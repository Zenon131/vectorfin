#!/bin/bash
# This script runs a daily prediction using Alpha Vantage instead of NewsAPI

# Set up logging
TIMESTAMP=$(date +%Y-%m-%d)
LOG_FILE="./logs/prediction_$TIMESTAMP.log"
mkdir -p ./logs

echo "Starting daily prediction run at $(date)" | tee -a "$LOG_FILE"

# Check for Alpha Vantage API key in .env file
if [ -f .env ]; then
    source .env
    if [ -n "$ALPHA_VANTAGE_API_KEY" ]; then
        echo "Alpha Vantage API key found in .env file" | tee -a "$LOG_FILE"
    else
        echo "Error: ALPHA_VANTAGE_API_KEY not found in .env file." | tee -a "$LOG_FILE"
        echo "Please set this variable before running the script." | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "Error: .env file not found. Ensure it contains ALPHA_VANTAGE_API_KEY." | tee -a "$LOG_FILE"
    exit 1
fi

# Define tickers to analyze
TICKERS="AAPL MSFT GOOGL META AMZN"

# Run the prediction using the Alpha Vantage adapter
echo "Running prediction for tickers: $TICKERS" | tee -a "$LOG_FILE"
python examples/interact_with_alpha_vantage.py --tickers $TICKERS --days 10 --horizon 5 2>&1 | tee -a "$LOG_FILE"

# Check if the prediction was successful
if [ $? -eq 0 ]; then
    echo "Daily prediction completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "Error: Daily prediction failed at $(date)" | tee -a "$LOG_FILE"
fi

# Exit with success
exit 0
