#!/bin/bash
# filepath: /Users/jonathanwallace/vectorfin/run_daily_prediction.sh

# Script to run VectorFin predictions on a daily basis
# To be scheduled via cron job

# Set variables
DATE_TODAY=$(date +%Y-%m-%d)
TIME_NOW=$(date +%H:%M:%S)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$SCRIPT_DIR/prediction_outputs"

# Ensure log and output directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$LOG_DIR/prediction_$DATE_TODAY.log"

# Start logging
echo "===== VectorFin Daily Prediction =====" > "$LOG_FILE"
echo "Date: $DATE_TODAY" >> "$LOG_FILE"
echo "Time: $TIME_NOW" >> "$LOG_FILE"
echo "===================================" >> "$LOG_FILE"

# Navigate to script directory
cd "$SCRIPT_DIR" || { echo "Failed to navigate to script directory" >> "$LOG_FILE"; exit 1; }

# Activate virtual environment if needed (uncomment and modify if using venv)
# source venv/bin/activate >> "$LOG_FILE" 2>&1 || { echo "Failed to activate virtual environment" >> "$LOG_FILE"; exit 1; }

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Error: .env file not found. Ensure it contains NEWS_API_KEY." >> "$LOG_FILE"
    exit 1
fi

# Run the prediction script
echo "Running prediction script..." >> "$LOG_FILE"
python3 "$SCRIPT_DIR/examples/interact_with_model.py" >> "$LOG_FILE" 2>&1

# Check if prediction was successful
if [ $? -eq 0 ]; then
    echo "Prediction completed successfully." >> "$LOG_FILE"
    
    # Save prediction results to dated file if needed
    # cp "$SCRIPT_DIR/prediction_results.json" "$OUTPUT_DIR/prediction_$DATE_TODAY.json"
    
    # Optional: Send notification email
    # mail -s "VectorFin Daily Prediction Complete" your.email@example.com < "$LOG_FILE"
else
    echo "Error: Prediction script failed." >> "$LOG_FILE"
    
    # Optional: Send error notification
    # mail -s "VectorFin Daily Prediction Failed" your.email@example.com < "$LOG_FILE"
fi

echo "===================================" >> "$LOG_FILE"
echo "Process completed at $(date +%H:%M:%S)" >> "$LOG_FILE"

# Deactivate virtual environment if activated (uncomment if using venv)
# deactivate

exit 0
