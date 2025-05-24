#!/bin/zsh

# VectorFin Training Script
# This script runs the VectorFin training pipeline with common configurations

# Default parameters
TICKERS="AAPL,MSFT,GOOGL,AMZN,META"
START_DATE="2022-01-01"
END_DATE="2023-01-01"
MODELS_DIR="./trained_models"
EPOCHS=5
BATCH_SIZE=16

# Print banner
echo "================================"
echo "VectorFin Training Process"
echo "================================"
echo ""

# Create models directory if it doesn't exist
mkdir -p $MODELS_DIR

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking requirements..."
python3 -c "import torch, pandas, numpy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Run the training pipeline with synthetic data
echo "\nStarting training with synthetic data..."
echo "- Tickers: $TICKERS"
echo "- Date range: $START_DATE to $END_DATE" 
echo "- Models will be saved to: $MODELS_DIR"
echo "- Training for $EPOCHS epochs with batch size $BATCH_SIZE"
echo ""

python3 examples/training_pipeline.py \
    --tickers $TICKERS \
    --start_date $START_DATE \
    --end_date $END_DATE \
    --models_dir $MODELS_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_synthetic_news

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "\n================================"
    echo "Training completed successfully!"
    echo "Trained models are saved in: $MODELS_DIR"
    echo "================================"
else
    echo "\n================================"
    echo "Training failed. Check error messages above."
    echo "================================"
fi
