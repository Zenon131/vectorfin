#!/bin/bash
# Script to activate conda environment and run VectorFin

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate vectorfin

# Display environment information
echo "Using Python: $(python --version)"
echo "Environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Execute the specified command or script
if [ $# -eq 0 ]; then
    # No arguments provided, show usage
    echo "Usage: $0 [command_or_script]"
    echo "Examples:"
    echo "  $0 python examples/evaluate_predictions.py"
    echo "  $0 python examples/basic_usage.py"
    echo "  $0 ./run_daily_prediction.sh"
else
    # Execute the provided command with all arguments
    echo "Running: $@"
    exec "$@"
fi
