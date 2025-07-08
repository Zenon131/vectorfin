#!/bin/bash
# Start the VectorFin API server

# Ensure conda environment is activated if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "vectorfin"; then
        conda activate vectorfin
        echo "Activated conda environment 'vectorfin'"
    fi
fi

# Start the API server
echo "Starting VectorFin API server..."
python api.py
