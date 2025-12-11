#!/bin/bash
# Run both zero-shot experiments sequentially
# Experiment 1: Univariate (no covariates)
# Experiment 2: Multivariate (Crude Oil + Copper)

# Change to the zeroshot experiments directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================="
echo "Zero-shot Experiments"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Run Experiment 1: Univariate
echo "Starting Experiment 1: Univariate (No Covariates)"
echo "----------------------------------"
python rolling_forecast.py univariate/config.yaml
if [ $? -ne 0 ]; then
    echo "Error: Experiment 1 failed!"
    exit 1
fi
echo ""
echo "Experiment 1 completed successfully!"
echo ""

# Wait a bit before starting next experiment
sleep 2

# Run Experiment 2: Multivariate
echo "Starting Experiment 2: Multivariate (Oil + Copper)"
echo "----------------------------------"
python rolling_forecast.py multivariate/config.yaml
if [ $? -ne 0 ]; then
    echo "Error: Experiment 2 failed!"
    exit 1
fi
echo ""
echo "Experiment 2 completed successfully!"
echo ""

echo "=================================="
echo "All experiments completed!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - Experiment 1 (univariate): ./univariate/results/"
echo "  - Experiment 2 (multivariate): ./multivariate/results/"
echo ""
