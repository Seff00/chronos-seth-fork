#!/bin/bash
# Run both LoRA fine-tuning experiments sequentially
# Experiment 1: 2 covariates (Crude Oil + Copper)
# Experiment 2: All covariates

echo "=================================="
echo "LoRA Fine-Tuning Experiments"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Run Experiment 1: 2 Covariates
echo "Starting Experiment 1: 2 Covariates (Crude Oil + Copper)"
echo "----------------------------------"
cd 2covariates
python train.py
if [ $? -ne 0 ]; then
    echo "Error: Experiment 1 failed!"
    exit 1
fi
cd ..
echo ""
echo "Experiment 1 completed successfully!"
echo ""

# Wait a bit before starting next experiment
sleep 2

# Run Experiment 2: All Covariates
echo "Starting Experiment 2: All Covariates"
echo "----------------------------------"
cd allcovariates
python train.py
if [ $? -ne 0 ]; then
    echo "Error: Experiment 2 failed!"
    exit 1
fi
cd ..
echo ""
echo "Experiment 2 completed successfully!"
echo ""

echo "=================================="
echo "All experiments completed!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  - Experiment 1 (2 covariates): ./2covariates/results/"
echo "  - Experiment 2 (all covariates): ./allcovariates/results/"
echo ""
