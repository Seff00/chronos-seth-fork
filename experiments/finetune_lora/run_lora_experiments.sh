#!/bin/bash
# Run both LoRA fine-tuning experiments sequentially
# Experiment 1: 2 covariates (Crude Oil + Copper)
# Experiment 2: All covariates

# Change to the finetune_lora experiments directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================="
echo "LoRA Fine-Tuning Experiments"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# # Run Experiment 1: 2 Covariates
# echo "Starting Experiment 1: 2 Covariates (Crude Oil + Copper)"
# echo "----------------------------------"
# echo "Step 1/2: Training..."
# python train.py 2covariates/config.yaml
# if [ $? -ne 0 ]; then
#     echo "Error: Experiment 1 training failed!"
#     exit 1
# fi
# echo ""
# echo "Step 2/2: Evaluating..."
# python evaluate.py 2covariates/config.yaml
# if [ $? -ne 0 ]; then
#     echo "Error: Experiment 1 evaluation failed!"
#     exit 1
# fi
# echo ""
# echo "Experiment 1 completed successfully!"
# echo ""

# Wait a bit before starting next experiment
# sleep 2

# Run Experiment 2: All Covariates
echo "Starting Experiment 2: All Covariates"
echo "----------------------------------"
echo "Step 1/2: Training..."
python train.py allcovariates/config.yaml
if [ $? -ne 0 ]; then
    echo "Error: Experiment 2 training failed!"
    exit 1
fi
echo ""
echo "Step 2/2: Evaluating..."
python evaluate.py allcovariates/config.yaml
if [ $? -ne 0 ]; then
    echo "Error: Experiment 2 evaluation failed!"
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
echo "  - Experiment 1 (2 covariates): ./2covariates/results/"
echo "  - Experiment 2 (all covariates): ./allcovariates/results/"
echo ""
