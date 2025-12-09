"""
LoRA Fine-tuning for Cotton Futures Forecasting with Chronos-2.
Uses Crude Oil and Copper as covariates.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from data_split import split_train_test, prepare_training_samples

# Configuration
DATA_FOLDER = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw"
TARGET_FILE = "Cotton_Futures.csv"
TARGET_NAME = "Cotton Futures"

# Covariates: Add/remove as needed
COVARIATES = [
    {"file": "Crude_Oil.csv", "name": "Crude Oil"},
    {"file": "Copper_Futures.csv", "name": "Copper Futures"},
]

# Model Configuration
MODEL_NAME = "amazon/chronos-2"
OUTPUT_DIR = "./experiments_finetune/lora/checkpoint"

# Fine-tuning Hyperparameters
FINETUNE_MODE = "lora"          # LoRA fine-tuning
LEARNING_RATE = 1e-5            # Higher LR for LoRA (recommended)
NUM_STEPS = 500                 # Training steps
BATCH_SIZE = 256                # Default batch size (reduce if OOM)
CONTEXT_LENGTH = 365            # Use last year (365 days) as context
PREDICTION_LENGTH = 7           # Predict 7 day ahead

# Training Data Configuration
USE_RECENT_YEARS_ONLY = False   # Set True to use only recent 5-7 years
RECENT_YEARS = 5                # If USE_RECENT_YEARS_ONLY=True, how many years

def load_csv_data(filepath, asset_name):
    """Load and preprocess commodity CSV data."""
    df = pd.read_csv(filepath, header=0, skiprows=[1, 2], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df = df.sort_index()
    prices = df['Close'].rename(asset_name)

    print(f"{asset_name:15} - {len(prices):5} points, "
          f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}, "
          f"range: ${prices.min():.2f} to ${prices.max():.2f}")

    return prices

def align_data(target, covariates_dict):
    """Align target and all covariate time series to common dates."""
    data_dict = {'target': target}
    data_dict.update(covariates_dict)
    combined = pd.DataFrame(data_dict)

    print(f"\nBefore alignment: {len(combined)} dates")
    print(f"Missing values:")
    print(f"  Target: {combined['target'].isna().sum()}")
    for name in covariates_dict.keys():
        print(f"  {name}: {combined[name].isna().sum()}")

    combined = combined.fillna(method='ffill').dropna()

    print(f"After alignment:  {len(combined)} dates")
    print(f"Date range: {combined.index.min().strftime('%Y-%m-%d')} to {combined.index.max().strftime('%Y-%m-%d')}")

    return combined

def prepare_training_data(combined_data):
    """
    Prepare training and test data for Chronos-2.

    Returns (train_inputs, test_data) where:
    - train_inputs: list of dicts for pipeline.fit()
    - test_data: DataFrame for hold-out evaluation
    """
    # Split data chronologically: train / test
    train_data, test_data = split_train_test(
        combined_data,
        use_recent_years_only=USE_RECENT_YEARS_ONLY,
        recent_years=RECENT_YEARS,
    )

    # Generate sliding window samples for training
    train_inputs = prepare_training_samples(
        train_data,
        context_length=CONTEXT_LENGTH,
        sample_name="training samples"
    )

    return train_inputs, test_data

def main():
    print("="*80)
    print(f"LoRA Fine-Tuning: {TARGET_NAME} Forecasting with Chronos-2")
    covariate_names = ', '.join([cov['name'] for cov in COVARIATES])
    print(f"Covariates: {covariate_names}")
    print("="*80)

    # Load target
    print("\nLoading datasets...")
    target_path = os.path.join(DATA_FOLDER, TARGET_FILE)
    target = load_csv_data(target_path, TARGET_NAME)

    # Load covariates
    covariates_dict = {}
    for cov in COVARIATES:
        cov_path = os.path.join(DATA_FOLDER, cov['file'])
        covariates_dict[cov['name']] = load_csv_data(cov_path, cov['name'])

    # Align data
    combined_data = align_data(target, covariates_dict)

    # Prepare training data (test_data is held out for later evaluation)
    train_inputs, test_data = prepare_training_data(combined_data)

    # Load pretrained model
    print(f"\nLoading pretrained model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Print fine-tuning configuration
    print(f"\n" + "="*80)
    print("FINE-TUNING CONFIGURATION")
    print("="*80)
    print(f"Mode:              {FINETUNE_MODE.upper()}")
    print(f"Learning Rate:     {LEARNING_RATE}")
    print(f"Training Steps:    {NUM_STEPS}")
    print(f"Batch Size:        {BATCH_SIZE}")
    print(f"Context Length:    {CONTEXT_LENGTH} days")
    print(f"Prediction Length: {PREDICTION_LENGTH} day(s)")
    print(f"Output Directory:  {OUTPUT_DIR}")
    print("="*80)

    print("\nStarting LoRA fine-tuning...")
    print("NOTE: Training WITHOUT validation (no early stopping)")

    # Fine-tune with LoRA
    try:
        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            validation_inputs=None,  # No validation
            prediction_length=PREDICTION_LENGTH,
            finetune_mode=FINETUNE_MODE,
            learning_rate=LEARNING_RATE,
            num_steps=NUM_STEPS,
            batch_size=BATCH_SIZE,
            context_length=CONTEXT_LENGTH,
            output_dir=OUTPUT_DIR,
        )

        print("\n" + "="*80)
        print("FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Model saved to: {OUTPUT_DIR}/finetuned-ckpt")
        print()
        print(f"Test set (HOLD-OUT): {len(test_data)} days from {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
        print("  → This data was NOT used in training or validation")
        print("  → Use this for final evaluation")
        print()
        print("Next steps:")
        print("  1. Evaluate on the hold-out test set using forecast_direct.py or forecast_rolling.py")
        print("  2. Compare fine-tuned vs zero-shot performance")
        print()
        print("Note: Update MODEL_NAME in forecast scripts to:")
        print(f"  MODEL_NAME = '{OUTPUT_DIR}/finetuned-ckpt'")
        print("="*80)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "="*80)
            print("OUT OF MEMORY ERROR!")
            print("="*80)
            print("Your GPU ran out of VRAM. Try these solutions:")
            print()
            print("1. Reduce BATCH_SIZE:")
            print("   BATCH_SIZE = 128  # or 64")
            print()
            print("2. Reduce CONTEXT_LENGTH:")
            print("   CONTEXT_LENGTH = 256  # or 180")
            print()
            print("3. Both:")
            print("   BATCH_SIZE = 128")
            print("   CONTEXT_LENGTH = 256")
            print("="*80)
        raise

if __name__ == "__main__":
    main()
