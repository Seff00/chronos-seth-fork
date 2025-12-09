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
PREDICTION_LENGTH = 1           # Predict 1 day ahead
VALIDATION_SPLIT = 0.2          # 20% for validation

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
    Prepare training and validation data for Chronos-2.

    Returns list of dicts in the format expected by pipeline.fit().
    """
    total_points = len(combined_data)

    # Filter to recent years if configured
    if USE_RECENT_YEARS_ONLY:
        days_to_use = RECENT_YEARS * 365
        if total_points > days_to_use:
            combined_data = combined_data.tail(days_to_use)
            print(f"\nUsing only recent {RECENT_YEARS} years: {len(combined_data)} days")

    # Split into train and validation
    val_size = int(len(combined_data) * VALIDATION_SPLIT)
    train_size = len(combined_data) - val_size

    train_data = combined_data.iloc[:train_size]
    val_data = combined_data.iloc[train_size:]

    print(f"\n" + "="*80)
    print("DATA SPLIT")
    print("="*80)
    print(f"Total data points:       {len(combined_data)}")
    print(f"Training set:            {len(train_data)} days ({len(train_data)/len(combined_data)*100:.1f}%)")
    print(f"  From: {train_data.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:   {train_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Validation set:          {len(val_data)} days ({len(val_data)/len(combined_data)*100:.1f}%)")
    print(f"  From: {val_data.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:   {val_data.index[-1].strftime('%Y-%m-%d')}")
    print("="*80)

    # Convert to format for pipeline.fit()
    # Each entry is a dict with 'target' and 'past_covariates'
    train_inputs = []
    val_inputs = []

    # Get covariate column names
    covariate_columns = [col for col in combined_data.columns if col != 'target']

    # Build training samples (sliding window approach)
    # For each possible context window, create a training sample
    for i in range(CONTEXT_LENGTH, len(train_data)):
        context = train_data.iloc[i-CONTEXT_LENGTH:i]

        # Build past covariates dict
        past_covariates = {}
        for col in covariate_columns:
            key = col.lower().replace(' ', '_')
            past_covariates[key] = context[col].values

        train_inputs.append({
            'target': context['target'].values,
            'past_covariates': past_covariates
        })

    # Build validation samples
    for i in range(CONTEXT_LENGTH, len(val_data)):
        context = val_data.iloc[i-CONTEXT_LENGTH:i]

        past_covariates = {}
        for col in covariate_columns:
            key = col.lower().replace(' ', '_')
            past_covariates[key] = context[col].values

        val_inputs.append({
            'target': context['target'].values,
            'past_covariates': past_covariates
        })

    print(f"\nGenerated {len(train_inputs)} training samples (sliding windows)")
    print(f"Generated {len(val_inputs)} validation samples")

    return train_inputs, val_inputs

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

    # Prepare training data
    train_inputs, val_inputs = prepare_training_data(combined_data)

    # Recommendation on training data size
    total_years = len(combined_data) / 365
    print(f"\n" + "="*80)
    print("TRAINING DATA RECOMMENDATIONS")
    print("="*80)
    print(f"You have {total_years:.1f} years of data ({len(combined_data)} days)")
    print()
    if total_years > 7:
        print("RECOMMENDATION: Consider using only recent 5-7 years for training.")
        print("  Reason: Market dynamics change over time. Recent data is more relevant.")
        print("  Current setting: USE_RECENT_YEARS_ONLY = False (using all data)")
        print()
        print("  To use recent years only, set:")
        print("    USE_RECENT_YEARS_ONLY = True")
        print("    RECENT_YEARS = 5  # or 7")
    else:
        print("Using all available data - good choice for your dataset size.")
    print("="*80)

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
    print("(This may take 15-25 minutes on RTX 4050)")

    # Fine-tune with LoRA
    try:
        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            validation_inputs=val_inputs,
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
        print(f"Model saved to: {OUTPUT_DIR}")
        print()
        print("To use the fine-tuned model in your rolling/multivariate scripts:")
        print()
        print("  1. Change MODEL_NAME:")
        print(f"     MODEL_NAME = '{OUTPUT_DIR}'")
        print()
        print("  2. Run your inference scripts as normal:")
        print("     python experiments_zeroshot/rolling/forecast_rolling.py")
        print("     python experiments_zeroshot/multivariate/forecast_multivariate.py")
        print()
        print("Next steps:")
        print("  - Compare fine-tuned metrics vs zero-shot baseline")
        print("  - Fine-tuned model should have better accuracy on your specific data")
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
