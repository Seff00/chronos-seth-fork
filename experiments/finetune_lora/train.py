"""
LoRA Fine-tuning for Cotton Futures Forecasting with Chronos-2.
Configuration loaded from config.yaml passed as argument.
Training only - use evaluate.py for inference.
"""

import os
import sys
import yaml
import pandas as pd
import torch
from chronos import Chronos2Pipeline

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_unified_data(config):
    """Load the unified commodity dataset."""
    print("Loading unified commodity dataset...")

    data_path = config['data']['unified_data_path']

    # Load all splits
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    # Combine all data (val is used as part of training data, not for validation)
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Convert Date to datetime and set as index
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.set_index('Date')
    combined_df = combined_df.sort_index()

    print(f"Total data: {len(combined_df)} days")
    print(f"Date range: {combined_df.index.min().strftime('%Y-%m-%d')} to {combined_df.index.max().strftime('%Y-%m-%d')}")

    return combined_df

def prepare_data_for_training(combined_df, config):
    """Prepare data with target and covariates."""
    target_column = config['data']['target_column']
    covariate_columns = config['data']['covariate_columns']

    # Select target and covariates
    columns_to_keep = [target_column] + covariate_columns
    data = combined_df[columns_to_keep].copy()

    # Rename target column to 'target' for consistency
    data = data.rename(columns={target_column: 'target'})

    print(f"\nPrepared data with {len(data)} days")
    print(f"Target: {target_column}")
    print(f"Covariates ({len(covariate_columns)}): {', '.join(covariate_columns)}")

    return data

def split_train_test(data, test_size):
    """Split data chronologically into train and test sets."""
    total_points = len(data)
    test_start_idx = total_points - test_size

    train_data = data.iloc[:test_start_idx]
    test_data = data.iloc[test_start_idx:]

    print("\n" + "="*80)
    print("DATA SPLIT SUMMARY")
    print("="*80)
    print(f"Total data points: {total_points} days")
    print(f"Training set: {len(train_data)} days ({len(train_data)/total_points*100:.1f}%)")
    print(f"  From: {train_data.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:   {train_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Test set: {len(test_data)} days ({len(test_data)/total_points*100:.1f}%)")
    print(f"  From: {test_data.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:   {test_data.index[-1].strftime('%Y-%m-%d')}")
    print("="*80)

    return train_data, test_data

def prepare_training_samples(data, context_length):
    """Create sliding window training samples."""
    samples = []
    covariate_columns = [col for col in data.columns if col != 'target']

    for i in range(context_length, len(data)):
        context = data.iloc[i-context_length:i]

        # Build past covariates dict
        past_covariates = {}
        for col in covariate_columns:
            key = col.lower().replace('_close', '').replace('_', '_')
            past_covariates[key] = context[col].values

        samples.append({
            'target': context['target'].values,
            'past_covariates': past_covariates
        })

    print(f"Generated {len(samples)} training samples (sliding windows with context_length={context_length})")
    return samples

def main():
    # Check for config file argument
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        print("Example: python train.py 2covariates/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    # Load configuration
    config = load_config(config_path)

    # Create output directories
    output_dir = config['output']['output_dir']
    checkpoint_dir = config['output']['checkpoint_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Redirect output to file
    output_file = os.path.join(output_dir, 'training_log.txt')
    original_stdout = sys.stdout

    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f

        print("="*80)
        print(f"{config['experiment']['name']}")
        print(f"{config['experiment']['description']}")
        print("="*80)

        # Load and prepare data
        combined_df = load_unified_data(config)
        data = prepare_data_for_training(combined_df, config)

        # Split into train and test
        prediction_days = config['forecast']['prediction_days']
        train_data, test_data = split_train_test(data, test_size=prediction_days)

        # Prepare training samples
        context_length = config['training']['context_length']
        train_inputs = prepare_training_samples(train_data, context_length)

        # Load pretrained model
        model_name = config['model']['name']
        print(f"\nLoading pretrained model: {model_name}")

        torch_dtype = getattr(torch, config['model']['torch_dtype'])
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=config['model']['device_map'],
            torch_dtype=torch_dtype,
        )

        # Print fine-tuning configuration
        print(f"\n" + "="*80)
        print("FINE-TUNING CONFIGURATION")
        print("="*80)
        print(f"Mode:              {config['training']['finetune_mode'].upper()}")
        print(f"Learning Rate:     {config['training']['learning_rate']}")
        print(f"Training Steps:    {config['training']['num_steps']}")
        print(f"Batch Size:        {config['training']['batch_size']}")
        print(f"Context Length:    {context_length} days")
        print(f"Prediction Length: {config['training']['prediction_length']} day(s)")
        print(f"Output Directory:  {checkpoint_dir}")
        print("="*80)

        print("\nStarting LoRA fine-tuning...")

        try:
            finetuned_pipeline = pipeline.fit(
                inputs=train_inputs,
                validation_inputs=config['training']['validation_inputs'],
                prediction_length=config['training']['prediction_length'],
                finetune_mode=config['training']['finetune_mode'],
                learning_rate=config['training']['learning_rate'],
                num_steps=config['training']['num_steps'],
                batch_size=config['training']['batch_size'],
                context_length=context_length,
                output_dir=checkpoint_dir,
            )

            print("\n" + "="*80)
            print("FINE-TUNING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Model checkpoint saved to: {checkpoint_dir}/finetuned-ckpt")
            print()
            print("Next steps:")
            print("  1. Run evaluation: python evaluate.py <config_path>")
            print(f"     Example: python evaluate.py {config_path}")
            print()
            print("="*80)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n" + "="*80)
                print("OUT OF MEMORY ERROR!")
                print("="*80)
                print("Try reducing batch_size or context_length in config.yaml")
                print("="*80)
            raise

    # Restore stdout
    sys.stdout = original_stdout
    print(f"Training log saved to: {output_file}")
    print(f"Model checkpoint saved to: {checkpoint_dir}/finetuned-ckpt")
    print(f"\nTo evaluate: python evaluate.py {config_path}")

if __name__ == "__main__":
    main()
