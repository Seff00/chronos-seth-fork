"""
Covariate importance analysis using ablation study for LoRA fine-tuned model.
Removes covariates one-by-one to measure their impact on forecast accuracy.
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

# Configuration
DATA_FOLDER = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw"
TARGET_FILE = "Cotton_Futures.csv"
TARGET_NAME = "Cotton Futures"

# Covariates: Add/remove as needed
COVARIATES = [
    {"file": "Crude_Oil.csv", "name": "Crude Oil"},
    {"file": "Copper_Futures.csv", "name": "Copper Futures"},
    {"file": "SP500.csv", "name": "SP500"},
    {"file": "Dollar_Index.csv", "name": "Dollar Index"},
]

# Cotton Futures internal features (from same CSV as target)
COTTON_FEATURES = ["High", "Low", "Open", "Volume"]

MODEL_NAME = "experiments_finetune/lora/checkpoint/finetuned-ckpt"
PREDICTION_LENGTH = 7
CONTEXT_LENGTH = 365
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Test on last N days
TEST_SET_SIZE = 7

def load_csv_data(filepath, asset_name, column='Close'):
    """Load and preprocess commodity CSV data."""
    df = pd.read_csv(filepath, header=0, skiprows=[1, 2], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df = df.sort_index()
    prices = df[column].rename(asset_name)
    return prices

def load_cotton_features(filepath, feature_columns):
    """Load multiple feature columns from Cotton Futures CSV."""
    df = pd.read_csv(filepath, header=0, skiprows=[1, 2], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df = df.sort_index()

    features_dict = {}
    for col in feature_columns:
        features_dict[f"Cotton {col}"] = df[col]

    return features_dict

def align_data(target, covariates_dict):
    """Align target and all covariate time series to common dates."""
    data_dict = {'target': target}
    data_dict.update(covariates_dict)
    combined = pd.DataFrame(data_dict)
    combined = combined.fillna(method='ffill').dropna()
    return combined

def run_experiment(pipeline, combined_data, test_start_idx, covariate_config, all_covariate_names):
    """
    Run inference with specific covariate configuration.

    Parameters
    ----------
    covariate_config : dict
        Keys: covariate names (sanitized)
        Values: True (include) or False (exclude)
    all_covariate_names : list
        List of all available covariate names from combined_data
    """
    train_data = combined_data.iloc[:test_start_idx]
    context_data = train_data.tail(CONTEXT_LENGTH)
    target = context_data['target'].values

    # Build covariates based on config
    past_covariates = {}

    # Add covariates if enabled in config
    for cov_name in all_covariate_names:
        key = cov_name.lower().replace(' ', '_')
        if covariate_config.get(key, False):
            past_covariates[key] = context_data[cov_name].values

    # Build input
    if past_covariates:
        input_dict = {
            "target": target,
            "past_covariates": past_covariates
        }
        inputs = [input_dict]
    else:
        # Univariate case (no covariates)
        inputs = torch.tensor(target).unsqueeze(0).unsqueeze(0)  # (1, 1, context_length)

    # Run prediction
    predictions = pipeline.predict(inputs, prediction_length=PREDICTION_LENGTH)
    forecast = predictions[0].squeeze(0).numpy()

    return forecast

def calculate_metrics(forecast, actual):
    """Calculate forecast metrics."""
    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecast = forecast[median_idx, :]

    errors = actual - median_forecast
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(errors / actual)) * 100

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}

def main():
    # Build feature names
    external_cov_names = ', '.join([cov['name'] for cov in COVARIATES])
    cotton_feature_names = ', '.join([f"Cotton {f}" for f in COTTON_FEATURES])

    print("="*80)
    print(f"{TARGET_NAME} - Covariate Importance Analysis (LoRA Fine-Tuned)")
    print(f"External Covariates: {external_cov_names}")
    print(f"Cotton Features: {cotton_feature_names}")
    print("="*80)

    # Load target
    print("\nLoading datasets...")
    target_path = os.path.join(DATA_FOLDER, TARGET_FILE)
    target = load_csv_data(target_path, TARGET_NAME, column='Close')

    # Load external covariates
    covariates_dict = {}
    for cov in COVARIATES:
        cov_path = os.path.join(DATA_FOLDER, cov['file'])
        covariates_dict[cov['name']] = load_csv_data(cov_path, cov['name'])

    # Load Cotton features
    cotton_features = load_cotton_features(target_path, COTTON_FEATURES)
    covariates_dict.update(cotton_features)

    # Align data
    combined_data = align_data(target, covariates_dict)

    # Get all covariate column names
    all_covariate_names = [col for col in combined_data.columns if col != 'target']

    test_start_idx = len(combined_data) - TEST_SET_SIZE
    test_end_idx = test_start_idx + PREDICTION_LENGTH
    actual_prices = combined_data['target'].iloc[test_start_idx:test_end_idx].values

    print(f"\nTest period: {combined_data.index[test_start_idx].strftime('%Y-%m-%d')} to " +
          f"{combined_data.index[test_end_idx-1].strftime('%Y-%m-%d')}")

    # Load fine-tuned model
    print(f"\nLoading fine-tuned model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Build experiments dynamically
    experiments = {}

    # Create sanitized keys for ALL features (external + cotton)
    all_feature_keys = {}
    for cov in COVARIATES:
        all_feature_keys[cov['name']] = cov['name'].lower().replace(' ', '_')
    for cotton_feat in COTTON_FEATURES:
        feat_name = f"Cotton {cotton_feat}"
        all_feature_keys[feat_name] = feat_name.lower().replace(' ', '_')

    # 1. Baseline: no covariates
    experiments['Baseline (No Covariates)'] = {}

    # 2. All covariates
    all_config = {key: True for key in all_feature_keys.values()}
    experiments['All Covariates'] = all_config

    # 3. External covariates only (no Cotton features)
    external_only_config = {all_feature_keys[cov['name']]: True for cov in COVARIATES}
    experiments['External Covariates Only'] = external_only_config

    # 4. Cotton features only (no external)
    cotton_only_config = {all_feature_keys[f"Cotton {feat}"]: True for feat in COTTON_FEATURES}
    experiments['Cotton Features Only'] = cotton_only_config

    # 5. Individual features
    for feat_name, feat_key in all_feature_keys.items():
        experiments[f'{feat_name} Only'] = {feat_key: True}

    # 6. Leave-one-out: All except one feature
    for feat_name, feat_key in all_feature_keys.items():
        loo_config = {key: True for key in all_feature_keys.values() if key != feat_key}
        experiments[f'All Except {feat_name}'] = loo_config

    results = {}

    print("\n" + "="*80)
    print("Running Ablation Experiments...")
    print("="*80)

    for exp_name, config in experiments.items():
        print(f"\n{exp_name}...")
        active_features = [k for k, v in config.items() if v] if config else []
        print(f"  Active features: {len(active_features)}")

        forecast = run_experiment(pipeline, combined_data, test_start_idx, config, all_covariate_names)
        metrics = calculate_metrics(forecast, actual_prices)
        results[exp_name] = metrics

        print(f"  RMSE: ${metrics['rmse']:.4f}, MAE: ${metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%")

    # Print summary table
    print("\n" + "="*80)
    print("COVARIATE IMPORTANCE SUMMARY")
    print("="*80)
    print(f"{'Experiment':<40} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
    print("-"*80)

    # Sort by RMSE (best to worst)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])

    for exp_name, metrics in sorted_results:
        print(f"{exp_name:<40} ${metrics['rmse']:<11.4f} ${metrics['mae']:<11.4f} {metrics['mape']:<11.2f}%")

    print("="*80)

    # Calculate importance scores (based on leave-one-out)
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Leave-One-Out Analysis)")
    print("="*80)

    baseline_rmse = results['Baseline (No Covariates)']['rmse']
    all_rmse = results['All Covariates']['rmse']

    importance_scores = {}

    # For each feature, compare "All" vs "All Except [feature]"
    for feat_name in all_feature_keys.keys():
        loo_exp_name = f'All Except {feat_name}'
        if loo_exp_name in results:
            # Positive value = performance degraded when removed = important
            impact = results[loo_exp_name]['rmse'] - all_rmse
            importance_scores[feat_name] = impact

    print(f"Baseline RMSE (no covariates):     ${baseline_rmse:.4f}")
    print(f"All Covariates RMSE:               ${all_rmse:.4f}")
    print(f"Improvement from covariates:       ${baseline_rmse - all_rmse:.4f} ({(baseline_rmse - all_rmse)/baseline_rmse*100:.1f}%)")
    print()

    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Feature':<25} {'RMSE Increase When Removed':<30} {'Importance':<15}")
    print("-"*80)
    for feature, impact in sorted_importance:
        if impact > 0:
            importance_label = "⚠️ Important!"
        elif impact > -0.01:
            importance_label = "Neutral"
        else:
            importance_label = "Redundant"

        print(f"{feature:<25} ${impact:>8.4f} ({impact/all_rmse*100:>6.2f}%)         {importance_label}")

    print("="*80)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: RMSE comparison for key experiments
    key_experiments = [
        'Baseline (No Covariates)',
        'All Covariates',
        'External Covariates Only',
        'Cotton Features Only'
    ]

    key_exp_names = [name for name in key_experiments if name in results]
    key_rmse_values = [results[name]['rmse'] for name in key_exp_names]

    colors = ['red' if 'Baseline' in name else
              'green' if 'All Covariates' == name else
              'steelblue' for name in key_exp_names]

    ax1.barh(range(len(key_exp_names)), key_rmse_values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(key_exp_names)))
    ax1.set_yticklabels(key_exp_names, fontsize=10)
    ax1.set_xlabel('RMSE (lower is better)', fontsize=11)
    ax1.set_title('Forecast Accuracy: Key Configurations', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Feature importance (leave-one-out)
    features = [f for f, _ in sorted_importance]
    impacts = [imp for _, imp in sorted_importance]

    impact_colors = ['darkred' if imp > 0.01 else 'gray' if imp > -0.01 else 'green' for imp in impacts]

    ax2.barh(features, impacts, color=impact_colors, alpha=0.7)
    ax2.set_xlabel('RMSE Increase When Removed\n(positive = important, negative = redundant)', fontsize=10)
    ax2.set_title('Individual Feature Importance (Leave-One-Out)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    output_path = 'experiments_finetune/lora/feature_importance_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Save detailed results to text file
    output_text = 'experiments_finetune/lora/feature_importance_analysis.txt'
    with open(output_text, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{TARGET_NAME} - Feature Importance Analysis (LoRA Fine-Tuned)\n")
        f.write("="*80 + "\n\n")

        f.write("FULL RESULTS (sorted by RMSE):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Experiment':<40} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}\n")
        f.write("-"*80 + "\n")
        for exp_name, metrics in sorted_results:
            f.write(f"{exp_name:<40} ${metrics['rmse']:<11.4f} ${metrics['mae']:<11.4f} {metrics['mape']:<11.2f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("FEATURE IMPORTANCE RANKING:\n")
        f.write("="*80 + "\n")
        for i, (feature, impact) in enumerate(sorted_importance, 1):
            f.write(f"{i}. {feature:<25} Impact: ${impact:>8.4f} ({impact/all_rmse*100:>6.2f}%)\n")

    print(f"Detailed analysis saved to: {output_text}")
    print("\nDone!")

if __name__ == "__main__":
    main()
