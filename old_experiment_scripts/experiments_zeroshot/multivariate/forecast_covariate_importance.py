"""
Covariate importance analysis using ablation study.
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
    # Add more covariates here:
    # {"file": "Gold_Futures.csv", "name": "Gold"},
    # {"file": "Natural_Gas.csv", "name": "Natural Gas"},
]

MODEL_NAME = "amazon/chronos-2"
PREDICTION_LENGTH = 7
CONTEXT_LENGTH = 1024
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_csv_data(filepath, asset_name):
    """Load and preprocess commodity CSV data."""
    df = pd.read_csv(filepath, header=0, skiprows=[1, 2], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df = df.sort_index()
    prices = df['Close'].rename(asset_name)
    return prices

def align_data(target, covariates_dict):
    """Align target and all covariate time series to common dates."""
    data_dict = {'target': target}
    data_dict.update(covariates_dict)
    combined = pd.DataFrame(data_dict)
    combined = combined.fillna(method='ffill').dropna()
    return combined

def create_calendar_features(dates, prediction_dates):
    """Create calendar features."""
    past_features = pd.DataFrame({
        'month': dates.month,
        'day_of_week': dates.dayofweek,
        'quarter': dates.quarter,
    }, index=dates)

    future_features = pd.DataFrame({
        'month': prediction_dates.month,
        'day_of_week': prediction_dates.dayofweek,
        'quarter': prediction_dates.quarter,
    }, index=prediction_dates)

    return past_features, future_features

def run_experiment(pipeline, combined_data, test_start_idx, covariate_config, all_covariate_names):
    """
    Run inference with specific covariate configuration.

    Parameters
    ----------
    covariate_config : dict
        Keys: covariate names (sanitized) or 'calendar'
        Values: True (include) or False (exclude)
    all_covariate_names : list
        List of all available covariate names from combined_data
    """
    train_data = combined_data.iloc[:test_start_idx]
    context_data = train_data.tail(CONTEXT_LENGTH)
    target = context_data['target'].values

    past_calendar, future_calendar = create_calendar_features(
        context_data.index,
        combined_data.index[test_start_idx:test_start_idx + PREDICTION_LENGTH]
    )

    # Build covariates based on config
    past_covariates = {}
    future_covariates = {}

    # Add price covariates if enabled in config
    for cov_name in all_covariate_names:
        key = cov_name.lower().replace(' ', '_')
        if covariate_config.get(key, False):
            past_covariates[key] = context_data[cov_name].values

    # Add calendar features if enabled
    if covariate_config.get('calendar', False):
        past_covariates['month'] = past_calendar['month'].values
        past_covariates['day_of_week'] = past_calendar['day_of_week'].values
        past_covariates['quarter'] = past_calendar['quarter'].values
        future_covariates['month'] = future_calendar['month'].values
        future_covariates['day_of_week'] = future_calendar['day_of_week'].values
        future_covariates['quarter'] = future_calendar['quarter'].values

    # Build input
    if past_covariates or future_covariates:
        input_dict = {"target": target}
        if past_covariates:
            input_dict["past_covariates"] = past_covariates
        if future_covariates:
            input_dict["future_covariates"] = future_covariates
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
    covariate_names_str = ', '.join([cov['name'] for cov in COVARIATES])

    print("="*80)
    print(f"{TARGET_NAME} - Covariate Importance Analysis (Ablation Study)")
    print(f"Available Covariates: {covariate_names_str}")
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

    # Get covariate column names
    all_covariate_names = [col for col in combined_data.columns if col != 'target']

    test_start_idx = len(combined_data) - PREDICTION_LENGTH
    actual_prices = combined_data['target'].iloc[test_start_idx:].values

    print(f"\nTest period: {combined_data.index[test_start_idx].strftime('%Y-%m-%d')} to " +
          f"{combined_data.index[-1].strftime('%Y-%m-%d')}")

    # Load model once
    print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Build experiments dynamically based on available covariates
    experiments = {}

    # Create sanitized keys for each covariate
    cov_keys = {cov['name']: cov['name'].lower().replace(' ', '_') for cov in COVARIATES}

    # Baseline: no covariates
    experiments['Baseline (No Covariates)'] = {}

    # All covariates (including calendar)
    all_config = {key: True for key in cov_keys.values()}
    all_config['calendar'] = True
    experiments['All Covariates'] = all_config

    # Individual covariates only
    for cov_name, cov_key in cov_keys.items():
        experiments[f'{cov_name} Only'] = {cov_key: True}

    # Calendar only
    experiments['Calendar Only'] = {'calendar': True}

    # Each covariate + calendar
    for cov_name, cov_key in cov_keys.items():
        experiments[f'{cov_name} + Calendar'] = {cov_key: True, 'calendar': True}

    # All combinations of price covariates (without calendar)
    if len(COVARIATES) == 2:
        # For 2 covariates, just add the combination
        all_price_config = {key: True for key in cov_keys.values()}
        cov_combo_name = ' + '.join([cov['name'] for cov in COVARIATES])
        experiments[cov_combo_name] = all_price_config

    results = {}

    print("\n" + "="*80)
    print("Running Ablation Experiments...")
    print("="*80)

    for exp_name, config in experiments.items():
        print(f"\n{exp_name}...")
        print(f"  Config: {config if config else 'None (univariate)'}")

        forecast = run_experiment(pipeline, combined_data, test_start_idx, config, all_covariate_names)
        metrics = calculate_metrics(forecast, actual_prices)
        results[exp_name] = metrics

        print(f"  RMSE: ${metrics['rmse']:.4f}, MAE: ${metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%")

    # Print summary table
    print("\n" + "="*80)
    print("COVARIATE IMPORTANCE SUMMARY")
    print("="*80)
    print(f"{'Experiment':<30} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
    print("-"*80)

    # Sort by RMSE (best to worst)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])

    for exp_name, metrics in sorted_results:
        print(f"{exp_name:<30} ${metrics['rmse']:<11.4f} ${metrics['mae']:<11.4f} {metrics['mape']:<11.2f}%")

    print("="*80)

    # Calculate importance scores
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (based on RMSE degradation)")
    print("="*80)

    baseline_rmse = results['Baseline (No Covariates)']['rmse']
    all_rmse = results['All Covariates']['rmse']

    # Individual importance: degradation when removing from "All Covariates"
    # Compare "All Covariates" vs combinations without each feature
    importance_scores = {}

    # For each covariate, compare "All" vs configuration without that covariate
    for cov in COVARIATES:
        cov_name = cov['name']
        # Find the experiment that has all OTHER covariates + calendar (excluding this one)
        other_covs = [c for c in COVARIATES if c['name'] != cov_name]

        if len(COVARIATES) == 2:
            # For 2 covariates, the complement is just the other one + calendar
            other_name = other_covs[0]['name']
            complement_exp_name = f'{other_name} + Calendar'
            if complement_exp_name in results:
                impact = results[complement_exp_name]['rmse'] - all_rmse
                importance_scores[cov_name] = impact

    # Calendar importance: compare "All" vs price covariates only (no calendar)
    if len(COVARIATES) == 2:
        price_only_name = ' + '.join([cov['name'] for cov in COVARIATES])
        if price_only_name in results:
            calendar_impact = results[price_only_name]['rmse'] - all_rmse
            importance_scores['Calendar'] = calendar_impact

    print(f"Baseline RMSE (no covariates):     ${baseline_rmse:.4f}")
    print(f"All Covariates RMSE:               ${all_rmse:.4f}")
    print(f"Improvement from covariates:       ${baseline_rmse - all_rmse:.4f} ({(baseline_rmse - all_rmse)/baseline_rmse*100:.1f}%)")
    print()

    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    for feature, impact in sorted_importance:
        if impact > 0:
            print(f"{feature:15} - RMSE increase when removed: ${impact:.4f} ⚠️ (Important!)")
        else:
            print(f"{feature:15} - RMSE increase when removed: ${impact:.4f} (Negligible)")

    print("="*80)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: RMSE comparison
    exp_names = [name for name, _ in sorted_results]
    rmse_values = [metrics['rmse'] for _, metrics in sorted_results]

    colors = ['green' if name == 'All Covariates' else 'red' if name == 'Baseline (No Covariates)' else 'steelblue'
              for name in exp_names]

    ax1.barh(range(len(exp_names)), rmse_values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(exp_names)))
    ax1.set_yticklabels(exp_names, fontsize=9)
    ax1.set_xlabel('RMSE (lower is better)', fontsize=11)
    ax1.set_title('Forecast Accuracy by Covariate Configuration', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Feature importance
    features = list(importance_scores.keys())
    impacts = list(importance_scores.values())

    impact_colors = ['darkred' if imp > 0 else 'gray' for imp in impacts]

    ax2.barh(features, impacts, color=impact_colors, alpha=0.7)
    ax2.set_xlabel('RMSE Increase When Removed (higher = more important)', fontsize=11)
    ax2.set_title('Individual Feature Importance', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = 'experiments_zeroshot/multivariate/forecast_covariate_importance_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()

    print("\nDone!")

if __name__ == "__main__":
    main()
