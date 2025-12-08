"""
Covariate importance analysis for Cotton Futures forecasting.
Uses ablation study: removes covariates one-by-one to measure their impact.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

# Configuration
COTTON_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Cotton_Futures.csv"
CRUDE_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Crude_Oil.csv"
COPPER_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Copper_Futures.csv"
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

def align_data(cotton, crude, copper):
    """Align all time series to common dates."""
    combined = pd.DataFrame({'cotton': cotton, 'crude': crude, 'copper': copper})
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

def run_experiment(pipeline, combined_data, test_start_idx, covariate_config):
    """
    Run inference with specific covariate configuration.

    Parameters
    ----------
    covariate_config : dict
        Keys: 'crude_oil', 'copper', 'calendar'
        Values: True (include) or False (exclude)
    """
    train_data = combined_data.iloc[:test_start_idx]
    context_data = train_data.tail(CONTEXT_LENGTH)
    target = context_data['cotton'].values

    past_calendar, future_calendar = create_calendar_features(
        context_data.index,
        combined_data.index[test_start_idx:test_start_idx + PREDICTION_LENGTH]
    )

    # Build covariates based on config
    past_covariates = {}
    future_covariates = {}

    if covariate_config.get('crude_oil', False):
        past_covariates['crude_oil'] = context_data['crude'].values

    if covariate_config.get('copper', False):
        past_covariates['copper'] = context_data['copper'].values

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
    print("="*80)
    print("Cotton Futures - Covariate Importance Analysis (Ablation Study)")
    print("="*80)

    # Load data
    print("\nLoading datasets...")
    cotton = load_csv_data(COTTON_PATH, "Cotton Futures")
    crude = load_csv_data(CRUDE_PATH, "Crude Oil")
    copper = load_csv_data(COPPER_PATH, "Copper Futures")
    combined_data = align_data(cotton, crude, copper)

    test_start_idx = len(combined_data) - PREDICTION_LENGTH
    actual_prices = combined_data['cotton'].iloc[test_start_idx:].values

    print(f"\nTest period: {combined_data.index[test_start_idx].strftime('%Y-%m-%d')} to " +
          f"{combined_data.index[-1].strftime('%Y-%m-%d')}")

    # Load model once
    print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Define experiments
    experiments = {
        'Baseline (No Covariates)': {},
        'All Covariates': {'crude_oil': True, 'copper': True, 'calendar': True},
        'Crude Oil Only': {'crude_oil': True},
        'Copper Only': {'copper': True},
        'Calendar Only': {'calendar': True},
        'Crude + Calendar': {'crude_oil': True, 'calendar': True},
        'Copper + Calendar': {'copper': True, 'calendar': True},
        'Crude + Copper': {'crude_oil': True, 'copper': True},
    }

    results = {}

    print("\n" + "="*80)
    print("Running Ablation Experiments...")
    print("="*80)

    for exp_name, config in experiments.items():
        print(f"\n{exp_name}...")
        print(f"  Config: {config if config else 'None (univariate)'}")

        forecast = run_experiment(pipeline, combined_data, test_start_idx, config)
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

    # Crude Oil importance: compare (All) vs (Copper + Calendar)
    if 'Copper + Calendar' in results:
        crude_impact = results['Copper + Calendar']['rmse'] - all_rmse
        importance_scores['Crude Oil'] = crude_impact

    # Copper importance: compare (All) vs (Crude + Calendar)
    if 'Crude + Calendar' in results:
        copper_impact = results['Crude + Calendar']['rmse'] - all_rmse
        importance_scores['Copper'] = copper_impact

    # Calendar importance: compare (All) vs (Crude + Copper)
    if 'Crude + Copper' in results:
        calendar_impact = results['Crude + Copper']['rmse'] - all_rmse
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
    output_path = 'experiments/multivariate/covariate_importance_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()

    print("\nDone!")

if __name__ == "__main__":
    main()
