"""
Ablation Study Loop: Remove one covariate at a time.
Tests model performance when each covariate is individually excluded.
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# Configuration
DATA_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\commodity-forecasting\data\processed"
MODEL_PATH = r"experiments\finetune_lora\allcovariates\results\checkpoint\finetuned-ckpt"
OUTPUT_DIR = r"experiments\ablation_study\results"

# Target and Covariates
TARGET_COLUMN = "Cotton_Futures_Close"
ALL_COVARIATES = [
    "Crude_Oil_Close",
    "Copper_Futures_Close",
    "SP500_Close",
    "Dollar_Index_Close",
    "Cotton_Futures_High",
    "Cotton_Futures_Low",
    "Cotton_Futures_Open",
    "Cotton_Futures_Volume"
]

# Forecasting Configuration
PREDICTION_DAYS = 30
CONTEXT_LENGTH = 365
# Note: Quantiles are determined by the model, not hardcoded here

def load_unified_data():
    """Load the unified commodity dataset."""
    print("Loading unified commodity dataset...")

    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_PATH, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.set_index('Date')
    combined_df = combined_df.sort_index()

    print(f"Total data: {len(combined_df)} days")
    print(f"Date range: {combined_df.index.min().strftime('%Y-%m-%d')} to {combined_df.index.max().strftime('%Y-%m-%d')}")

    return combined_df

def prepare_data_with_covariates(combined_df, covariate_list):
    """Prepare data with specific covariates."""
    columns_to_keep = [TARGET_COLUMN] + covariate_list
    data = combined_df[columns_to_keep].copy()
    data = data.rename(columns={TARGET_COLUMN: 'target'})
    return data

def get_direction(current_price, previous_price):
    """Determine direction of price movement."""
    return 'UP' if current_price > previous_price else 'DOWN'

def calculate_classification_metrics(predicted_directions, actual_directions):
    """Calculate classification metrics for direction prediction."""
    y_pred = np.array([1 if d == 'UP' else 0 for d in predicted_directions])
    y_true = np.array([1 if d == 'UP' else 0 for d in actual_directions])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def predict_single_day(pipeline, context_data, covariate_columns, median_idx):
    """Predict a single day using context data."""
    context = context_data.tail(CONTEXT_LENGTH)
    target = context['target'].values

    # Build past covariates dictionary
    past_covariates = {}
    for col in covariate_columns:
        key = col.lower().replace('_close', '').replace('_', '_')
        past_covariates[key] = context[col].values

    # Build input dictionary
    input_dict = {
        "target": target,
        "past_covariates": past_covariates
    }

    # Run prediction (1-day ahead)
    predictions = pipeline.predict([input_dict], prediction_length=1)
    forecast = predictions[0].squeeze().numpy()

    return forecast

def run_rolling_forecast(pipeline, combined_data, test_start_idx, covariate_columns):
    """Run rolling 1-day ahead forecasting on test set."""
    # Get model's actual quantiles
    model_quantiles = pipeline.quantiles
    try:
        median_idx = list(model_quantiles).index(0.5)
    except ValueError:
        # If exact 0.5 not found, find closest
        median_idx = min(range(len(model_quantiles)), key=lambda i: abs(model_quantiles[i] - 0.5))
        print(f"Warning: Using quantile {model_quantiles[median_idx]:.3f} as median (0.5 not found)")

    print(f"Model quantiles: {model_quantiles}")
    print(f"Using median index: {median_idx} (quantile: {model_quantiles[median_idx]})")

    rolling_forecasts = []
    actual_values = []
    previous_actual_values = []
    test_dates = []

    for day_offset in range(PREDICTION_DAYS):
        current_idx = test_start_idx + day_offset
        prediction_date = combined_data.index[current_idx]

        # Use all data up to (but not including) the current day
        context_data = combined_data.iloc[:current_idx]
        actual_value = combined_data['target'].iloc[current_idx]
        previous_actual = combined_data['target'].iloc[current_idx - 1]

        # Predict single day
        forecast = predict_single_day(pipeline, context_data, covariate_columns, median_idx)

        median_pred = forecast[median_idx]

        rolling_forecasts.append(forecast)
        actual_values.append(actual_value)
        previous_actual_values.append(previous_actual)
        test_dates.append(prediction_date)

    rolling_forecasts = np.array(rolling_forecasts)
    actual_values = np.array(actual_values)
    previous_actual_values = np.array(previous_actual_values)

    return rolling_forecasts, actual_values, previous_actual_values, test_dates, median_idx

def calculate_metrics(forecasts, actual_values, previous_actual_values, median_idx):
    """Calculate regression error metrics and classification metrics."""
    median_forecasts = forecasts[:, median_idx]

    # Regression metrics
    errors = actual_values - median_forecasts
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(errors / actual_values)) * 100

    # Classification metrics
    predicted_directions = [get_direction(pred, prev)
                           for pred, prev in zip(median_forecasts, previous_actual_values)]
    actual_directions = [get_direction(actual, prev)
                        for actual, prev in zip(actual_values, previous_actual_values)]

    classification_metrics = calculate_classification_metrics(predicted_directions, actual_directions)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'accuracy': classification_metrics['accuracy'],
        'precision': classification_metrics['precision'],
        'recall': classification_metrics['recall'],
        'f1_score': classification_metrics['f1_score']
    }

def run_ablation_experiment(excluded_covariate=None):
    """
    Run ablation experiment with one covariate excluded.

    Parameters
    ----------
    excluded_covariate : str or None
        Name of covariate to exclude. If None, run with all covariates (baseline).
    """
    if excluded_covariate is None:
        experiment_name = "baseline_all_covariates"
        covariates_to_use = ALL_COVARIATES.copy()
        print("\n" + "="*80)
        print(f"BASELINE: Using ALL {len(covariates_to_use)} covariates")
        print("="*80)
    else:
        experiment_name = f"exclude_{excluded_covariate.replace('_', '')}"
        covariates_to_use = [c for c in ALL_COVARIATES if c != excluded_covariate]
        print("\n" + "="*80)
        print(f"ABLATION: Excluding '{excluded_covariate}'")
        print(f"Using {len(covariates_to_use)} covariates: {', '.join(covariates_to_use)}")
        print("="*80)

    # Load data
    combined_df = load_unified_data()

    # Prepare data with selected covariates
    combined_data = prepare_data_with_covariates(combined_df, covariates_to_use)

    # Split
    test_start_idx = len(combined_data) - PREDICTION_DAYS

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Get covariate columns (all except 'target')
    covariate_columns = [col for col in combined_data.columns if col != 'target']

    # Run rolling forecast
    print(f"Running rolling forecast on {PREDICTION_DAYS} days...")
    rolling_forecasts, actual_values, previous_actual_values, test_dates, median_idx = run_rolling_forecast(
        pipeline, combined_data, test_start_idx, covariate_columns
    )

    # Calculate metrics
    metrics = calculate_metrics(rolling_forecasts, actual_values, previous_actual_values, median_idx)

    print(f"\nResults for {experiment_name}:")
    print(f"  RMSE: ${metrics['rmse']:.4f}")
    print(f"  MAE:  ${metrics['mae']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")

    return {
        'experiment_name': experiment_name,
        'excluded_covariate': excluded_covariate,
        'num_covariates': len(covariates_to_use),
        'covariates_used': covariates_to_use,
        'metrics': metrics
    }

def save_results(all_results):
    """Save all ablation results to files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save JSON
    results_for_json = []
    for result in all_results:
        results_for_json.append({
            'experiment_name': result['experiment_name'],
            'excluded_covariate': result['excluded_covariate'],
            'num_covariates': result['num_covariates'],
            'covariates_used': result['covariates_used'],
            'mae': result['metrics']['mae'],
            'mse': result['metrics']['mse'],
            'rmse': result['metrics']['rmse'],
            'mape': result['metrics']['mape'],
            'accuracy': result['metrics']['accuracy'],
            'precision': result['metrics']['precision'],
            'recall': result['metrics']['recall'],
            'f1_score': result['metrics']['f1_score']
        })

    json_path = os.path.join(OUTPUT_DIR, 'ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Save text report
    text_path = os.path.join(OUTPUT_DIR, 'ablation_report.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION STUDY REPORT\n")
        f.write("Systematically removing one covariate at a time\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test Period: {PREDICTION_DAYS} days (rolling 1-day ahead)\n")
        f.write(f"Total Covariates: {len(ALL_COVARIATES)}\n\n")

        # Find baseline
        baseline = next(r for r in all_results if r['excluded_covariate'] is None)

        f.write("="*80 + "\n")
        f.write("BASELINE (All Covariates)\n")
        f.write("="*80 + "\n")
        f.write(f"RMSE:      ${baseline['metrics']['rmse']:.4f}\n")
        f.write(f"MAE:       ${baseline['metrics']['mae']:.4f}\n")
        f.write(f"MAPE:      {baseline['metrics']['mape']:.2f}%\n")
        f.write(f"Accuracy:  {baseline['metrics']['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {baseline['metrics']['precision']*100:.2f}%\n")
        f.write(f"Recall:    {baseline['metrics']['recall']*100:.2f}%\n")
        f.write(f"F1-Score:  {baseline['metrics']['f1_score']:.4f}\n\n")

        f.write("="*80 + "\n")
        f.write("ABLATION RESULTS (Excluding One Covariate at a Time)\n")
        f.write("="*80 + "\n\n")

        # Sort by RMSE change (most important = biggest degradation when removed)
        ablation_results = [r for r in all_results if r['excluded_covariate'] is not None]
        ablation_results.sort(key=lambda x: x['metrics']['rmse'] - baseline['metrics']['rmse'], reverse=True)

        f.write(f"{'Excluded Covariate':<30} {'RMSE':<12} {'Δ RMSE':<12} {'Accuracy':<12} {'Δ Acc':<12} {'Impact':<10}\n")
        f.write("-"*80 + "\n")

        for result in ablation_results:
            excluded = result['excluded_covariate']
            rmse = result['metrics']['rmse']
            acc = result['metrics']['accuracy']

            delta_rmse = rmse - baseline['metrics']['rmse']
            delta_acc = acc - baseline['metrics']['accuracy']

            # Determine impact
            if delta_rmse > 0.05:
                impact = "HIGH"
            elif delta_rmse > 0.02:
                impact = "MEDIUM"
            else:
                impact = "LOW"

            f.write(f"{excluded:<30} "
                   f"${rmse:<11.4f} "
                   f"${delta_rmse:+.4f}     "
                   f"{acc*100:<11.2f}% "
                   f"{delta_acc*100:+.2f}%      "
                   f"{impact:<10}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n")
        f.write("Δ RMSE > 0: Removing covariate WORSENS performance (important feature)\n")
        f.write("Δ RMSE < 0: Removing covariate IMPROVES performance (noisy/redundant feature)\n")
        f.write("Δ RMSE ≈ 0: Removing covariate has minimal impact\n\n")

        f.write("Impact Levels:\n")
        f.write("  HIGH:   Δ RMSE > $0.05 (critical feature)\n")
        f.write("  MEDIUM: Δ RMSE > $0.02 (useful feature)\n")
        f.write("  LOW:    Δ RMSE ≤ $0.02 (minimal contribution)\n")

    print(f"Report saved to: {text_path}")

def plot_ablation_results(all_results):
    """Create visualization of ablation study results."""
    # Find baseline
    baseline = next(r for r in all_results if r['excluded_covariate'] is None)
    baseline_rmse = baseline['metrics']['rmse']
    baseline_acc = baseline['metrics']['accuracy']

    # Get ablation results
    ablation_results = [r for r in all_results if r['excluded_covariate'] is not None]

    # Sort by RMSE impact
    ablation_results.sort(key=lambda x: x['metrics']['rmse'] - baseline_rmse, reverse=True)

    excluded_names = [r['excluded_covariate'].replace('_Close', '').replace('Cotton_Futures_', '')
                     for r in ablation_results]
    rmse_values = [r['metrics']['rmse'] for r in ablation_results]
    acc_values = [r['metrics']['accuracy'] * 100 for r in ablation_results]

    # Calculate deltas
    rmse_deltas = [rmse - baseline_rmse for rmse in rmse_values]
    acc_deltas = [(acc/100 - baseline_acc) * 100 for acc in acc_values]

    # Create figure with constrained layout instead of tight_layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)

    # Plot 1: RMSE Change
    colors1 = ['red' if d > 0 else 'green' for d in rmse_deltas]
    bars1 = ax1.barh(excluded_names, rmse_deltas, color=colors1, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax1.set_xlabel('Δ RMSE (Excluded - Baseline)', fontsize=11)
    ax1.set_title('Impact of Removing Each Covariate on RMSE\n(Positive = Performance Degradation = Important Feature)',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, delta) in enumerate(zip(bars1, rmse_deltas)):
        x_pos = delta + (0.005 if delta > 0 else -0.005)
        ha = 'left' if delta > 0 else 'right'
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2, f'${delta:+.4f}',
                ha=ha, va='center', fontweight='bold', fontsize=8)

    # Plot 2: Accuracy Change
    colors2 = ['green' if d > 0 else 'red' for d in acc_deltas]
    bars2 = ax2.barh(excluded_names, acc_deltas, color=colors2, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax2.set_xlabel('Δ Accuracy % (Excluded - Baseline)', fontsize=11)
    ax2.set_title('Impact of Removing Each Covariate on Direction Accuracy\n(Positive = Performance Improvement)',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, delta) in enumerate(zip(bars2, acc_deltas)):
        x_pos = delta + (0.3 if delta > 0 else -0.3)
        ha = 'left' if delta > 0 else 'right'
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, f'{delta:+.1f}%',
                ha=ha, va='center', fontweight='bold', fontsize=8)

    plot_path = os.path.join(OUTPUT_DIR, 'ablation_results.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()

def main():
    print("="*80)
    print("ABLATION STUDY: Systematic Covariate Removal")
    print("="*80)
    print(f"Testing {len(ALL_COVARIATES)} covariates")
    print(f"Total experiments: {len(ALL_COVARIATES) + 1} (baseline + {len(ALL_COVARIATES)} ablations)")
    print("="*80)

    all_results = []

    # Run baseline (all covariates)
    baseline_result = run_ablation_experiment(excluded_covariate=None)
    all_results.append(baseline_result)

    # Run ablation for each covariate
    for i, covariate in enumerate(ALL_COVARIATES, 1):
        print(f"\nExperiment {i+1}/{len(ALL_COVARIATES)+1}")
        result = run_ablation_experiment(excluded_covariate=covariate)
        all_results.append(result)

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)

    # Save results
    save_results(all_results)
    plot_ablation_results(all_results)

    print("\nAll results saved!")
    print(f"  - JSON: {os.path.join(OUTPUT_DIR, 'ablation_results.json')}")
    print(f"  - Report: {os.path.join(OUTPUT_DIR, 'ablation_report.txt')}")
    print(f"  - Plot: {os.path.join(OUTPUT_DIR, 'ablation_results.png')}")

if __name__ == "__main__":
    main()
