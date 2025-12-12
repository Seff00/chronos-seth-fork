"""
Ablation Study: All Covariates Forecasting with Fine-Tuned Chronos-2.
Rolling 1-day ahead forecasting on the test set.
Tests the fine-tuned all covariates model performance.
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuration
DATA_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\commodity-forecasting\data\processed"
MODEL_PATH = r"experiments\finetune_lora\allcovariates\results\checkpoint\finetuned-ckpt"
OUTPUT_DIR = r"experiments\ablation_study\results"

# Target and Covariates (matching the all covariates experiment)
TARGET_COLUMN = "Cotton_Futures_Close"
COVARIATE_COLUMNS = [
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
PREDICTION_DAYS = 30  # Rolling forecast for last 30 days
CONTEXT_LENGTH = 365  # Use last year as context
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_unified_data():
    """Load the unified commodity dataset."""
    print("Loading unified commodity dataset...")

    # Load all splits
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_PATH, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

    # Combine all data
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Convert Date to datetime and set as index
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.set_index('Date')
    combined_df = combined_df.sort_index()

    print(f"Total data: {len(combined_df)} days")
    print(f"Date range: {combined_df.index.min().strftime('%Y-%m-%d')} to {combined_df.index.max().strftime('%Y-%m-%d')}")

    # Select target and covariates
    columns_to_keep = [TARGET_COLUMN] + COVARIATE_COLUMNS
    data = combined_df[columns_to_keep].copy()

    # Rename target column
    data = data.rename(columns={TARGET_COLUMN: 'target'})

    print(f"\nPrepared data with {len(data)} days")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Covariates ({len(COVARIATE_COLUMNS)}): {', '.join(COVARIATE_COLUMNS)}")

    return data

def get_direction(current_price, previous_price):
    """Determine direction of price movement."""
    return 'UP' if current_price > previous_price else 'DOWN'

def calculate_classification_metrics(predicted_directions, actual_directions):
    """Calculate classification metrics for direction prediction."""
    y_pred = np.array([1 if d == 'UP' else 0 for d in predicted_directions])
    y_true = np.array([1 if d == 'UP' else 0 for d in actual_directions])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division='warn')
    recall = recall_score(y_true, y_pred, zero_division='warn')
    f1 = f1_score(y_true, y_pred, zero_division='warn')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def predict_single_day(pipeline, context_data, covariate_columns):
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

def run_rolling_forecast(combined_data, test_start_idx):
    """
    Run rolling 1-day ahead forecasting on test set.

    Parameters
    ----------
    combined_data : pd.DataFrame
        Full dataset with target and covariates
    test_start_idx : int
        Index where test period starts
    """
    print(f"\nLoading fine-tuned Chronos-2 model: {MODEL_PATH}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"\n" + "="*80)
    print("Running Rolling Forecast on Test Set...")
    print("="*80)

    rolling_forecasts = []
    actual_values = []
    previous_actual_values = []
    test_dates = []

    # Get covariate columns (all except 'target')
    covariate_columns = [col for col in combined_data.columns if col != 'target']

    for day_offset in range(PREDICTION_DAYS):
        current_idx = test_start_idx + day_offset
        prediction_date = combined_data.index[current_idx]

        # Use all data up to (but not including) the current day
        context_data = combined_data.iloc[:current_idx]
        actual_value = combined_data['target'].iloc[current_idx]
        previous_actual = combined_data['target'].iloc[current_idx - 1]

        print(f"\nDay {day_offset + 1}/{PREDICTION_DAYS}: Predicting {prediction_date.strftime('%Y-%m-%d')}")
        print(f"  Context size: {len(context_data)} days")

        # Predict single day
        forecast = predict_single_day(pipeline, context_data, covariate_columns)

        median_idx = QUANTILE_LEVELS.index(0.5)
        median_pred = forecast[median_idx]

        predicted_direction = get_direction(median_pred, previous_actual)
        actual_direction = get_direction(actual_value, previous_actual)

        rolling_forecasts.append(forecast)
        actual_values.append(actual_value)
        previous_actual_values.append(previous_actual)
        test_dates.append(prediction_date)

        error = actual_value - median_pred
        correct = "✓" if predicted_direction == actual_direction else "✗"
        print(f"  Previous: ${previous_actual:.2f}")
        print(f"  Actual: ${actual_value:.2f}, Predicted: ${median_pred:.2f}, Error: ${error:.2f}")
        print(f"  Direction - Predicted: {predicted_direction}, Actual: {actual_direction} {correct}")

    rolling_forecasts = np.array(rolling_forecasts)
    actual_values = np.array(actual_values)
    previous_actual_values = np.array(previous_actual_values)

    print("\n" + "="*80)

    return rolling_forecasts, actual_values, previous_actual_values, test_dates

def calculate_metrics(forecasts, actual_values, previous_actual_values):
    """Calculate regression error metrics and classification metrics."""
    median_idx = QUANTILE_LEVELS.index(0.5)
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
        'median_forecasts': median_forecasts,
        'errors': errors,
        'predicted_directions': predicted_directions,
        'actual_directions': actual_directions,
        'classification': classification_metrics
    }

def print_metrics(metrics, actual_values, test_dates):
    """Print detailed metrics and comparison."""
    print("\n" + "="*80)
    print("REGRESSION METRICS")
    print("="*80)
    print(f"Mean Absolute Error (MAE):       ${metrics['mae']:.4f}")
    print(f"Mean Squared Error (MSE):        ${metrics['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE):  ${metrics['rmse']:.4f}")
    print(f"Mean Absolute Percentage Error:  {metrics['mape']:.2f}%")
    print("="*80)

    print("\n" + "="*80)
    print("CLASSIFICATION METRICS (Direction Prediction: UP vs DOWN)")
    print("="*80)
    cls_metrics = metrics['classification']
    print(f"Accuracy:   {cls_metrics['accuracy']:.4f} ({cls_metrics['accuracy']*100:.2f}%)")
    print(f"Precision:  {cls_metrics['precision']:.4f} ({cls_metrics['precision']*100:.2f}%)")
    print(f"Recall:     {cls_metrics['recall']:.4f} ({cls_metrics['recall']*100:.2f}%)")
    print(f"F1-Score:   {cls_metrics['f1_score']:.4f}")
    print("="*80)
    print("Note: Positive class = UP, Negative class = DOWN")

    print("\n" + "="*80)
    print("DAY-BY-DAY COMPARISON")
    print("="*80)
    print(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10} {'Pred Dir':<10} {'Actual Dir':<11} {'Correct':<8}")
    print("-"*80)

    for i, date in enumerate(test_dates):
        actual_val = actual_values[i]
        pred_val = metrics['median_forecasts'][i]
        error = metrics['errors'][i]
        error_pct = (error / actual_val) * 100
        pred_dir = metrics['predicted_directions'][i]
        actual_dir = metrics['actual_directions'][i]
        correct = "✓" if pred_dir == actual_dir else "✗"

        date_str = date.strftime('%Y-%m-%d')
        print(f"{date_str:<12} "
              f"${actual_val:<9.2f} "
              f"${pred_val:<9.2f} "
              f"${error:<9.2f} "
              f"{error_pct:<9.2f}% "
              f"{pred_dir:<10} "
              f"{actual_dir:<11} "
              f"{correct:<8}")

    print("="*80)

def plot_forecast(combined_data, test_start_idx, forecasts, actual_values, test_dates, metrics):
    """Plot time series forecast vs actual with direction-based coloring."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot last 14 days of historical data
    historical_data = combined_data.iloc[test_start_idx - 14:test_start_idx]
    historical_dates = historical_data.index
    historical_prices = historical_data['target'].values

    ax.plot(historical_dates, historical_prices, label='Historical Data',
            color='blue', linewidth=2, marker='o', markersize=4)
    ax.plot(test_dates, actual_values, label='Actual',
            color='darkblue', linewidth=2.5, marker='D', markersize=5)

    # Plot predictions with direction-based colors
    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecasts = forecasts[:, median_idx]
    predicted_dirs = metrics['predicted_directions']

    ax.plot(test_dates, median_forecasts, color='gray', linewidth=1.5,
            linestyle='--', alpha=0.5, zorder=1)

    for i, (date, pred, direction) in enumerate(zip(test_dates, median_forecasts, predicted_dirs)):
        color = 'green' if direction == 'UP' else 'red'
        label = f'Predicted {direction}' if i == 0 or (i > 0 and direction != predicted_dirs[i-1]) else None
        ax.scatter(date, pred, color=color, s=75, marker='s',
                  edgecolor='black', linewidth=1, zorder=3, label=label)

    covariate_str = ", ".join(COVARIATE_COLUMNS)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cotton Futures Price (USD)', fontsize=12)
    ax.set_title(f'Ablation Study: All Covariates Rolling Forecast vs Actual\n' +
                 f'Covariates: {covariate_str}\n' +
                 '(Prediction colors: Green=UP, Red=DOWN)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'forecast_allcovariates_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nForecast plot saved to: {output_path}")
    plt.close()

def plot_metrics_chart(forecasts, actual_values, test_dates, metrics):
    """Plot regression and classification metrics by day."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecasts = forecasts[:, median_idx]

    errors = actual_values - median_forecasts
    error_colors = ['green' if e > 0 else 'red' for e in errors]

    ax1.bar(range(len(test_dates)), errors, color=error_colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Forecast Error (Actual - Predicted)', fontsize=12)
    ax1.set_title('Regression Errors by Day', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(test_dates)))
    ax1.set_xticklabels([d.strftime('%m-%d') for d in test_dates], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    predicted_dirs = metrics['predicted_directions']
    actual_dirs = metrics['actual_directions']
    correct = [1 if p == a else 0 for p, a in zip(predicted_dirs, actual_dirs)]

    colors = ['green' if c == 1 else 'red' for c in correct]
    ax2.bar(range(len(test_dates)), correct, color=colors, alpha=0.7)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Correct (1) / Incorrect (0)', fontsize=12)
    ax2.set_title(f'Direction Prediction Accuracy by Day (Overall: {metrics["classification"]["accuracy"]*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(test_dates)))
    ax2.set_xticklabels([d.strftime('%m-%d') for d in test_dates], rotation=45)
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Incorrect', 'Correct'])
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (pred, actual, corr) in enumerate(zip(predicted_dirs, actual_dirs, correct)):
        label = f"{pred}\n({actual})"
        y_pos = 0.5
        ax2.text(i, y_pos, label, ha='center', va='center', fontsize=8,
                color='white' if corr else 'black', fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'forecast_allcovariates_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {output_path}")
    plt.close()

def save_output(metrics, actual_values, test_dates, combined_data, test_start_idx):
    """Save evaluation output to text file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'forecast_allcovariates_output.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Cotton Futures: All Covariates Ablation Study\n")
        f.write("Fine-Tuned LoRA Model - Rolling 1-Day Ahead Forecast\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Prediction Method: Rolling 1-day ahead\n")
        f.write(f"Context Length: {CONTEXT_LENGTH} days\n\n")

        f.write(f"Covariates ({len(COVARIATE_COLUMNS)}):\n")
        for cov in COVARIATE_COLUMNS:
            f.write(f"  - {cov}\n")
        f.write("\n")

        f.write(f"Train period: {combined_data.index[0].strftime('%Y-%m-%d')} to " +
                f"{combined_data.index[test_start_idx - 1].strftime('%Y-%m-%d')}\n")
        f.write(f"Test period:  {test_dates[0].strftime('%Y-%m-%d')} to " +
                f"{test_dates[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"Train size: {test_start_idx} days\n")
        f.write(f"Test size:  {PREDICTION_DAYS} days\n\n")

        f.write("="*80 + "\n")
        f.write("REGRESSION METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Mean Absolute Error (MAE):       ${metrics['mae']:.4f}\n")
        f.write(f"Mean Squared Error (MSE):        ${metrics['mse']:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE):  ${metrics['rmse']:.4f}\n")
        f.write(f"Mean Absolute Percentage Error:  {metrics['mape']:.2f}%\n")
        f.write("="*80 + "\n\n")

        f.write("="*80 + "\n")
        f.write("CLASSIFICATION METRICS (Direction Prediction: UP vs DOWN)\n")
        f.write("="*80 + "\n")
        cls_metrics = metrics['classification']
        f.write(f"Accuracy:   {cls_metrics['accuracy']:.4f} ({cls_metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:  {cls_metrics['precision']:.4f} ({cls_metrics['precision']*100:.2f}%)\n")
        f.write(f"Recall:     {cls_metrics['recall']:.4f} ({cls_metrics['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:   {cls_metrics['f1_score']:.4f}\n")
        f.write("="*80 + "\n")
        f.write("Note: Positive class = UP, Negative class = DOWN\n\n")

        f.write("="*80 + "\n")
        f.write("DAY-BY-DAY COMPARISON\n")
        f.write("="*80 + "\n")
        f.write(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10} {'Pred Dir':<10} {'Actual Dir':<11} {'Correct':<8}\n")
        f.write("-"*80 + "\n")

        for i, date in enumerate(test_dates):
            actual_val = actual_values[i]
            pred_val = metrics['median_forecasts'][i]
            error = metrics['errors'][i]
            error_pct = (error / actual_val) * 100
            pred_dir = metrics['predicted_directions'][i]
            actual_dir = metrics['actual_directions'][i]
            correct = "✓" if pred_dir == actual_dir else "✗"

            date_str = date.strftime('%Y-%m-%d')
            f.write(f"{date_str:<12} "
                   f"${actual_val:<9.2f} "
                   f"${pred_val:<9.2f} "
                   f"${error:<9.2f} "
                   f"{error_pct:<9.2f}% "
                   f"{pred_dir:<10} "
                   f"{actual_dir:<11} "
                   f"{correct:<8}\n")

        f.write("="*80 + "\n")

    print(f"Output saved to: {output_file}")

def main():
    print("="*80)
    print("Cotton Futures: All Covariates Ablation Study")
    print("Fine-Tuned LoRA Model - Rolling 1-Day Ahead Forecast")
    print(f"Evaluation Mode: Rolling Forecast on Last {PREDICTION_DAYS} Days")
    print("="*80)

    # Load data
    combined_data = load_unified_data()

    # Split: use all except last N days for testing
    test_start_idx = len(combined_data) - PREDICTION_DAYS

    print(f"\nTest period: {combined_data.index[test_start_idx].strftime('%Y-%m-%d')} to " +
          f"{combined_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Test size: {PREDICTION_DAYS} days")

    # Run rolling forecast
    rolling_forecasts, actual_values, previous_actual_values, test_dates = run_rolling_forecast(
        combined_data, test_start_idx
    )

    # Calculate metrics
    metrics = calculate_metrics(rolling_forecasts, actual_values, previous_actual_values)

    # Print metrics
    print_metrics(metrics, actual_values, test_dates)

    # Plot results
    plot_forecast(combined_data, test_start_idx, rolling_forecasts, actual_values, test_dates, metrics)
    plot_metrics_chart(rolling_forecasts, actual_values, test_dates, metrics)

    # Save output
    save_output(metrics, actual_values, test_dates, combined_data, test_start_idx)

    print("\nDone!")
    print(f"\nNOTE: This rolling forecast uses all {len(COVARIATE_COLUMNS)} covariates")
    print("      for 1-day ahead predictions over {PREDICTION_DAYS} days.")

if __name__ == "__main__":
    main()
