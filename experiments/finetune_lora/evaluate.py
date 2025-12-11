"""
Evaluate fine-tuned Chronos-2 model on test set.
Loads checkpoint and runs rolling 1-day ahead forecasting.
Configuration loaded from config.yaml passed as argument.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    return train_data, test_data

# Evaluation Functions

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

def predict_single_day(pipeline, context_data, covariate_columns, context_length):
    """Predict a single day using context data."""
    context = context_data.tail(context_length)
    target = context['target'].values

    past_covariates = {}
    for col in covariate_columns:
        key = col.lower().replace('_close', '').replace('_', '_')
        past_covariates[key] = context[col].values

    input_dict = {
        "target": target,
        "past_covariates": past_covariates
    }

    predictions = pipeline.predict([input_dict], prediction_length=1)
    forecast = predictions[0].squeeze().numpy()

    return forecast

def run_rolling_forecast(pipeline, combined_data, test_start_idx, config):
    """Run rolling forecast for the test set."""
    prediction_days = config['forecast']['prediction_days']
    context_length = config['training']['context_length']
    quantile_levels = config['forecast']['quantile_levels']

    print(f"\n" + "="*80)
    print("Running Rolling Forecast on Test Set...")
    print("="*80)

    rolling_forecasts = []
    actual_values = []
    previous_actual_values = []
    test_dates = []

    covariate_columns = [col for col in combined_data.columns if col != 'target']

    for day_offset in range(prediction_days):
        current_idx = test_start_idx + day_offset
        prediction_date = combined_data.index[current_idx]

        context_data = combined_data.iloc[:current_idx]
        actual_value = combined_data['target'].iloc[current_idx]
        previous_actual = combined_data['target'].iloc[current_idx - 1]

        print(f"\nDay {day_offset + 1}/{prediction_days}: Predicting {prediction_date.strftime('%Y-%m-%d')}")
        print(f"  Context size: {len(context_data)} days")

        forecast = predict_single_day(pipeline, context_data, covariate_columns, context_length)

        median_idx = quantile_levels.index(0.5)
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

def calculate_metrics(forecasts, actual_values, previous_actual_values, quantile_levels):
    """Calculate regression error metrics and classification metrics."""
    median_idx = quantile_levels.index(0.5)
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

def plot_forecast(combined_data, test_start_idx, forecasts, actual_values, test_dates, metrics, config):
    """Plot time series forecast vs actual with direction-based coloring."""
    fig, ax = plt.subplots(figsize=(14, 6))

    quantile_levels = config['forecast']['quantile_levels']

    historical_data = combined_data.iloc[test_start_idx - 14:test_start_idx]
    historical_dates = historical_data.index
    historical_prices = historical_data['target'].values

    median_idx = quantile_levels.index(0.5)
    median_forecasts = forecasts[:, median_idx]

    ax.plot(historical_dates, historical_prices, label='Historical Data',
            color='blue', linewidth=2, marker='o', markersize=4)
    ax.plot(test_dates, actual_values, label='Actual',
            color='darkblue', linewidth=2.5, marker='D', markersize=5)

    predicted_dirs = metrics['predicted_directions']
    ax.plot(test_dates, median_forecasts, color='gray', linewidth=1.5,
            linestyle='--', alpha=0.5, zorder=1)

    for i, (date, pred, direction) in enumerate(zip(test_dates, median_forecasts, predicted_dirs)):
        color = 'green' if direction == 'UP' else 'red'
        label = f'Predicted {direction}' if i == 0 or (i > 0 and direction != predicted_dirs[i-1]) else None
        ax.scatter(date, pred, color=color, s=75, marker='s',
                  edgecolor='black', linewidth=1, zorder=3, label=label)

    covariate_str = ", ".join(config['data']['covariate_columns'])
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cotton Futures Price (USD)', fontsize=12)
    ax.set_title(f'{config["experiment"]["name"]}: Rolling Forecast vs Actual\n' +
                 f'Covariates: {covariate_str}\n' +
                 '(Prediction colors: Green=UP, Red=DOWN)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'forecast_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nForecast plot saved to: {output_path}")
    plt.close()

def plot_metrics_chart(forecasts, actual_values, test_dates, metrics, config):
    """Plot regression and classification metrics by day."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    quantile_levels = config['forecast']['quantile_levels']
    median_idx = quantile_levels.index(0.5)
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

    output_dir = config['output']['output_dir']
    output_path = os.path.join(output_dir, 'metrics_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {output_path}")
    plt.close()

def main():
    # Check for config file argument
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <config_path>")
        print("Example: python evaluate.py 2covariates/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    # Load configuration
    config = load_config(config_path)

    # Get checkpoint directory
    checkpoint_dir = config['output']['checkpoint_dir']
    checkpoint_path = os.path.join(checkpoint_dir, "finetuned-ckpt")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Please run training first: python train.py {config_path}")
        sys.exit(1)

    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Redirect output to file
    output_file = os.path.join(output_dir, 'evaluation_output.txt')
    original_stdout = sys.stdout

    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f

        print("="*80)
        print(f"{config['experiment']['name']} - Evaluation")
        print(f"{config['experiment']['description']}")
        print("="*80)

        # Load and prepare data
        combined_df = load_unified_data(config)
        data = prepare_data_for_training(combined_df, config)

        # Split into train and test
        prediction_days = config['forecast']['prediction_days']
        train_data, test_data = split_train_test(data, test_size=prediction_days)

        # Prepare full data for rolling evaluation
        full_data = pd.concat([train_data, test_data])
        test_start_idx = len(full_data) - prediction_days

        print(f"\nTest period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Test size: {prediction_days} days")

        # Load fine-tuned model
        print(f"\nLoading fine-tuned model from: {checkpoint_path}")
        torch_dtype = getattr(torch, config['model']['torch_dtype'])
        finetuned_pipeline = Chronos2Pipeline.from_pretrained(
            checkpoint_path,
            device_map=config['model']['device_map'],
            torch_dtype=torch_dtype,
        )

        # Run rolling forecast on test set
        rolling_forecasts, actual_values, previous_actual_values, test_dates = run_rolling_forecast(
            finetuned_pipeline, full_data, test_start_idx, config
        )

        # Calculate metrics
        metrics = calculate_metrics(rolling_forecasts, actual_values, previous_actual_values,
                                    config['forecast']['quantile_levels'])

        # Print metrics
        print_metrics(metrics, actual_values, test_dates)

        print("\nDone!")

    # Restore stdout
    sys.stdout = original_stdout
    print(f"Evaluation output saved to: {output_file}")

    # Generate plots
    if config['output']['save_plots']:
        print("Generating plots...")

        # Reload data for plotting
        combined_df = load_unified_data(config)
        data = prepare_data_for_training(combined_df, config)
        train_data, test_data = split_train_test(data, test_size=prediction_days)
        full_data = pd.concat([train_data, test_data])
        test_start_idx = len(full_data) - prediction_days

        # Load model
        torch_dtype = getattr(torch, config['model']['torch_dtype'])
        finetuned_pipeline = Chronos2Pipeline.from_pretrained(
            checkpoint_path,
            device_map=config['model']['device_map'],
            torch_dtype=torch_dtype,
        )

        rolling_forecasts, actual_values, previous_actual_values, test_dates = run_rolling_forecast(
            finetuned_pipeline, full_data, test_start_idx, config
        )

        metrics = calculate_metrics(rolling_forecasts, actual_values, previous_actual_values,
                                    config['forecast']['quantile_levels'])

        plot_forecast(full_data, test_start_idx, rolling_forecasts, actual_values, test_dates, metrics, config)
        plot_metrics_chart(rolling_forecasts, actual_values, test_dates, metrics, config)

    print(f"All results saved to: {output_dir}/")

if __name__ == "__main__":
    main()
