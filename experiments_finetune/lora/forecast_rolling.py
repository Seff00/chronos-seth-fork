"""
Rolling Forecasting on Cotton Futures using Chronos-2.
Uses a rolling window approach: predict d1 with history, then predict d2 with history+actual_d1, etc.
Uses Crude Oil and Copper Futures as covariates.
"""

import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

MODEL_NAME = "./experiments_finetune/lora/checkpoint"
PREDICTION_DAYS = 7  # Rolling prediction for last N days
CONTEXT_LENGTH = 1024  # Use last 1024 days for context
PLOT_METRICS = True  # Set to False for long prediction periods to avoid cluttered plots
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_csv_data(filepath, asset_name):
    """Load and preprocess commodity CSV data."""
    # CSV structure: Row 0=headers, Row 1=tickers, Row 2="Date" label, Row 3+=data
    df = pd.read_csv(filepath, header=0, skiprows=[1, 2], index_col=0)

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    # Sort by date
    df = df.sort_index()

    # Extract Close prices and rename
    prices = df['Close'].rename(asset_name)

    print(f"{asset_name:15} - {len(prices):5} points, "
          f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}, "
          f"range: ${prices.min():.2f} to ${prices.max():.2f}")

    return prices

def align_data(target, covariates_dict):
    """
    Align target and all covariate time series to common dates.

    Parameters
    ----------
    target : pd.Series
        Target time series
    covariates_dict : dict
        Dictionary of {name: series} for all covariates
    """
    # Combine target and all covariates
    data_dict = {'target': target}
    data_dict.update(covariates_dict)

    combined = pd.DataFrame(data_dict)

    print(f"\nBefore alignment: {len(combined)} dates")
    print(f"Missing values:")
    print(f"  Target: {combined['target'].isna().sum()}")
    for name in covariates_dict.keys():
        print(f"  {name}: {combined[name].isna().sum()}")

    # Forward fill missing values
    combined = combined.fillna(method='ffill')

    # Drop any remaining NaN (at the start)
    combined = combined.dropna()

    print(f"After alignment:  {len(combined)} dates")
    print(f"Date range: {combined.index.min().strftime('%Y-%m-%d')} to {combined.index.max().strftime('%Y-%m-%d')}")

    return combined

# ============================================================================
# Classification Functions
# ============================================================================

def get_direction(current_price, previous_price):
    """
    Determine direction of price movement.

    Parameters
    ----------
    current_price : float
        Current day's price
    previous_price : float
        Previous day's price

    Returns
    -------
    direction : str
        'UP' if price increased, 'DOWN' if price decreased or stayed same
    """
    return 'UP' if current_price > previous_price else 'DOWN'

def calculate_classification_metrics(predicted_directions, actual_directions):
    """
    Calculate classification metrics for direction prediction.

    Parameters
    ----------
    predicted_directions : list of str
        Predicted directions ('UP' or 'DOWN')
    actual_directions : list of str
        Actual directions ('UP' or 'DOWN')

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    # Convert to binary (1 = UP, 0 = DOWN)
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
        'f1_score': f1,
        'y_pred': y_pred,
        'y_true': y_true
    }

# ============================================================================
# Prediction Functions
# ============================================================================

def predict_single_day(pipeline, context_data, covariate_columns):
    """
    Predict a single day using context data.

    Parameters
    ----------
    pipeline : Chronos2Pipeline
        Loaded model pipeline
    context_data : pd.DataFrame
        Historical data with columns: target + covariates
    covariate_columns : list of str
        Column names for covariates

    Returns
    -------
    forecast : np.ndarray
        Shape (num_quantiles,) with predictions for all quantiles
    """
    # Get context (last CONTEXT_LENGTH points)
    context = context_data.tail(CONTEXT_LENGTH)

    # Prepare target
    target = context['target'].values

    # Build past covariates dictionary dynamically
    past_covariates = {}
    for col in covariate_columns:
        # Use column name as key (sanitize for model)
        key = col.lower().replace(' ', '_')
        past_covariates[key] = context[col].values

    # Build input dictionary
    input_dict = {
        "target": target,
        "past_covariates": past_covariates
    }

    # Run prediction (predict 1 day ahead)
    predictions = pipeline.predict(
        [input_dict],
        prediction_length=1,
    )

    # Extract predictions: (1, num_quantiles, 1) -> (num_quantiles,)
    forecast = predictions[0].squeeze().numpy()

    return forecast

def run_rolling_forecast(pipeline, combined_data, test_start_idx, covariate_columns):
    """
    Run rolling forecast for the last PREDICTION_DAYS days.

    For each day:
    - Day 1: Use historical data to predict day 1
    - Day 2: Use historical data + actual day 1 to predict day 2
    - Day 3: Use historical data + actual days 1-2 to predict day 3
    - etc.

    Parameters
    ----------
    covariate_columns : list of str
        Column names for covariates (excluding 'target')
    """
    print(f"\n" + "="*80)
    print("Running Rolling Forecast...")
    print("="*80)

    rolling_forecasts = []
    actual_values = []
    previous_actual_values = []
    test_dates = []

    for day_offset in range(PREDICTION_DAYS):
        current_idx = test_start_idx + day_offset
        prediction_date = combined_data.index[current_idx]

        # Context: everything before current_idx (includes actual values from previous predictions)
        context_data = combined_data.iloc[:current_idx]

        # Actual value for this day
        actual_value = combined_data['target'].iloc[current_idx]

        # Previous day's actual value (for direction comparison)
        previous_actual = combined_data['target'].iloc[current_idx - 1]

        print(f"\nDay {day_offset + 1}/{PREDICTION_DAYS}: Predicting {prediction_date.strftime('%Y-%m-%d')}")
        print(f"  Context size: {len(context_data)} days")

        # Predict single day
        forecast = predict_single_day(pipeline, context_data, covariate_columns)

        # Get median prediction
        median_idx = QUANTILE_LEVELS.index(0.5)
        median_pred = forecast[median_idx]

        # Calculate directions
        predicted_direction = get_direction(median_pred, previous_actual)
        actual_direction = get_direction(actual_value, previous_actual)

        # Store results
        rolling_forecasts.append(forecast)
        actual_values.append(actual_value)
        previous_actual_values.append(previous_actual)
        test_dates.append(prediction_date)

        # Calculate error
        error = actual_value - median_pred

        # Display with classification
        correct = "✓" if predicted_direction == actual_direction else "✗"
        print(f"  Previous: ${previous_actual:.2f}")
        print(f"  Actual: ${actual_value:.2f}, Predicted: ${median_pred:.2f}, Error: ${error:.2f}")
        print(f"  Direction - Predicted: {predicted_direction}, Actual: {actual_direction} {correct}")

    # Convert to arrays
    rolling_forecasts = np.array(rolling_forecasts)  # (PREDICTION_DAYS, num_quantiles)
    actual_values = np.array(actual_values)  # (PREDICTION_DAYS,)
    previous_actual_values = np.array(previous_actual_values)  # (PREDICTION_DAYS,)

    print("\n" + "="*80)

    return rolling_forecasts, actual_values, previous_actual_values, test_dates

def calculate_metrics(forecasts, actual_values, previous_actual_values):
    """
    Calculate regression error metrics and classification metrics.

    Parameters
    ----------
    forecasts : np.ndarray
        Shape (PREDICTION_DAYS, num_quantiles)
    actual_values : np.ndarray
        Shape (PREDICTION_DAYS,)
    previous_actual_values : np.ndarray
        Shape (PREDICTION_DAYS,) - previous day's actual values
    """
    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecasts = forecasts[:, median_idx]  # (PREDICTION_DAYS,)

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

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_forecast(combined_data, test_start_idx, forecasts, actual_values, test_dates, metrics, target_name):
    """Plot time series forecast vs actual (main plot) with direction-based coloring."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get historical data for context (last 14 days before test period)
    historical_data = combined_data.iloc[test_start_idx - 14:test_start_idx]
    historical_dates = historical_data.index
    historical_prices = historical_data['target'].values

    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecasts = forecasts[:, median_idx]

    # Plot historical data
    ax.plot(historical_dates, historical_prices, label='Historical Data',
            color='blue', linewidth=2, marker='o', markersize=4)

    # Plot actual values
    ax.plot(test_dates, actual_values, label='Actual',
            color='darkblue', linewidth=2.5, marker='D', markersize=5)

    # Plot predicted values with dynamic colors based on direction
    predicted_dirs = metrics['predicted_directions']

    # Plot connecting line (gray dashed)
    ax.plot(test_dates, median_forecasts, color='gray', linewidth=1.5,
            linestyle='--', alpha=0.5, zorder=1)

    # Plot each prediction point with color based on direction
    for i, (date, pred, direction) in enumerate(zip(test_dates, median_forecasts, predicted_dirs)):
        color = 'green' if direction == 'UP' else 'red'
        label = f'Predicted {direction}' if i == 0 or (i > 0 and direction != predicted_dirs[i-1]) else None
        ax.scatter(date, pred, color=color, s=75, marker='s',
                  edgecolor='black', linewidth=1, zorder=3, label=label)

    # Build covariate list for title
    covariate_names = ', '.join([cov['name'] for cov in COVARIATES])

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{target_name} Price (USD)', fontsize=12)
    ax.set_title(f'{target_name} Rolling Forecast vs Actual\n' +
                 f'Covariates: {covariate_names}\n' +
                 '(Prediction colors: Green=UP, Red=DOWN)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save plot
    output_path = 'experiments_finetune/lora/forecast_rolling_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nForecast plot saved to: {output_path}")
    plt.close()

def plot_metrics(forecasts, actual_values, test_dates, metrics):
    """Plot regression and classification metrics by day (separate plot)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecasts = forecasts[:, median_idx]

    # Plot 1: Regression error analysis
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

    # Plot 2: Classification analysis (direction prediction)
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

    # Add direction labels on bars
    for i, (pred, actual, corr) in enumerate(zip(predicted_dirs, actual_dirs, correct)):
        label = f"{pred}\n({actual})"
        y_pos = 0.5
        ax2.text(i, y_pos, label, ha='center', va='center', fontsize=8,
                color='white' if corr else 'black', fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_path = 'experiments_finetune/lora/forecast_rolling_metrics_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {output_path}")
    plt.close()

def main():
    # Redirect output to file
    output_file = 'experiments_finetune/lora/forecast_rolling_output.txt'
    original_stdout = sys.stdout

    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f

        # Build covariate names list
        covariate_names = ', '.join([cov['name'] for cov in COVARIATES])

        print("="*80)
        print(f"{TARGET_NAME} Rolling Forecasting with Chronos-2")
        print(f"Covariates: {covariate_names}")
        print("Method: Rolling window (predict d1, then predict d2 with d1 actual, etc.)")
        print("="*80)

        # Load target
        print("\nLoading datasets...")
        import os
        target_path = os.path.join(DATA_FOLDER, TARGET_FILE)
        target = load_csv_data(target_path, TARGET_NAME)

        # Load all covariates dynamically
        covariates_dict = {}
        for cov in COVARIATES:
            cov_path = os.path.join(DATA_FOLDER, cov['file'])
            covariates_dict[cov['name']] = load_csv_data(cov_path, cov['name'])

        # Align data to common dates
        combined_data = align_data(target, covariates_dict)

        # Get covariate column names (all columns except 'target')
        covariate_columns = [col for col in combined_data.columns if col != 'target']

        # Define test period (last PREDICTION_DAYS days)
        test_start_idx = len(combined_data) - PREDICTION_DAYS
        test_dates = combined_data.index[test_start_idx:]

        print(f"\nTrain period: {combined_data.index[0].strftime('%Y-%m-%d')} to " +
              f"{combined_data.index[test_start_idx - 1].strftime('%Y-%m-%d')}")
        print(f"Test period:  {test_dates[0].strftime('%Y-%m-%d')} to " +
              f"{test_dates[-1].strftime('%Y-%m-%d')}")
        print(f"Train size: {test_start_idx} days")
        print(f"Test size:  {PREDICTION_DAYS} days")

        # Load model
        print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
        pipeline = Chronos2Pipeline.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Run rolling forecast
        rolling_forecasts, actual_values, previous_actual_values, test_dates = run_rolling_forecast(
            pipeline, combined_data, test_start_idx, covariate_columns
        )

        # Calculate metrics
        metrics = calculate_metrics(rolling_forecasts, actual_values, previous_actual_values)

        # Print metrics
        print_metrics(metrics, actual_values, test_dates)

        # Plot forecast (time series - always generated)
        plot_forecast(combined_data, test_start_idx, rolling_forecasts, actual_values, test_dates, metrics, TARGET_NAME)

        # Plot metrics (regression + classification - optional)
        if PLOT_METRICS:
            plot_metrics(rolling_forecasts, actual_values, test_dates, metrics)
        else:
            print("\nMetrics plot skipped (PLOT_METRICS=False)")

        print("\nDone!")
        print("\nNOTE: This rolling forecast updates context with actual values after each prediction,")
        print("      simulating real-world sequential forecasting where we learn from recent outcomes.")
        print("\nCLASSIFICATION: Direction prediction compares each day's price to the previous day.")
        print("                UP = price increased, DOWN = price decreased or stayed same.")

    # Restore stdout
    sys.stdout = original_stdout
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
