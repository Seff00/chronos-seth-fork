"""
Multivariate forecasting using Chronos-2.
Predicts on the last N days (holdout) and evaluates against actual values.
"""

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

MODEL_NAME = "experiments_finetune/lora/checkpoint/finetuned-ckpt"
PREDICTION_LENGTH = 7
CONTEXT_LENGTH = 365  # Match training context length
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Test set configuration
# TEST_SET_SIZE = 365  # Hold-out test set is last 365 days (1 year)
TEST_SET_SIZE = 7    # Hold-out test set is last 7 days

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

def create_calendar_features(dates, prediction_dates):
    """Create calendar features (month, day of week) for past and future."""
    # Past features (for context)
    past_features = pd.DataFrame({
        'month': dates.month,
        'day_of_week': dates.dayofweek,
        'quarter': dates.quarter,
    }, index=dates)

    # Future features (for prediction period)
    future_features = pd.DataFrame({
        'month': prediction_dates.month,
        'day_of_week': prediction_dates.dayofweek,
        'quarter': prediction_dates.quarter,
    }, index=prediction_dates)

    return past_features, future_features

def run_inference(combined_data, test_start_idx, covariate_columns):
    """
    Run multivariate forecasting using Chronos-2.

    Parameters
    ----------
    combined_data : pd.DataFrame
        Full aligned dataset
    test_start_idx : int
        Index where test set starts (beginning of hold-out test set)
    covariate_columns : list of str
        Column names for covariates (excluding 'target')
    """
    print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Training data: everything before test set
    train_data = combined_data.iloc[:test_start_idx]

    # Get context data (last CONTEXT_LENGTH points from training data)
    context_data = train_data.tail(CONTEXT_LENGTH)

    print(f"\nContext window for prediction:")
    print(f"  From: {context_data.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:   {context_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Size: {len(context_data)} days")

    # Prepare target
    target = context_data['target'].values

    # Prepare calendar features
    past_calendar, future_calendar = create_calendar_features(
        context_data.index,
        combined_data.index[test_start_idx:test_start_idx + PREDICTION_LENGTH]
    )

    # Build past covariates dictionary dynamically
    past_covariates = {}
    for col in covariate_columns:
        # Use column name as key (sanitize for model)
        key = col.lower().replace(' ', '_')
        past_covariates[key] = context_data[col].values

    # Add calendar features to past covariates
    past_covariates['month'] = past_calendar['month'].values
    past_covariates['day_of_week'] = past_calendar['day_of_week'].values
    past_covariates['quarter'] = past_calendar['quarter'].values

    # Build future covariates (calendar only)
    future_covariates = {
        "month": future_calendar['month'].values,
        "day_of_week": future_calendar['day_of_week'].values,
        "quarter": future_calendar['quarter'].values,
    }

    # Build input dictionary
    input_dict = {
        "target": target,
        "past_covariates": past_covariates,
        "future_covariates": future_covariates
    }

    print(f"\nInput structure:")
    print(f"  Target:        shape {target.shape}")
    print(f"  Past covariates:")
    for name, values in input_dict['past_covariates'].items():
        print(f"    - {name:15}: shape {values.shape}")
    print(f"  Future covariates:")
    for name, values in input_dict['future_covariates'].items():
        print(f"    - {name:15}: shape {values.shape}")

    print(f"\nRunning multivariate inference...")
    print(f"Context: {len(target)} days")
    print(f"Forecasting {PREDICTION_LENGTH} days ahead...")

    # Run prediction (input must be a LIST of dicts, where each dict is one task)
    predictions = pipeline.predict(
        [input_dict],  # Wrap in list!
        prediction_length=PREDICTION_LENGTH,
    )

    # Extract predictions: (n_variates, num_quantiles, prediction_length)
    # For univariate target, squeeze first dimension
    forecast = predictions[0].squeeze(0).numpy()  # (num_quantiles, prediction_length)

    print(f"Model quantiles: {pipeline.quantiles}")
    print(f"Forecast shape: {forecast.shape}")

    return forecast

def calculate_metrics(forecast, actual):
    """Calculate forecast accuracy metrics."""
    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecast = forecast[median_idx, :]

    errors = actual - median_forecast
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(errors / actual)) * 100

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'median_forecast': median_forecast,
        'errors': errors
    }

def print_metrics(metrics, actual, dates):
    """Print detailed metrics and comparison."""
    print("\n" + "="*80)
    print("FORECAST EVALUATION METRICS (Multivariate)")
    print("="*80)
    print(f"Mean Absolute Error (MAE):       ${metrics['mae']:.4f}")
    print(f"Mean Squared Error (MSE):        ${metrics['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE):  ${metrics['rmse']:.4f}")
    print(f"Mean Absolute Percentage Error:  {metrics['mape']:.2f}%")
    print("="*80)

    print("\n" + "="*80)
    print("DAY-BY-DAY COMPARISON")
    print("="*80)
    print(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10}")
    print("-"*80)

    for i, date in enumerate(dates):
        actual_val = actual[i]
        pred_val = metrics['median_forecast'][i]
        error = metrics['errors'][i]
        error_pct = (error / actual_val) * 100

        date_str = date.strftime('%Y-%m-%d')
        print(f"{date_str:<12} "
              f"${actual_val:<9.2f} "
              f"${pred_val:<9.2f} "
              f"${error:<9.2f} "
              f"{error_pct:<9.2f}%")

    print("="*80)

def plot_results(combined_data, test_start_idx, forecast, actual_prices, test_dates, target_name):
    """Plot last 14 days of historical data + prediction vs actual."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get last 14 days of training data
    historical_data = combined_data.iloc[test_start_idx - 14:test_start_idx]
    historical_dates = historical_data.index
    historical_prices = historical_data['target'].values

    # Plot last 14 days of historical data
    ax.plot(historical_dates, historical_prices, label=f'Historical Data ({target_name})',
            color='blue', linewidth=2, marker='o', markersize=4)

    # Plot actual test data (ground truth)
    ax.plot(test_dates, actual_prices, label='Actual (Test Period)',
            color='darkblue', linewidth=2.5, marker='D', markersize=7)

    # Determine forecast direction (compare first forecast with last historical value)
    median_idx = QUANTILE_LEVELS.index(0.5)
    last_historical_value = historical_prices[-1]
    first_forecast_value = forecast[median_idx, 0]

    # Choose color based on direction: green if up, red if down
    if first_forecast_value > last_historical_value:
        forecast_color = 'green'
        direction_label = 'UP'
    else:
        forecast_color = 'red'
        direction_label = 'DOWN'

    # Plot median forecast
    ax.plot(test_dates, forecast[median_idx, :], label=f'Median Forecast ({direction_label})',
            color=forecast_color, linewidth=2, marker='s', markersize=6, linestyle='--')

    # Plot uncertainty bands
    q10_idx = QUANTILE_LEVELS.index(0.1)
    q90_idx = QUANTILE_LEVELS.index(0.9)
    ax.fill_between(test_dates, forecast[q10_idx, :], forecast[q90_idx, :],
                     alpha=0.2, color=forecast_color, label='80% Interval (Q10-Q90)')

    q20_idx = QUANTILE_LEVELS.index(0.2)
    q80_idx = QUANTILE_LEVELS.index(0.8)
    ax.fill_between(test_dates, forecast[q20_idx, :], forecast[q80_idx, :],
                     alpha=0.3, color=forecast_color, label='60% Interval (Q20-Q80)')

    # Build covariate list for title
    covariate_names = ', '.join([cov['name'] for cov in COVARIATES])

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{target_name} Price (USD)', fontsize=12)
    ax.set_title(f'{target_name} Multivariate Forecast vs Actual\n' +
                 f'Covariates: {covariate_names}, Calendar Features',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = 'experiments_finetune/lora/forecast_direct_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

def main():
    # Build covariate names list
    covariate_names = ', '.join([cov['name'] for cov in COVARIATES])

    print("="*80)
    print(f"{TARGET_NAME} Direct 7-Day Forecasting with Fine-Tuned Chronos-2")
    print(f"Covariates: {covariate_names}, Calendar Features")
    print(f"Model: LoRA Fine-Tuned (PREDICTION_LENGTH={PREDICTION_LENGTH})")
    print(f"Evaluation: First {PREDICTION_LENGTH} days of hold-out test set")
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

    # Split: Hold-out test set is last TEST_SET_SIZE days
    # We predict on the FIRST PREDICTION_LENGTH days of this test set
    test_set_start = len(combined_data) - TEST_SET_SIZE
    test_start_idx = test_set_start
    test_end_idx = test_start_idx + PREDICTION_LENGTH

    test_dates = combined_data.index[test_start_idx:test_end_idx]
    actual_prices = combined_data['target'].iloc[test_start_idx:test_end_idx].values

    print(f"\n" + "="*80)
    print("DATA SPLIT FOR EVALUATION")
    print("="*80)
    print(f"Training period: {combined_data.index[0].strftime('%Y-%m-%d')} to " +
          f"{combined_data.index[test_set_start - 1].strftime('%Y-%m-%d')}")
    print(f"  Size: {test_set_start} days")
    print()
    print(f"Hold-out test set: {combined_data.index[test_set_start].strftime('%Y-%m-%d')} to " +
          f"{combined_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Size: {TEST_SET_SIZE} days (NOT used in training)")
    print()
    print(f"Evaluation period (first {PREDICTION_LENGTH} days of test set):")
    print(f"  {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Size: {PREDICTION_LENGTH} days")
    print("="*80)

    # Run multivariate inference
    forecast = run_inference(combined_data, test_start_idx, covariate_columns)

    # Calculate metrics
    metrics = calculate_metrics(forecast, actual_prices)

    # Print metrics
    print_metrics(metrics, actual_prices, test_dates)

    # Plot results
    plot_results(combined_data, test_start_idx, forecast, actual_prices, test_dates, TARGET_NAME)

    print("\nDone!")
    print(f"\nNOTE: This forecast uses {covariate_names} as covariates,")
    print("      plus calendar features (month, day of week, quarter).")

if __name__ == "__main__":
    main()
