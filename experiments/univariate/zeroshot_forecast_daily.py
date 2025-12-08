"""
Zero-shot forecasting on Cotton Futures data using Chronos-2.
Predicts on the last 7 days (holdout) and evaluates against actual values.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

# Configuration
DATA_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Cotton_Futures.csv"
MODEL_NAME = "amazon/chronos-2"  # 120M parameter model
PREDICTION_LENGTH = 7  # 7 days
CONTEXT_LENGTH = 1024  # Use last 1024 days for context (~2.8 years)
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_data(filepath):
    """Load and preprocess Cotton Futures data."""
    # CSV structure: Row 0=headers, Row 1=tickers, Row 2="Date" label, Row 3+=data
    # Skip rows 1 and 2, use first column as date index
    df = pd.read_csv(filepath, header=0, skiprows=[1, 2], index_col=0)

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    # Sort by date
    df = df.sort_index()

    # Extract Close prices
    close_prices = df['Close'].values

    print(f"Loaded {len(close_prices)} data points")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Price range: ${close_prices.min():.2f} to ${close_prices.max():.2f}")

    return df, close_prices

def run_inference(close_prices_train, context_length):
    """Run zero-shot forecasting using Chronos-2."""
    print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Use the last context_length points as input
    context = close_prices_train[-context_length:]
    # Shape needs to be (n_series, n_variates, history_length) = (1, 1, context_length)
    context_tensor = torch.tensor(context).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, context_length)

    print(f"Running inference with context of {len(context)} days...")
    print(f"Forecasting {PREDICTION_LENGTH} days ahead...")

    # Run prediction (returns all model's default quantiles automatically)
    predictions = pipeline.predict(
        context_tensor,
        prediction_length=PREDICTION_LENGTH,
    )

    # Extract predictions: list element has shape (n_variates, num_quantiles, prediction_length)
    # Model returns predictions for its default quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # For univariate: (1, num_quantiles, prediction_length), so squeeze the first dimension
    forecast = predictions[0].squeeze(0).numpy()  # Shape: (num_quantiles, prediction_length)

    print(f"Model quantiles: {pipeline.quantiles}")
    print(f"Forecast shape: {forecast.shape}")

    return forecast

def calculate_metrics(forecast, actual):
    """Calculate forecast accuracy metrics."""
    # Use median forecast (0.5 quantile)
    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecast = forecast[median_idx, :]

    # Calculate metrics
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
    print("FORECAST EVALUATION METRICS")
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

        # Convert pandas Timestamp to string for formatting
        date_str = date.strftime('%Y-%m-%d')
        print(f"{date_str:<12} "
              f"${actual_val:<9.2f} "
              f"${pred_val:<9.2f} "
              f"${error:<9.2f} "
              f"{error_pct:<9.2f}%")

    print("="*80)

def plot_results(df, close_prices_train, forecast, actual_prices, test_dates):
    """Plot last 14 days of historical data + 7-day prediction vs actual."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get last 14 days of training data (before the test period)
    train_end_idx = len(close_prices_train)
    historical_dates = df.index[train_end_idx - 14:train_end_idx]
    historical_prices = close_prices_train[-14:]

    # Plot last 14 days of historical data
    ax.plot(historical_dates, historical_prices, label='Historical Data',
            color='blue', linewidth=2, marker='o', markersize=4)

    # Plot actual test data (ground truth)
    ax.plot(test_dates, actual_prices, label='Actual (Test Period)',
            color='green', linewidth=2, marker='o', markersize=6)

    # Plot median forecast (0.5 quantile)
    median_idx = QUANTILE_LEVELS.index(0.5)
    ax.plot(test_dates, forecast[median_idx, :], label='Median Forecast',
            color='red', linewidth=2, marker='s', markersize=6, linestyle='--')

    # Plot uncertainty bands
    # 80% interval (0.1 - 0.9)
    q10_idx = QUANTILE_LEVELS.index(0.1)
    q90_idx = QUANTILE_LEVELS.index(0.9)
    ax.fill_between(test_dates, forecast[q10_idx, :], forecast[q90_idx, :],
                     alpha=0.2, color='red', label='80% Interval (Q10-Q90)')

    # 60% interval (0.2 - 0.8)
    q20_idx = QUANTILE_LEVELS.index(0.2)
    q80_idx = QUANTILE_LEVELS.index(0.8)
    ax.fill_between(test_dates, forecast[q20_idx, :], forecast[q80_idx, :],
                     alpha=0.3, color='red', label='60% Interval (Q20-Q80)')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cotton Futures Price (USD)', fontsize=12)
    ax.set_title('Cotton Futures 7-Day Forecast vs Actual (Chronos-2 Zero-Shot)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = 'experiments/univariate/zeroshot_forecast_daily_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

def main():
    print("="*80)
    print("Cotton Futures Zero-Shot Forecasting with Chronos-2")
    print("Evaluation Mode: Predicting on Last 7 Days (Holdout)")
    print("="*80)

    # Load data
    df, close_prices = load_data(DATA_PATH)

    # Split data: use all except last 7 days for training
    train_end_idx = len(close_prices) - PREDICTION_LENGTH
    close_prices_train = close_prices[:train_end_idx]
    close_prices_test = close_prices[train_end_idx:]

    # Get test dates
    test_dates = df.index[train_end_idx:]

    print(f"\nTrain period: {df.index[0]} to {df.index[train_end_idx - 1]}")
    print(f"Test period:  {test_dates[0]} to {test_dates[-1]}")
    print(f"Train size: {len(close_prices_train)} days")
    print(f"Test size:  {len(close_prices_test)} days")

    # Run inference on training data to predict test period
    forecast = run_inference(close_prices_train, CONTEXT_LENGTH)

    # Calculate metrics
    metrics = calculate_metrics(forecast, close_prices_test)

    # Print metrics
    print_metrics(metrics, close_prices_test, test_dates)

    # Plot results (last 14 days + predictions)
    plot_results(df, close_prices_train, forecast, close_prices_test, test_dates)

    print("\nDone!")

if __name__ == "__main__":
    main()
