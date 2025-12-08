"""
Zero-shot forecasting on Cotton Futures data using Chronos-2 with WEEKLY aggregation.
Aggregates daily data to weekly to capture full 11-year history in context window.
Predicts on the last 4 weeks (holdout) and evaluates against actual values.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

# Configuration
DATA_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Cotton_Futures.csv"
MODEL_NAME = "amazon/chronos-2"  # 120M parameter model
PREDICTION_LENGTH = 4  # 4 weeks ahead (1 month)
CONTEXT_LENGTH = 2048  # Use full model capacity
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_and_aggregate_data(filepath):
    """Load daily data and aggregate to weekly frequency."""
    # CSV structure: Row 0=headers, Row 1=tickers, Row 2="Date" label, Row 3+=data
    # Skip rows 1 and 2, use first column as date index
    df = pd.read_csv(filepath, header=0, skiprows=[1, 2], index_col=0)

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    # Sort by date
    df = df.sort_index()

    print(f"Loaded {len(df)} daily data points")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Aggregate to weekly frequency (using last business day of week)
    # 'W-FRI' = week ending on Friday (common for financial data)
    weekly_df = df.resample('W-FRI').agg({
        'Close': 'last',      # Last closing price of the week
        'High': 'max',        # Highest price of the week
        'Low': 'min',         # Lowest price of the week
        'Open': 'first',      # First opening price of the week
        'Volume': 'sum'       # Total volume for the week
    }).dropna()

    # Reset index to get Date column back
    weekly_df.reset_index(inplace=True)

    # Extract weekly close prices
    weekly_prices = weekly_df['Close'].values

    print(f"\nAfter weekly aggregation: {len(weekly_prices)} weeks")
    print(f"Date range: {weekly_df['Date'].min()} to {weekly_df['Date'].max()}")
    print(f"Price range: ${weekly_prices.min():.2f} to ${weekly_prices.max():.2f}")
    print(f"Can fit {min(len(weekly_prices), CONTEXT_LENGTH)} weeks in context "
          f"({min(len(weekly_prices), CONTEXT_LENGTH) / 52:.1f} years)")

    return weekly_df, weekly_prices

def run_inference(weekly_prices_train, context_length):
    """Run zero-shot forecasting using Chronos-2."""
    print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Use the last context_length weeks as input
    context = weekly_prices_train[-context_length:]
    # Shape needs to be (n_series, n_variates, history_length) = (1, 1, context_length)
    context_tensor = torch.tensor(context).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, context_length)

    print(f"Running inference with context of {len(context)} weeks ({len(context)/52:.1f} years)...")
    print(f"Forecasting {PREDICTION_LENGTH} weeks ahead...")

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
    print("WEEK-BY-WEEK COMPARISON")
    print("="*80)
    print(f"{'Week Ending':<15} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10}")
    print("-"*80)

    for i, date in enumerate(dates):
        actual_val = actual[i]
        pred_val = metrics['median_forecast'][i]
        error = metrics['errors'][i]
        error_pct = (error / actual_val) * 100

        # Convert numpy datetime64 to pandas Timestamp for strftime
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        print(f"{date_str:<15} "
              f"${actual_val:<9.2f} "
              f"${pred_val:<9.2f} "
              f"${error:<9.2f} "
              f"{error_pct:<9.2f}%")

    print("="*80)

def print_comparison_summary(weekly_df, weekly_prices):
    """Print comparison between weekly aggregation vs daily approach."""
    print("\n" + "="*80)
    print("WEEKLY AGGREGATION BENEFITS")
    print("="*80)

    total_days = 2771  # Approximate from previous check
    total_weeks = len(weekly_prices)

    print(f"Daily approach:")
    print(f"  - Total data points: ~{total_days} days (~11 years)")
    print(f"  - Context limit: 1024 days (~2.8 years)")
    print(f"  - Missing history: ~{total_days - 1024} days (~{(total_days - 1024)/365:.1f} years)")
    print(f"  - Coverage: {1024/total_days*100:.1f}% of available data")

    print(f"\nWeekly approach:")
    print(f"  - Total data points: {total_weeks} weeks (~{total_weeks/52:.1f} years)")
    print(f"  - Context limit: {min(CONTEXT_LENGTH, total_weeks)} weeks ({min(CONTEXT_LENGTH, total_weeks)/52:.1f} years)")

    if total_weeks <= CONTEXT_LENGTH:
        print(f"  - Missing history: NONE - FULL HISTORY CAPTURED! ✓")
        print(f"  - Coverage: 100% of available data ✓")
    else:
        print(f"  - Missing history: {total_weeks - CONTEXT_LENGTH} weeks ({(total_weeks - CONTEXT_LENGTH)/52:.1f} years)")
        print(f"  - Coverage: {CONTEXT_LENGTH/total_weeks*100:.1f}% of available data")

    print(f"\nKey advantages:")
    print(f"  ✓ Captures long-term trends and multi-year cycles")
    print(f"  ✓ Sees seasonal patterns across full history")
    print(f"  ✓ Better for commodities with long-term price dynamics")
    print("="*80)

def plot_results(weekly_df, weekly_prices_train, forecast, actual_prices, test_dates):
    """Plot last 8 weeks of historical data + 4-week prediction vs actual."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get last 8 weeks of training data (before the test period)
    train_end_idx = len(weekly_prices_train)
    historical_dates = weekly_df['Date'].values[train_end_idx - 8:train_end_idx]
    historical_prices = weekly_prices_train[-8:]

    # Plot last 8 weeks of historical data
    ax.plot(historical_dates, historical_prices, label='Historical Data',
            color='blue', linewidth=2, marker='o', markersize=5)

    # Plot actual test data (ground truth)
    ax.plot(test_dates, actual_prices, label='Actual (Test Period)',
            color='green', linewidth=2, marker='o', markersize=7)

    # Plot median forecast (0.5 quantile)
    median_idx = QUANTILE_LEVELS.index(0.5)
    ax.plot(test_dates, forecast[median_idx, :], label='Median Forecast',
            color='red', linewidth=2.5, marker='s', markersize=7, linestyle='--')

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
    ax.set_title('Cotton Futures 4-Week Forecast vs Actual - Weekly Aggregation (Chronos-2)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = 'experiments/cotton_forecast_weekly_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

def main():
    print("="*80)
    print("Cotton Futures Zero-Shot Forecasting with Chronos-2 (WEEKLY AGGREGATION)")
    print("Evaluation Mode: Predicting on Last 4 Weeks (Holdout)")
    print("="*80)

    # Load and aggregate data
    weekly_df, weekly_prices = load_and_aggregate_data(DATA_PATH)

    # Print comparison
    print_comparison_summary(weekly_df, weekly_prices)

    # Split data: use all except last 4 weeks for training
    train_end_idx = len(weekly_prices) - PREDICTION_LENGTH
    weekly_prices_train = weekly_prices[:train_end_idx]
    weekly_prices_test = weekly_prices[train_end_idx:]

    # Get test dates
    test_dates = weekly_df['Date'].values[train_end_idx:]

    print(f"\nTrain period: {weekly_df['Date'].iloc[0]} to {weekly_df['Date'].iloc[train_end_idx - 1]}")
    print(f"Test period:  {test_dates[0]} to {test_dates[-1]}")
    print(f"Train size: {len(weekly_prices_train)} weeks ({len(weekly_prices_train)/52:.1f} years)")
    print(f"Test size:  {len(weekly_prices_test)} weeks")

    # Run inference on training data to predict test period
    forecast = run_inference(weekly_prices_train, CONTEXT_LENGTH)

    # Calculate metrics
    metrics = calculate_metrics(forecast, weekly_prices_test)

    # Print metrics
    print_metrics(metrics, weekly_prices_test, test_dates)

    # Plot results (last 8 weeks + predictions)
    plot_results(weekly_df, weekly_prices_train, forecast, weekly_prices_test, test_dates)

    print("\nDone!")
    print("\nNOTE: This weekly forecast predicts prices at week-end (Friday close).")
    print("      Each prediction represents the price ~7 days ahead from previous point.")

if __name__ == "__main__":
    main()
