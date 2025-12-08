"""
Zero-shot forecasting on Cotton Futures data using Chronos-2.
Forecasts 7 days ahead and plots results with quantile uncertainty bands.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

# Configuration
DATA_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Cotton_Futures.csv"
MODEL_NAME = "amazon/chronos-2-120m"
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

def run_inference(close_prices, context_length):
    """Run zero-shot forecasting using Chronos-2."""
    print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Use the last context_length points as input
    context = close_prices[-context_length:]
    context_tensor = torch.tensor(context).unsqueeze(0)  # Shape: (1, context_length)

    print(f"Running inference with context of {len(context)} days...")
    print(f"Forecasting {PREDICTION_LENGTH} days ahead...")

    # Run prediction
    predictions = pipeline.predict(
        context_tensor,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
    )

    # Extract predictions: shape (1, num_quantiles, prediction_length)
    forecast = predictions[0].numpy()  # Shape: (num_quantiles, prediction_length)

    return forecast

def plot_results(df, close_prices, forecast, context_length):
    """Plot historical data, context window, and forecast with uncertainty bands."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get the last portion of data for visualization
    viz_length = min(100, len(close_prices))
    viz_dates = df.index.values[-viz_length:]
    viz_prices = close_prices[-viz_length:]

    # Context dates (last context_length points)
    context_dates = df.index.values[-context_length:]
    context_prices = close_prices[-context_length:]

    # Generate future dates for forecast
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH)

    # Plot historical data (lighter)
    ax.plot(viz_dates, viz_prices, label='Historical Data', color='gray', alpha=0.5, linewidth=1)

    # Plot context window (darker)
    ax.plot(context_dates, context_prices, label='Context Window', color='blue', linewidth=2)

    # Plot median forecast (0.5 quantile)
    median_idx = QUANTILE_LEVELS.index(0.5)
    ax.plot(future_dates, forecast[median_idx, :], label='Median Forecast', color='red', linewidth=2)

    # Plot uncertainty bands
    # 80% interval (0.1 - 0.9)
    q10_idx = QUANTILE_LEVELS.index(0.1)
    q90_idx = QUANTILE_LEVELS.index(0.9)
    ax.fill_between(future_dates, forecast[q10_idx, :], forecast[q90_idx, :],
                     alpha=0.2, color='red', label='80% Interval (Q10-Q90)')

    # 60% interval (0.2 - 0.8)
    q20_idx = QUANTILE_LEVELS.index(0.2)
    q80_idx = QUANTILE_LEVELS.index(0.8)
    ax.fill_between(future_dates, forecast[q20_idx, :], forecast[q80_idx, :],
                     alpha=0.3, color='red', label='60% Interval (Q20-Q80)')

    # 40% interval (0.3 - 0.7)
    q30_idx = QUANTILE_LEVELS.index(0.3)
    q70_idx = QUANTILE_LEVELS.index(0.7)
    ax.fill_between(future_dates, forecast[q30_idx, :], forecast[q70_idx, :],
                     alpha=0.4, color='red', label='40% Interval (Q30-Q70)')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cotton Futures Price (USD)', fontsize=12)
    ax.set_title('Cotton Futures 7-Day Zero-Shot Forecast (Chronos-2)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = 'experiments/cotton_forecast_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

def print_forecast_summary(forecast, df):
    """Print detailed forecast summary."""
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH)

    print("\n" + "="*80)
    print("FORECAST SUMMARY (7-Day Ahead)")
    print("="*80)

    median_idx = QUANTILE_LEVELS.index(0.5)
    q10_idx = QUANTILE_LEVELS.index(0.1)
    q90_idx = QUANTILE_LEVELS.index(0.9)

    for i, date in enumerate(future_dates):
        median = forecast[median_idx, i]
        q10 = forecast[q10_idx, i]
        q90 = forecast[q90_idx, i]

        print(f"{date.strftime('%Y-%m-%d')} | "
              f"Median: ${median:.2f} | "
              f"80% Interval: [${q10:.2f}, ${q90:.2f}] | "
              f"Range: ${q90 - q10:.2f}")

    print("="*80)

    # Last actual price
    last_price = df['Close'].iloc[-1]
    print(f"\nLast actual price ({df.index[-1].strftime('%Y-%m-%d')}): ${last_price:.2f}")
    print(f"7-day median forecast: ${forecast[median_idx, -1]:.2f}")
    print(f"Expected change: ${forecast[median_idx, -1] - last_price:.2f} "
          f"({(forecast[median_idx, -1] - last_price) / last_price * 100:.2f}%)")

def main():
    print("="*80)
    print("Cotton Futures Zero-Shot Forecasting with Chronos-2")
    print("="*80)

    # Load data
    df, close_prices = load_data(DATA_PATH)

    # Run inference
    forecast = run_inference(close_prices, CONTEXT_LENGTH)

    # Print summary
    print_forecast_summary(forecast, df)

    # Plot results
    plot_results(df, close_prices, forecast, CONTEXT_LENGTH)

    print("\nDone!")

if __name__ == "__main__":
    main()
