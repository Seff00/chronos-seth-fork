"""
Zero-shot forecasting on Cotton Futures data using Chronos-2 with WEEKLY aggregation.
Aggregates daily data to weekly to capture full 11-year history in context window.
Forecasts 4 weeks ahead (1 month) and plots results with quantile uncertainty bands.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline

# Configuration
DATA_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Cotton_Futures.csv"
MODEL_NAME = "amazon/chronos-2-120m"
PREDICTION_LENGTH = 4  # 4 weeks ahead (1 month)
CONTEXT_LENGTH = 2048  # Use full model capacity
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_and_aggregate_data(filepath):
    """Load daily data and aggregate to weekly frequency."""
    # Read CSV, skip the Ticker row
    df = pd.read_csv(filepath, skiprows=[1])

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Loaded {len(df)} daily data points")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Set Date as index for resampling
    df.set_index('Date', inplace=True)

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

def run_inference(weekly_prices, context_length):
    """Run zero-shot forecasting using Chronos-2."""
    print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Use the last context_length weeks as input
    context = weekly_prices[-context_length:]
    context_tensor = torch.tensor(context).unsqueeze(0)  # Shape: (1, context_length)

    print(f"Running inference with context of {len(context)} weeks ({len(context)/52:.1f} years)...")
    print(f"Forecasting {PREDICTION_LENGTH} weeks ahead...")

    # Run prediction
    predictions = pipeline.predict(
        context_tensor,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
    )

    # Extract predictions: shape (1, num_quantiles, prediction_length)
    forecast = predictions[0].numpy()  # Shape: (num_quantiles, prediction_length)

    return forecast

def plot_results(weekly_df, weekly_prices, forecast, context_length):
    """Plot historical data, context window, and forecast with uncertainty bands."""
    fig, ax = plt.subplots(figsize=(15, 7))

    # Get the last portion of data for visualization (last ~2 years)
    viz_length = min(104, len(weekly_prices))  # 104 weeks = 2 years
    viz_dates = weekly_df['Date'].values[-viz_length:]
    viz_prices = weekly_prices[-viz_length:]

    # Context dates (last context_length weeks used)
    actual_context_length = min(context_length, len(weekly_prices))
    context_dates = weekly_df['Date'].values[-actual_context_length:]
    context_prices = weekly_prices[-actual_context_length:]

    # Generate future dates for forecast (weekly)
    last_date = pd.to_datetime(weekly_df['Date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                  periods=PREDICTION_LENGTH, freq='W-FRI')

    # Plot historical data (lighter)
    ax.plot(viz_dates, viz_prices, label='Historical Weekly Data',
            color='gray', alpha=0.5, linewidth=1, marker='o', markersize=2)

    # Plot context window (darker)
    ax.plot(context_dates, context_prices, label=f'Context Window ({actual_context_length} weeks)',
            color='blue', linewidth=2, marker='o', markersize=3)

    # Plot median forecast (0.5 quantile)
    median_idx = QUANTILE_LEVELS.index(0.5)
    ax.plot(future_dates, forecast[median_idx, :], label='Median Forecast',
            color='red', linewidth=2.5, marker='s', markersize=6)

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
    ax.set_title(f'Cotton Futures 4-Week Forecast - Weekly Aggregation (Chronos-2)\n'
                 f'Context: {actual_context_length} weeks ({actual_context_length/52:.1f} years)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = 'experiments/cotton_forecast_weekly_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

def print_forecast_summary(forecast, weekly_df):
    """Print detailed forecast summary."""
    last_date = pd.to_datetime(weekly_df['Date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                  periods=PREDICTION_LENGTH, freq='W-FRI')

    print("\n" + "="*80)
    print(f"FORECAST SUMMARY ({PREDICTION_LENGTH}-Week Ahead)")
    print("="*80)

    median_idx = QUANTILE_LEVELS.index(0.5)
    q10_idx = QUANTILE_LEVELS.index(0.1)
    q90_idx = QUANTILE_LEVELS.index(0.9)

    for i, date in enumerate(future_dates):
        median = forecast[median_idx, i]
        q10 = forecast[q10_idx, i]
        q90 = forecast[q90_idx, i]

        print(f"Week ending {date.strftime('%Y-%m-%d')} | "
              f"Median: ${median:.2f} | "
              f"80% Interval: [${q10:.2f}, ${q90:.2f}] | "
              f"Range: ${q90 - q10:.2f}")

    print("="*80)

    # Last actual price
    last_price = weekly_df['Close'].iloc[-1]
    print(f"\nLast actual weekly price (week ending {weekly_df['Date'].iloc[-1].strftime('%Y-%m-%d')}): ${last_price:.2f}")
    print(f"4-week median forecast: ${forecast[median_idx, -1]:.2f}")
    print(f"Expected change: ${forecast[median_idx, -1] - last_price:.2f} "
          f"({(forecast[median_idx, -1] - last_price) / last_price * 100:.2f}%)")

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

def main():
    print("="*80)
    print("Cotton Futures Zero-Shot Forecasting with Chronos-2 (WEEKLY AGGREGATION)")
    print("="*80)

    # Load and aggregate data
    weekly_df, weekly_prices = load_and_aggregate_data(DATA_PATH)

    # Print comparison
    print_comparison_summary(weekly_df, weekly_prices)

    # Run inference
    forecast = run_inference(weekly_prices, CONTEXT_LENGTH)

    # Print summary
    print_forecast_summary(forecast, weekly_df)

    # Plot results
    plot_results(weekly_df, weekly_prices, forecast, CONTEXT_LENGTH)

    print("\nDone!")
    print("\nNOTE: This weekly forecast predicts prices at week-end (Friday close).")
    print("      Each prediction represents the price ~7 days ahead from previous point.")

if __name__ == "__main__":
    main()
