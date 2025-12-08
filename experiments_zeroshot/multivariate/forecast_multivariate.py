"""
Multivariate forecasting on Cotton Futures using Chronos-2.
Uses Crude Oil and Copper Futures as covariates.
Predicts on the last 7 days (holdout) and evaluates against actual values.
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
MODEL_NAME = "amazon/chronos-2"  # 120M parameter model
PREDICTION_LENGTH = 7  # 7 days
CONTEXT_LENGTH = 1024  # Use last 1024 days for context (~2.8 years)
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

def align_data(cotton, crude, copper):
    """Align all time series to common dates and handle missing values."""
    # Combine all series
    combined = pd.DataFrame({
        'cotton': cotton,
        'crude': crude,
        'copper': copper
    })

    print(f"\nBefore alignment: {len(combined)} dates")
    print(f"Missing values - Cotton: {combined['cotton'].isna().sum()}, "
          f"Crude: {combined['crude'].isna().sum()}, "
          f"Copper: {combined['copper'].isna().sum()}")

    # Forward fill missing values (commodities markets have different trading hours/holidays)
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

def run_inference(combined_data, test_start_idx):
    """Run multivariate forecasting using Chronos-2."""
    print(f"\nLoading Chronos-2 model: {MODEL_NAME}")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Split data for training
    train_data = combined_data.iloc[:test_start_idx]

    # Get context data (last CONTEXT_LENGTH points)
    context_data = train_data.tail(CONTEXT_LENGTH)

    # Prepare target (cotton futures)
    target = context_data['cotton'].values

    # Prepare past covariates (crude oil, copper, and calendar features)
    past_calendar, future_calendar = create_calendar_features(
        context_data.index,
        combined_data.index[test_start_idx:test_start_idx + PREDICTION_LENGTH]
    )

    # Build input dictionary (note: pipeline.predict expects a LIST of dicts)
    input_dict = {
        "target": target,  # (context_length,)
        "past_covariates": {
            "crude_oil": context_data['crude'].values,      # (context_length,)
            "copper": context_data['copper'].values,         # (context_length,)
            "month": past_calendar['month'].values,          # (context_length,)
            "day_of_week": past_calendar['day_of_week'].values,  # (context_length,)
            "quarter": past_calendar['quarter'].values,      # (context_length,)
        },
        "future_covariates": {
            "month": future_calendar['month'].values,        # (prediction_length,)
            "day_of_week": future_calendar['day_of_week'].values,  # (prediction_length,)
            "quarter": future_calendar['quarter'].values,    # (prediction_length,)
        }
    }

    print(f"\nInput structure:")
    print(f"  Target (Cotton):        shape {target.shape}")
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

def plot_results(combined_data, test_start_idx, forecast, actual_prices, test_dates):
    """Plot last 14 days of historical data + 7-day prediction vs actual."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get last 14 days of training data
    historical_data = combined_data.iloc[test_start_idx - 14:test_start_idx]
    historical_dates = historical_data.index
    historical_prices = historical_data['cotton'].values

    # Plot last 14 days of historical data
    ax.plot(historical_dates, historical_prices, label='Historical Data (Cotton)',
            color='blue', linewidth=2, marker='o', markersize=4)

    # Plot actual test data (ground truth)
    ax.plot(test_dates, actual_prices, label='Actual (Test Period)',
            color='orange', linewidth=2, marker='o', markersize=6)

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

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cotton Futures Price (USD)', fontsize=12)
    ax.set_title('Cotton Futures 7-Day Multivariate Forecast vs Actual\n' +
                 'Covariates: Crude Oil, Copper, Calendar Features',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = 'experiments_zeroshot/multivariate/forecast_multivariate_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

def main():
    print("="*80)
    print("Cotton Futures Multivariate Zero-Shot Forecasting with Chronos-2")
    print("Covariates: Crude Oil, Copper Futures, Calendar Features")
    print("Evaluation Mode: Predicting on Last 7 Days (Holdout)")
    print("="*80)

    # Load all datasets
    print("\nLoading datasets...")
    cotton = load_csv_data(COTTON_PATH, "Cotton Futures")
    crude = load_csv_data(CRUDE_PATH, "Crude Oil")
    copper = load_csv_data(COPPER_PATH, "Copper Futures")

    # Align data to common dates
    combined_data = align_data(cotton, crude, copper)

    # Split: use all except last 7 days for training
    test_start_idx = len(combined_data) - PREDICTION_LENGTH
    test_dates = combined_data.index[test_start_idx:]
    actual_prices = combined_data['cotton'].iloc[test_start_idx:].values

    print(f"\nTrain period: {combined_data.index[0].strftime('%Y-%m-%d')} to " +
          f"{combined_data.index[test_start_idx - 1].strftime('%Y-%m-%d')}")
    print(f"Test period:  {test_dates[0].strftime('%Y-%m-%d')} to " +
          f"{test_dates[-1].strftime('%Y-%m-%d')}")
    print(f"Train size: {test_start_idx} days")
    print(f"Test size:  {PREDICTION_LENGTH} days")

    # Run multivariate inference
    forecast = run_inference(combined_data, test_start_idx)

    # Calculate metrics
    metrics = calculate_metrics(forecast, actual_prices)

    # Print metrics
    print_metrics(metrics, actual_prices, test_dates)

    # Plot results
    plot_results(combined_data, test_start_idx, forecast, actual_prices, test_dates)

    print("\nDone!")
    print("\nNOTE: This forecast uses Crude Oil and Copper prices as covariates,")
    print("      plus calendar features (month, day of week, quarter).")

if __name__ == "__main__":
    main()
