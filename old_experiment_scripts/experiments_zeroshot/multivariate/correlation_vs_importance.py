"""
Investigate why correlation doesn't match feature importance.
Analyzes contemporaneous, lagged, and predictive relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
COTTON_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Cotton_Futures.csv"
CRUDE_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Crude_Oil.csv"
COPPER_PATH = r"C:\Users\Seth\Desktop\AIAP\proj\myaiapprojrepo\data\raw\Copper_Futures.csv"

def load_csv_data(filepath, asset_name):
    """Load and preprocess commodity CSV data."""
    df = pd.read_csv(filepath, header=0, skiprows=[1, 2], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df = df.sort_index()
    prices = df['Close'].rename(asset_name)
    return prices

def align_data(cotton, crude, copper):
    """Align all time series to common dates."""
    combined = pd.DataFrame({'cotton': cotton, 'crude': crude, 'copper': copper})
    combined = combined.fillna(method='ffill').dropna()
    return combined

def calculate_lagged_correlations(data, target_col, feature_col, max_lag=30):
    """Calculate correlation at different lags."""
    lags = range(0, max_lag + 1)
    correlations = []

    for lag in lags:
        if lag == 0:
            # Contemporaneous correlation
            corr = data[target_col].corr(data[feature_col])
        else:
            # Lagged correlation: feature[t] vs target[t+lag]
            # How well does feature TODAY predict target LAG days from now
            target_future = data[target_col].shift(-lag)
            corr = data[feature_col].corr(target_future)

        correlations.append(corr)

    return list(lags), correlations

def calculate_returns_correlation(data, target_col, feature_col):
    """Calculate correlation of returns (changes) instead of levels."""
    target_returns = data[target_col].pct_change()
    feature_returns = data[feature_col].pct_change()

    return target_returns.corr(feature_returns)

def main():
    print("="*80)
    print("Correlation vs Feature Importance Analysis")
    print("="*80)

    # Load data
    print("\nLoading datasets...")
    cotton = load_csv_data(COTTON_PATH, "cotton")
    crude = load_csv_data(CRUDE_PATH, "crude")
    copper = load_csv_data(COPPER_PATH, "copper")
    data = align_data(cotton, crude, copper)

    print(f"Data range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    print(f"Total observations: {len(data)}")

    # 1. Contemporaneous Correlation (what you saw in EDA)
    print("\n" + "="*80)
    print("1. CONTEMPORANEOUS CORRELATION (same time t)")
    print("="*80)

    corr_matrix = data.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    cotton_corr = corr_matrix['cotton'].drop('cotton').sort_values(ascending=False)
    print("\nCotton correlations (sorted):")
    for asset, corr in cotton_corr.items():
        print(f"  {asset:10} : {corr:.4f}")

    # 2. Lagged Correlation (predictive power)
    print("\n" + "="*80)
    print("2. LAGGED CORRELATION (feature[t] vs cotton[t+lag])")
    print("="*80)
    print("This shows: How well does feature TODAY predict cotton FUTURE?")

    max_lag = 30  # Look up to 30 days ahead

    crude_lags, crude_corr = calculate_lagged_correlations(data, 'cotton', 'crude', max_lag)
    copper_lags, copper_corr = calculate_lagged_correlations(data, 'cotton', 'copper', max_lag)

    print(f"\nLagged correlation analysis (0-{max_lag} days ahead):")
    print(f"\nCrude Oil:")
    print(f"  Lag 0 (contemporaneous): {crude_corr[0]:.4f}")
    print(f"  Lag 7 (predict 7 days):  {crude_corr[7]:.4f}")
    print(f"  Best lag: {crude_lags[np.argmax(np.abs(crude_corr))]} (corr={max(crude_corr, key=abs):.4f})")

    print(f"\nCopper:")
    print(f"  Lag 0 (contemporaneous): {copper_corr[0]:.4f}")
    print(f"  Lag 7 (predict 7 days):  {copper_corr[7]:.4f}")
    print(f"  Best lag: {copper_lags[np.argmax(np.abs(copper_corr))]} (corr={max(copper_corr, key=abs):.4f})")

    # 3. Returns Correlation (changes vs changes)
    print("\n" + "="*80)
    print("3. RETURNS CORRELATION (% changes)")
    print("="*80)
    print("This shows: Do price CHANGES move together?")

    crude_ret_corr = calculate_returns_correlation(data, 'cotton', 'crude')
    copper_ret_corr = calculate_returns_correlation(data, 'cotton', 'copper')

    print(f"\nCrude Oil returns correlation: {crude_ret_corr:.4f}")
    print(f"Copper returns correlation:    {copper_ret_corr:.4f}")

    # 4. Rolling Correlation (time-varying)
    print("\n" + "="*80)
    print("4. ROLLING CORRELATION (90-day window)")
    print("="*80)

    window = 90
    rolling_crude = data['cotton'].rolling(window).corr(data['crude'])
    rolling_copper = data['cotton'].rolling(window).corr(data['copper'])

    print(f"\nRecent 90-day correlation:")
    print(f"  Crude Oil: {rolling_crude.iloc[-1]:.4f}")
    print(f"  Copper:    {rolling_copper.iloc[-1]:.4f}")

    print(f"\nAverage correlation (last year):")
    print(f"  Crude Oil: {rolling_crude.iloc[-252:].mean():.4f}")
    print(f"  Copper:    {rolling_copper.iloc[-252:].mean():.4f}")

    # 5. Visualization
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Lagged correlation
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(crude_lags, crude_corr, 'b-', linewidth=2, label='Crude Oil', marker='o')
    ax1.plot(copper_lags, copper_corr, 'r-', linewidth=2, label='Copper', marker='s')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(x=7, color='green', linestyle='--', alpha=0.3, label='7-day horizon')
    ax1.set_xlabel('Lag (days ahead)')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Lagged Correlation: Feature[t] → Cotton[t+lag]', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Correlation heatmap
    ax2 = plt.subplot(3, 2, 2)
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title('Contemporaneous Correlation Matrix', fontweight='bold')

    # Plot 3: Rolling correlation over time
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(rolling_crude.index, rolling_crude, 'b-', linewidth=1.5, label='Crude Oil', alpha=0.8)
    ax3.plot(rolling_copper.index, rolling_copper, 'r-', linewidth=1.5, label='Copper', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Date')
    ax3.set_ylabel(f'{window}-day Rolling Correlation')
    ax3.set_title(f'Time-Varying Correlation with Cotton', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.xticks(rotation=45)

    # Plot 4: Price series (normalized)
    ax4 = plt.subplot(3, 2, 4)
    normalized = data / data.iloc[0] * 100
    ax4.plot(normalized.index, normalized['cotton'], 'g-', linewidth=2, label='Cotton', alpha=0.8)
    ax4.plot(normalized.index, normalized['crude'], 'b-', linewidth=1.5, label='Crude Oil', alpha=0.7)
    ax4.plot(normalized.index, normalized['copper'], 'r-', linewidth=1.5, label='Copper', alpha=0.7)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Normalized Price (Base=100)')
    ax4.set_title('Price Trends (Normalized)', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    plt.xticks(rotation=45)

    # Plot 5: Returns distribution
    ax5 = plt.subplot(3, 2, 5)
    cotton_returns = data['cotton'].pct_change().dropna()
    crude_returns = data['crude'].pct_change().dropna()
    copper_returns = data['copper'].pct_change().dropna()

    ax5.scatter(crude_returns, cotton_returns, alpha=0.3, s=10, label=f'Crude (corr={crude_ret_corr:.3f})')
    ax5.scatter(copper_returns, cotton_returns, alpha=0.3, s=10, label=f'Copper (corr={copper_ret_corr:.3f})')
    ax5.set_xlabel('Feature Daily Return (%)')
    ax5.set_ylabel('Cotton Daily Return (%)')
    ax5.set_title('Returns Relationship', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Plot 6: Forecast horizon specific correlation
    ax6 = plt.subplot(3, 2, 6)
    horizons = [1, 3, 7, 14, 30]
    crude_horizon_corr = [crude_corr[h] if h <= max_lag else np.nan for h in horizons]
    copper_horizon_corr = [copper_corr[h] if h <= max_lag else np.nan for h in horizons]

    x = np.arange(len(horizons))
    width = 0.35
    ax6.bar(x - width/2, crude_horizon_corr, width, label='Crude Oil', color='blue', alpha=0.7)
    ax6.bar(x + width/2, copper_horizon_corr, width, label='Copper', color='red', alpha=0.7)
    ax6.set_xlabel('Forecast Horizon (days)')
    ax6.set_ylabel('Correlation')
    ax6.set_title('Predictive Correlation by Horizon', fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(horizons)
    ax6.legend()
    ax6.grid(alpha=0.3, axis='y')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    output_path = 'experiments_zeroshot/multivariate/correlation_vs_importance_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    plt.show()

    # Summary
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n1. Contemporaneous vs Predictive:")
    print(f"   - Crude Oil: Same-time corr = {crude_corr[0]:.3f}, 7-day predictive = {crude_corr[7]:.3f}")
    print(f"   - Copper:    Same-time corr = {copper_corr[0]:.3f}, 7-day predictive = {copper_corr[7]:.3f}")

    print("\n2. Why model might prefer Copper despite lower correlation:")
    if abs(copper_corr[7]) > abs(crude_corr[7]):
        print("   ✓ Copper has BETTER predictive correlation at 7-day horizon!")
    if copper_ret_corr > crude_ret_corr:
        print("   ✓ Copper's CHANGES are more correlated with cotton CHANGES")
    print("   ✓ Model sees temporal sequences, not just single values")
    print("   ✓ Model can capture non-linear relationships")
    print("   ✓ Copper volatility might signal cotton price movements")

    print("\n3. Remember:")
    print("   - Correlation measures LINEAR relationships at ONE point in time")
    print("   - Feature importance measures PREDICTIVE power across TIME")
    print("   - The model uses 1024 days of sequence data, not just correlations")
    print("="*80)

if __name__ == "__main__":
    main()
