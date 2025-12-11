"""
Data splitting utilities for time series training.

Splits data chronologically:
- Test set: Latest 1 year (hold-out)
- Training set: Everything before test set

Simple workflow:
1. Train on train (no validation) → model
2. Test on test → final evaluation
"""

import pandas as pd


def split_train_test(
    combined_data: pd.DataFrame,
    use_recent_years_only: bool = False,
    recent_years: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets chronologically.

    Parameters
    ----------
    combined_data : pd.DataFrame
        Aligned time series data with 'target' and covariate columns
    use_recent_years_only : bool
        If True, use only the most recent `recent_years` of data
    recent_years : int
        Number of recent years to use when use_recent_years_only=True

    Returns
    -------
    train_data : pd.DataFrame
        Training set (all data except latest year)
    test_data : pd.DataFrame
        Test set (latest 1 year, hold-out)

    Split Logic
    -----------
    - Test: Latest 365 days (1 year)
    - Train: All remaining data before test

    If use_recent_years_only=True with recent_years=5:
    - Uses only last 5 years of data total
    - Test: 365 days (year 5)
    - Train: Remaining ~4 years (years 1-4)
    """
    total_points = len(combined_data)

    # Filter to recent years if configured
    if use_recent_years_only:
        days_to_use = recent_years * 365
        if total_points > days_to_use:
            combined_data = combined_data.tail(days_to_use)
            print(f"\nFiltering to recent {recent_years} years: {len(combined_data)} days")
            total_points = len(combined_data)

    # Define split sizes
    # test_size = 365  # 1 year
    test_size = 7    # Last 7 days only

    # Check if we have enough data
    if total_points <= test_size:
        raise ValueError(
            f"Insufficient data for train/test split. "
            f"Need at least {test_size + 1} days, got {total_points} days. "
            f"Consider setting use_recent_years_only=False or reducing recent_years."
        )

    # Calculate split index (chronological)
    test_start_idx = total_points - test_size

    # Split the data
    train_data = combined_data.iloc[:test_start_idx]
    test_data = combined_data.iloc[test_start_idx:]

    # Print split summary
    print("\n" + "="*80)
    print("DATA SPLIT SUMMARY")
    print("="*80)
    print(f"Total data points:       {total_points} days ({total_points/365:.2f} years)")
    print()

    print(f"Training set:            {len(train_data)} days ({len(train_data)/365:.2f} years, {len(train_data)/total_points*100:.1f}%)")
    print(f"  From: {train_data.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:   {train_data.index[-1].strftime('%Y-%m-%d')}")
    print()

    print(f"Test set (HOLD-OUT):     {len(test_data)} days ({len(test_data)/365:.2f} years, {len(test_data)/total_points*100:.1f}%)")
    print(f"  From: {test_data.index[0].strftime('%Y-%m-%d')}")
    print(f"  To:   {test_data.index[-1].strftime('%Y-%m-%d')}")
    print()

    print("NOTE: Test set is held out and NOT used during training")
    print("      No validation set - training without early stopping")
    print("="*80)

    return train_data, test_data


def prepare_training_samples(
    data: pd.DataFrame,
    context_length: int,
    sample_name: str = "samples"
) -> list[dict]:
    """
    Create sliding window training samples from time series data.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with 'target' and covariate columns
    context_length : int
        Number of timesteps in each context window
    sample_name : str
        Name for logging (e.g., "training", "validation")

    Returns
    -------
    samples : list of dict
        List of samples in Chronos-2 format:
        [{'target': array, 'past_covariates': {name: array}}, ...]
    """
    samples = []

    # Get covariate column names (everything except 'target')
    covariate_columns = [col for col in data.columns if col != 'target']

    # Build sliding window samples
    for i in range(context_length, len(data)):
        context = data.iloc[i-context_length:i]

        # Build past covariates dict
        past_covariates = {}
        for col in covariate_columns:
            key = col.lower().replace(' ', '_')
            past_covariates[key] = context[col].values

        samples.append({
            'target': context['target'].values,
            'past_covariates': past_covariates
        })

    print(f"Generated {len(samples)} {sample_name} (sliding windows with context_length={context_length})")

    return samples
