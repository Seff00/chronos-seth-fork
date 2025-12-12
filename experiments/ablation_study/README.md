# Ablation Study: All Covariates Model

This folder contains scripts for ablation studies on the fine-tuned all covariates model.

## Purpose

Systematically remove covariates one at a time to measure their individual contribution to model performance.

## Key Differences from Evaluation Script

| Feature | finetune_lora/evaluate.py | ablation_study/forecast_allcovariates.py |
|---------|---------------------------|------------------------------------------|
| **Purpose** | Standard evaluation | Ablation study baseline |
| **Method** | Same (rolling 1-day) | Same (rolling 1-day) |
| **Calendar Features** | Not included | Not included |
| **Future Covariates** | Not used | Not used |
| **Test Period** | Last 30 days | Last 30 days |
| **Output** | Direction + price metrics | Direction + price metrics |

Both scripts are essentially identical - this serves as the **baseline** for comparing against ablation experiments (removing covariates, etc.).

## Covariates Used

**Past Covariates** (8 commodity features):
- Crude_Oil_Close
- Copper_Futures_Close
- SP500_Close
- Dollar_Index_Close
- Cotton_Futures_High
- Cotton_Futures_Low
- Cotton_Futures_Open
- Cotton_Futures_Volume

Total: 8 covariates

## Configuration

- **Model**: Fine-tuned LoRA checkpoint
- **Context Length**: 365 days (1 year)
- **Prediction Method**: Rolling 1-day ahead
- **Test Days**: 30 days
- **Quantiles**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

## Scripts

### 1. `forecast_allcovariates.py`
Baseline evaluation with all 8 covariates. Same as standard evaluation script.

**Usage:**
```bash
python experiments\ablation_study\forecast_allcovariates.py
```

### 2. `run_ablation_loop.py` ⭐
**Main ablation study script** - systematically excludes one covariate at a time.

**Usage:**
```bash
python experiments\ablation_study\run_ablation_loop.py
```

**What it does:**
- Runs baseline with all 8 covariates
- Runs 8 experiments, each excluding one covariate
- Compares performance to identify important features
- Total: 9 experiments (1 baseline + 8 ablations)

## Output

### Baseline Script (`forecast_allcovariates.py`)
Results saved to `experiments/ablation_study/results/`:
- `forecast_allcovariates_plot.png` - Forecast visualization with direction colors
- `forecast_allcovariates_metrics.png` - Regression errors and direction accuracy by day
- `forecast_allcovariates_output.txt` - Detailed metrics and predictions

### Ablation Loop (`run_ablation_loop.py`)
Results saved to `experiments/ablation_study/results/`:
- `ablation_results.json` - Machine-readable results for all experiments
- `ablation_report.txt` - Human-readable comparison report
- `ablation_results.png` - Bar chart showing impact of removing each covariate

## Metrics Reported

**Regression Metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

**Classification Metrics:**
- Accuracy (direction prediction)
- Precision
- Recall
- F1-Score

**Day-by-day comparison table** with both price and direction predictions

## Ablation Study Interpretation

The ablation loop measures **feature importance** by removal:

**Δ RMSE (Change in RMSE):**
- **Positive Δ**: Removing feature **worsens** performance → Feature is **important**
- **Negative Δ**: Removing feature **improves** performance → Feature is **noisy/redundant**
- **Zero Δ**: Removing feature has no impact → Feature is **irrelevant**

**Impact Levels:**
- **HIGH**: Δ RMSE > $0.05 (critical feature - major performance drop when removed)
- **MEDIUM**: Δ RMSE > $0.02 (useful feature - moderate contribution)
- **LOW**: Δ RMSE ≤ $0.02 (minimal contribution)

**Example:**
If removing "Crude_Oil_Close" increases RMSE from $0.57 to $0.65 (Δ = +$0.08), then Crude Oil is a **HIGH impact** feature that the model heavily relies on.

## Notes

- No calendar features or future covariates used - pure past covariate forecasting
- Rolling method means each prediction uses actual historical values up to that point
- All experiments use the same fine-tuned model - only input covariates change
- Runtime: ~40 minutes for all 9 experiments (depends on GPU)
