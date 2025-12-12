# Ablation Study: All Covariates Model

This folder contains scripts for ablation studies on the fine-tuned all covariates model.

## Purpose

Systematically remove covariates one at a time to measure their individual contribution to model performance.

## Key Differences from Evaluation Script

| Feature | finetune_lora/evaluate.py | ablation_study/run_ablation_loop.py |
|---------|---------------------------|-------------------------------------|
| **Purpose** | Standard evaluation | Feature importance analysis |
| **Method** | Rolling 1-day (all covariates) | Rolling 1-day (ablate 1 at a time) |
| **Experiments** | 1 (baseline) | 9 (baseline + 8 ablations) |
| **Output** | Performance metrics | Feature importance ranking |

## Covariates Tested

**Past Covariates** (8 commodity features):
- Crude_Oil_Close
- Copper_Futures_Close
- SP500_Close
- Dollar_Index_Close
- Cotton_Futures_High
- Cotton_Futures_Low
- Cotton_Futures_Open
- Cotton_Futures_Volume

## Configuration

- **Model**: Fine-tuned LoRA checkpoint
- **Context Length**: 365 days (1 year)
- **Prediction Method**: Rolling 1-day ahead
- **Test Days**: 30 days
- **Quantiles**: Automatically detected from model

## Scripts

### `run_ablation_loop.py` ⭐
**Main ablation study script** - systematically excludes one covariate at a time.

**Usage:**
```bash
cd C:\Users\Seth\Desktop\AIAP\proj\chronos-seth-fork
python experiments\finetune_lora\ablation_study\run_ablation_loop.py
```

**What it does:**
- Runs baseline with all 8 covariates
- Runs 8 experiments, each excluding one covariate
- Compares performance to identify important features
- Total: 9 experiments (1 baseline + 8 ablations)

## Output

Results saved to `experiments/finetune_lora/ablation_study/results/`:
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
- Plots are 16" wide to prevent label cramping
