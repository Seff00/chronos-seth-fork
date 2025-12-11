# Chronos-2 Commodity Forecasting Experiments

This directory contains refactored experiments using the unified commodity dataset with configuration-driven design.

## Dataset

**Location**: `C:\Users\Seth\Desktop\AIAP\proj\commodity-forecasting\data\processed`

**Structure**:
- `train.csv`: 2,012 days (85.3%)
- `val.csv`: 251 days (10.6%)
- `test.csv`: 251 days (10.6%)

**Columns**:
- `Cotton_Futures_Close` (target)
- `Crude_Oil_Close` (covariate 1)
- `Copper_Futures_Close` (covariate 2)
- `SP500_Close` (covariate 3)
- `Dollar_Index_Close` (covariate 4)
- `Cotton_Futures_High` (covariate 5)
- `Cotton_Futures_Low` (covariate 6)
- `Cotton_Futures_Open` (covariate 7)
- `Cotton_Futures_Volume` (covariate 8)

---

## Experiment Overview

| # | Experiment | Model | Covariates | Fine-tuning | Config File |
|---|------------|-------|------------|-------------|-------------|
| 1 | Zero-shot Univariate | Chronos-2 | None | No | `zeroshot/univariate/config.yaml` |
| 2 | Zero-shot Multivariate | Chronos-2 | Oil, Copper | No | `zeroshot/multivariate/config.yaml` |
| 3 | LoRA 2-Covariate | Chronos-2 | Oil, Copper | LoRA | `finetune_lora/2covariates/config.yaml` |
| 4 | LoRA All-Covariate | Chronos-2 | All 8 | LoRA | `finetune_lora/allcovariates/config.yaml` |

**Total**: 4 experiments (2 zero-shot + 2 LoRA fine-tuned)

All experiments perform:
- **Forecast horizon**: 1 day ahead
- **Rolling evaluation**: 30 days
- **Metrics**: MAE, MSE, RMSE, MAPE + Direction Accuracy

---

## Zero-shot Experiments

### 1. Univariate Zero-shot

**Directory**: `./zeroshot/univariate/`

**Description**: Baseline experiment using only historical cotton prices (no covariates).

**Configuration** (`config.yaml`):
```yaml
data:
  target_column: "Cotton_Futures_Close"
  covariate_columns: []  # No covariates

forecast:
  prediction_days: 30
  context_length: 1024
```

**To run**:
```bash
cd experiments/zeroshot/univariate
python rolling_forecast.py
```

**Output**:
- `results/output.txt`: Detailed metrics
- `results/forecast_plot.png`: Time series visualization
- `results/metrics_plot.png`: Error analysis

---

### 2. Multivariate Zero-shot

**Directory**: `./zeroshot/multivariate/`

**Description**: Zero-shot with Crude Oil and Copper as covariates.

**Configuration** (`config.yaml`):
```yaml
data:
  target_column: "Cotton_Futures_Close"
  covariate_columns:
    - "Crude_Oil_Close"
    - "Copper_Futures_Close"

forecast:
  prediction_days: 30
  context_length: 1024
```

**To run**:
```bash
cd experiments/zeroshot/multivariate
python rolling_forecast.py
```

**Output**:
- `results/output.txt`: Detailed metrics
- `results/forecast_plot.png`: Time series visualization
- `results/metrics_plot.png`: Error analysis

---

## LoRA Fine-tuning Experiments

### 3. LoRA with 2 Covariates

**Directory**: `./finetune_lora/2covariates/`

**Description**: Fine-tune using only Crude Oil and Copper.

**Configuration** (`config.yaml`):
```yaml
data:
  target_column: "Cotton_Futures_Close"
  covariate_columns:
    - "Crude_Oil_Close"
    - "Copper_Futures_Close"

training:
  finetune_mode: "lora"
  learning_rate: 0.00001
  num_steps: 500
  batch_size: 256
  context_length: 365

forecast:
  prediction_days: 30
```

**To run**:
```bash
cd experiments/finetune_lora/2covariates
python train.py
```

**Output**:
- `results/checkpoint/`: Fine-tuned model
- `results/training_output.txt`: Training log
- `results/forecast_plot.png`: Predictions
- `results/metrics_plot.png`: Analysis

---

### 4. LoRA with All Covariates

**Directory**: `./finetune_lora/allcovariates/`

**Description**: Fine-tune using all 8 available covariates.

**Configuration** (`config.yaml`):
```yaml
data:
  target_column: "Cotton_Futures_Close"
  covariate_columns:
    - "Crude_Oil_Close"
    - "Copper_Futures_Close"
    - "SP500_Close"
    - "Dollar_Index_Close"
    - "Cotton_Futures_High"
    - "Cotton_Futures_Low"
    - "Cotton_Futures_Open"
    - "Cotton_Futures_Volume"

training:
  finetune_mode: "lora"
  learning_rate: 0.00001
  num_steps: 500
  batch_size: 256
  context_length: 365

forecast:
  prediction_days: 30
```

**To run**:
```bash
cd experiments/finetune_lora/allcovariates
python train.py
```

**Output**:
- `results/checkpoint/`: Fine-tuned model
- `results/training_output.txt`: Training log
- `results/forecast_plot.png`: Predictions
- `results/metrics_plot.png`: Analysis

---

## Running All LoRA Experiments

To run both LoRA experiments sequentially:

**Windows**:
```cmd
cd experiments\finetune_lora
run_lora_experiments.bat
```

**Linux/Mac**:
```bash
cd experiments/finetune_lora
chmod +x run_lora_experiments.sh
./run_lora_experiments.sh
```

---

## Configuration Files

All experiments are driven by YAML configuration files located in each experiment directory:

- `zeroshot/univariate/config.yaml`
- `zeroshot/multivariate/config.yaml`
- `finetune_lora/2covariates/config.yaml`
- `finetune_lora/allcovariates/config.yaml`

### Configuration Structure

```yaml
experiment:
  name: "experiment_name"
  description: "Description"

data:
  unified_data_path: "path/to/data"
  target_column: "Cotton_Futures_Close"
  covariate_columns: [...]

model:
  name: "amazon/chronos-2"
  device_map: "auto"
  torch_dtype: "bfloat16"

# For zero-shot experiments
forecast:
  prediction_days: 30
  prediction_length: 1
  context_length: 1024
  quantile_levels: [0.1, 0.2, ..., 0.9]

# For LoRA experiments (additional)
training:
  finetune_mode: "lora"
  learning_rate: 0.00001
  num_steps: 500
  batch_size: 256
  context_length: 365
  prediction_length: 1
  validation_inputs: null

output:
  output_dir: "./results"
  checkpoint_dir: "./results/checkpoint"  # LoRA only
  save_plots: true
  save_metrics: true
```

To modify an experiment, simply edit its `config.yaml` file.

---

## Directory Structure

```
experiments/
├── README.md
├── zeroshot/
│   ├── univariate/
│   │   ├── config.yaml
│   │   ├── rolling_forecast.py
│   │   └── results/
│   └── multivariate/
│       ├── config.yaml
│       ├── rolling_forecast.py
│       └── results/
└── finetune_lora/
    ├── run_lora_experiments.sh
    ├── run_lora_experiments.bat
    ├── 2covariates/
    │   ├── config.yaml
    │   ├── train.py
    │   └── results/
    │       └── checkpoint/
    └── allcovariates/
        ├── config.yaml
        ├── train.py
        └── results/
            └── checkpoint/
```

---

## Evaluation Metrics

All experiments report:

**Regression Metrics**:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

**Classification Metrics** (Direction Prediction):
- Accuracy (UP vs DOWN)
- Precision
- Recall
- F1-Score

**Visualizations**:
- Time series plot with historical context and predictions
- Error analysis by day
- Direction accuracy by day
- Color-coded predictions (Green=UP, Red=DOWN)

---

## Requirements

- Python 3.8+
- PyYAML
- chronos-forecasting
- torch
- pandas
- numpy
- matplotlib
- scikit-learn

Install dependencies:
```bash
pip install pyyaml chronos-forecasting torch pandas numpy matplotlib scikit-learn
```

---

## Notes

- All experiments use rolling 1-day ahead predictions for 30 days
- Test set is the last 30 days of the unified dataset
- LoRA fine-tuning uses 365-day context window (1 year)
- Zero-shot uses 1024-day context window (pretrained default)
- Results include both numerical metrics and visualizations
- Config files make it easy to modify hyperparameters without touching code

---

**Last Updated**: December 2025
