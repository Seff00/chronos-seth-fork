# Chronos-2 Commodity Forecasting Experiments

This directory contains refactored experiments using the unified commodity dataset with configuration-driven design.

## Dataset

**Location**: `C:\Users\Seth\Desktop\AIAP\proj\commodity-forecasting\data\processed`

**Structure**:
- `train.csv`: 2,012 days (85.3%)
- `val.csv`: 251 days (10.6%) - **Note: Combined with train for training, not used for validation**
- `test.csv`: 251 days (10.6%)

**Training approach**: Train and val sets are combined for training. The last 30 days are used as test set for rolling evaluation.

**Columns**:
- `Cotton_Futures_Close` (target)
- `Crude_Oil_Close`, `Copper_Futures_Close` (covariates)
- `SP500_Close`, `Dollar_Index_Close` (covariates)
- `Cotton_Futures_High`, `Cotton_Futures_Low`, `Cotton_Futures_Open`, `Cotton_Futures_Volume` (covariates)

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

## Quick Start

### Run All Zero-shot Experiments
```bash
# Can be run from root directory
chmod +x experiments/zeroshot/run_zeroshot_experiments.sh
./experiments/zeroshot/run_zeroshot_experiments.sh
```

### Run All LoRA Experiments
```bash
# Can be run from root directory
chmod +x experiments/finetune_lora/run_lora_experiments.sh
./experiments/finetune_lora/run_lora_experiments.sh
```

### Run Individual Experiments

**Zero-shot**:
```bash
cd experiments/zeroshot
python rolling_forecast.py univariate/config.yaml
python rolling_forecast.py multivariate/config.yaml
```

**LoRA** (train then evaluate):
```bash
cd experiments/finetune_lora

# Experiment 1: Train and evaluate 2-covariate model
python train.py 2covariates/config.yaml
python evaluate.py 2covariates/config.yaml

# Experiment 2: Train and evaluate all-covariate model
python train.py allcovariates/config.yaml
python evaluate.py allcovariates/config.yaml
```

---

## Zero-shot Experiments

### 1. Univariate Zero-shot

**Directory**: `./zeroshot/univariate/`

**Description**: Baseline experiment using only historical cotton prices (no covariates).

**Configuration** (`univariate/config.yaml`):
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
cd experiments/zeroshot
python rolling_forecast.py univariate/config.yaml
```

**Output** (saved to `univariate/results/`):
- `output.txt`: Detailed metrics
- `forecast_plot.png`: Time series visualization
- `metrics_plot.png`: Error analysis

---

### 2. Multivariate Zero-shot

**Directory**: `./zeroshot/multivariate/`

**Description**: Zero-shot with Crude Oil and Copper as covariates.

**Configuration** (`multivariate/config.yaml`):
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
cd experiments/zeroshot
python rolling_forecast.py multivariate/config.yaml
```

**Output** (saved to `multivariate/results/`):
- `output.txt`: Detailed metrics
- `forecast_plot.png`: Time series visualization
- `metrics_plot.png`: Error analysis

---

## LoRA Fine-tuning Experiments

### 3. LoRA with 2 Covariates

**Directory**: `./finetune_lora/2covariates/`

**Description**: Fine-tune using only Crude Oil and Copper.

**Configuration** (`2covariates/config.yaml`):
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
  validation_inputs: null  # No validation during training

forecast:
  prediction_days: 30
```

**To run**:
```bash
cd experiments/finetune_lora

# Step 1: Train the model
python train.py 2covariates/config.yaml

# Step 2: Evaluate the trained model
python evaluate.py 2covariates/config.yaml
```

**Output** (saved to `2covariates/results/`):
- `checkpoint/`: Fine-tuned model (from training)
- `training_log.txt`: Training log
- `evaluation_output.txt`: Evaluation log
- `forecast_plot.png`: Predictions (from evaluation)
- `metrics_plot.png`: Analysis (from evaluation)

---

### 4. LoRA with All Covariates

**Directory**: `./finetune_lora/allcovariates/`

**Description**: Fine-tune using all 8 available covariates.

**Configuration** (`allcovariates/config.yaml`):
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
  validation_inputs: null  # No validation during training

forecast:
  prediction_days: 30
```

**To run**:
```bash
cd experiments/finetune_lora

# Step 1: Train the model
python train.py allcovariates/config.yaml

# Step 2: Evaluate the trained model
python evaluate.py allcovariates/config.yaml
```

**Output** (saved to `allcovariates/results/`):
- `checkpoint/`: Fine-tuned model (from training)
- `training_log.txt`: Training log
- `evaluation_output.txt`: Evaluation log
- `forecast_plot.png`: Predictions (from evaluation)
- `metrics_plot.png`: Analysis (from evaluation)

---

## Directory Structure

```
experiments/
├── README.md
│
├── zeroshot/
│   ├── rolling_forecast.py          # Single script for all zero-shot experiments
│   ├── run_zeroshot_experiments.sh  # Run all zero-shot experiments
│   │
│   ├── univariate/
│   │   ├── config.yaml
│   │   └── results/                  # Results saved here
│   │
│   └── multivariate/
│       ├── config.yaml
│       └── results/                  # Results saved here
│
└── finetune_lora/
    ├── train.py                      # Training script for all LoRA experiments
    ├── evaluate.py                   # Evaluation script for all LoRA experiments
    ├── run_lora_experiments.sh       # Run all LoRA experiments
    │
    ├── 2covariates/
    │   ├── config.yaml
    │   └── results/
    │       ├── checkpoint/            # Trained model saved here
    │       ├── training_log.txt       # Training output
    │       ├── evaluation_output.txt  # Evaluation output
    │       ├── forecast_plot.png      # Predictions visualization
    │       └── metrics_plot.png       # Metrics visualization
    │
    └── allcovariates/
        ├── config.yaml
        └── results/
            ├── checkpoint/            # Trained model saved here
            ├── training_log.txt       # Training output
            ├── evaluation_output.txt  # Evaluation output
            ├── forecast_plot.png      # Predictions visualization
            └── metrics_plot.png       # Metrics visualization
```

**Key Design**:
- Single script per experiment type (`rolling_forecast.py` for zero-shot)
- Separate train and evaluate scripts for LoRA experiments
- Config files passed as command-line arguments
- Results saved in respective subdirectories

---

## Configuration Files

All experiments are driven by YAML configuration files:

### Configuration Structure

```yaml
experiment:
  name: "experiment_name"
  description: "Description"

data:
  unified_data_path: "path/to/data"
  target_column: "Cotton_Futures_Close"
  covariate_columns: [...]  # Empty list for univariate

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
  validation_inputs: null  # No validation during training

output:
  output_dir: "subfolder/results"
  checkpoint_dir: "subfolder/results/checkpoint"  # LoRA only
  save_plots: true
  save_metrics: true
```

**To modify an experiment**, edit its `config.yaml` file. No code changes needed!

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

## Data Usage Notes

**Important**: The validation set (`val.csv`) is **not used for validation** during training. Instead:
- Train and val sets are combined and used for training
- The last 30 days of the combined data are used as the test set for rolling evaluation
- `validation_inputs: null` in LoRA config means no early stopping

This approach maximizes training data while maintaining a proper holdout test set.

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
- Config files make it easy to modify hyperparameters
- Zero-shot experiments use single script with config arguments
- LoRA experiments split into separate train.py and evaluate.py scripts
- Training must complete successfully before evaluation can run

---

**Last Updated**: December 2025
