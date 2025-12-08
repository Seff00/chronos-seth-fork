# Chronos Time Series Forecasting - Architecture Overview

## Model Families

**Chronos (Original)**: Language model with quantization (8M-710M params). Time series → discrete bins → T5 training.

**Chronos-Bolt**: Patch-based, 250x faster, 20x more memory-efficient (9M-205M params). Best for speed.

**Chronos-2** (Focus): 120M params, Oct 2025. Universal forecasting (univariate/multivariate/covariates). SOTA performance, 90%+ win rate vs Bolt.

---

## Chronos-2 Architecture

### Overview

**Pure Transformer encoder** (no LSTM/RNN/GRU). Attention-based temporal modeling with RoPE positional encoding. Treats time series like sequences, not traditional recurrent models.

**Key properties**:
- Output dimensionality matches input: `(n_variates, n_quantiles, prediction_length)`
- Distinguishes targets from covariates via **masking**, not separate architectures
- Encodes all series together, predicts all, trains only on unknown values

### High-Level Pipeline

```
Input Time Series
    ↓
Instance Normalization (scaling)
    ↓
Patching (chunk into patches)
    ↓
Time Encoding (sequential time indices)
    ↓
Patch Embedding (ResidualBlock)
    ↓
Dual-Attention Encoder (Time + Group attention)
    ↓
Output Patch Embedding
    ↓
Quantile Predictions
```

### Core Components

#### 1. Input Processing Pipeline
**Location**: `src/chronos/chronos2/model.py:373-423`

**Instance Normalization**:
- Normalizes each time series using mean/standard deviation
- Optional `arcsinh` transform for heavy-tailed distributions
- Applied in 32-bit precision for numerical stability
- Stores `(loc, scale)` for inverse transform later

**Patching**: Groups consecutive timesteps into chunks (like Vision Transformers). `patch_size=4` converts 2048 timesteps → 512 patches.

**Benefits**: 16x efficiency (O(n²/k²) complexity). Patch `[10,12,15,18]` becomes one token. Missing values: NaN→0, mask valid if ≥1 value observed.

**vs Chronos-1**: Quantization (discrete bins, lookup table) → Patching (continuous vectors, linear projection)

**Time Encoding**:
- Every observation gets a sequential time index
- Context encoding: `[-C, -(C-1), ..., -1] / time_encoding_scale`
- Future encoding: `[0, 1, ..., h-1] / time_encoding_scale`
- Default `time_encoding_scale = context_length`
- Enables model to understand temporal positions

**Patch Embedding**:
- ResidualBlock: `in_dim → h_dim → out_dim`
- Input: `[time_encoding, patch_values, patch_mask]` (3x patch_size)
- Hidden: `d_ff` dimensions
- Output: `d_model` dimensions
- Activation: ReLU (configurable via `feed_forward_proj`)

#### 2. Encoder Architecture
**Location**: `src/chronos/chronos2/model.py:89-187`

**Base Design**: T5-style encoder-only architecture

**Encoder Block Structure** (repeated `num_layers` times):
```
Input
  ↓
TimeSelfAttention (temporal relationships)
  ↓
GroupSelfAttention (cross-series mixing)
  ↓
FeedForward (MLP)
  ↓
Output
```

**TimeSelfAttention**:
- Standard multi-head self-attention
- Uses RoPE (Rotary Position Embeddings) for positional information
- Attention mask prevents attending to padded/missing values
- Shape: `(batch, seq_len, d_model)`

**GroupSelfAttention**:
- Cross-series attention mechanism
- `group_ids` tensor determines which series can share information
- Group-time mask: `group_mask ⊗ time_mask`
  - Only attends to tokens from same group AND valid time positions
- Enables:
  - Multivariate forecasting (same group)
  - Covariate-informed forecasting (same group, different roles)
  - Independent univariate forecasting (unique group IDs)

**Example Group Setup**:
```python
# Batch of 6 time series
group_ids = [0, 0, 1, 1, 1, 2]

# Group 0: 2 series (multivariate forecasting)
# Group 1: 3 series (2 covariates + 1 target to predict)
# Group 2: 1 series (univariate forecasting)
```

**Final Processing**:
- Layer normalization (T5-style: no bias, no mean subtraction)
- Dropout for regularization

#### 3. Attention Mechanism Details
**Location**: `src/chronos/chronos2/layers.py`

**Multi-Head Attention (MHA)**:
- Projection dimensions:
  - `d_model` → `n_heads * d_kv` (query, key, value)
  - `n_heads * d_kv` → `d_model` (output)
- **No scaling** in attention scores (differs from standard transformers)
- No bias terms in linear projections
- Supports two implementations:
  - `eager`: Manual matmul (for debugging/attention extraction)
  - `sdpa`: PyTorch Scaled Dot Product Attention (faster, default)

**Rotary Position Embeddings (RoPE)**:
- Applied to query and key tensors before attention
- Base frequency: `rope_theta = 10000.0`
- More effective than absolute positions for time series
- Enables better length extrapolation

**T5-Style LayerNorm**:
```python
# No mean subtraction, no bias
variance = hidden_states.pow(2).mean(-1, keepdim=True)
output = hidden_states * torch.rsqrt(variance + eps) * weight
```

**Feed-Forward Network**:
- Two linear layers: `d_model → d_ff → d_model`
- Activation: ReLU (default)
- Residual connection + dropout
- No bias terms

---

### Dual-Attention: Time vs Group

**Separate weights** (Q,K,V,O) for each attention type. No biases (T5-style).

**TimeSelfAttention**: Temporal sequence within one series. Uses RoPE for positional encoding (order matters).

**GroupSelfAttention**: Cross-series attention within group. No RoPE (variables unordered). Uses `group_ids` mask for membership.

**Two Masking Types**:
1. **Attention masking** (forward): Prevents padding/cross-group leakage
2. **Loss masking** (training): Only unknowns contribute to loss

---

### Factorized Attention Trade-off

**Limitation**: Single layer can't directly access cross-series temporal deps (A[t=1] ← B[t=0]).

**Solution**: 6 layers of alternating Time→Group attention. Info flows indirectly: B[t=1] encodes B[t=0], then GroupAttention shares to A[t=1]. Residual connections accumulate information.

**vs Full Spatiotemporal**: O(n²)+O(m²) vs O((n×m)²). Chronos-2 scales to 2048 context + many series. Full spatiotemporal would be 419M ops for context=2048, series=10.

**Trade**: Sacrificed direct access, gained 16x speed, scalability, modularity. Validated by SOTA results.

---

#### 4. Output & Forecasting

**Output**: `(n_variates, n_quantiles, prediction_length)` - matches input dimensionality. Only targets predicted, not covariates.

**Quantile Regression**: 9 quantiles `[0.1..0.9]` for probabilistic forecasts. Pinball loss: `2|target-pred|*((target≤pred)-quantile)`.

**Direct Multi-Step**: Generates patches in one pass (up to `max_output_patches`). Long horizons use autoregressive unrolling with multiple quantiles to prevent collapse.

**Output path**: Encoder hidden → ResidualBlock → quantiles → inverse normalization → original scale.

---

### Gradient Flow

**No information leakage**: Model sees `context` + `future_covariates`, NOT `future_target` labels during forward pass.

**Backward**: Gradients flow through all 6 layers (TimeSelfAttention → GroupSelfAttention → FeedForward).

**Key**: Covariates receive indirect gradients via GroupSelfAttention cross-series mixing, even though their predictions don't contribute to loss. Temperature hidden states get trained because demand predictions depend on them.

---

## Target vs Covariate Handling

**Unified architecture via masking**:
1. Encode all series together (targets + covariates)
2. Predict for ALL series
3. Compute loss only on unknowns: `loss_mask = future_target_mask * (1 - future_covariate_mask)`
4. Extract only target predictions at inference

**Example**: Forecast demand with temperature covariate. Model predicts both but only trains on demand. Temperature still learns via GroupSelfAttention gradients.

**Why elegant**: No separate pathways needed. Same model handles univariate, multivariate, covariates. Zero-shot flexible.

### Configuration Parameters

**Core Config** (`Chronos2CoreConfig`):
- `d_model = 512`: Hidden state dimension
- `d_kv = 64`: Key/value projection dimension per head
- `d_ff = 2048`: Feedforward intermediate dimension
- `num_layers = 6`: Number of encoder blocks
- `num_heads = 8`: Number of attention heads
- `dropout_rate = 0.1`: Dropout probability
- `layer_norm_epsilon = 1e-6`: LayerNorm stability
- `initializer_factor = 0.05`: Weight initialization scale
- `feed_forward_proj = "relu"`: Activation function
- `vocab_size = 2`: Special tokens ([PAD], [REG])
- `rope_theta = 10000.0`: RoPE base frequency
- `attn_implementation = "sdpa"`: Attention backend

**Forecasting Config** (`Chronos2ForecastingConfig`):
- `context_length = 2048`: Maximum historical window
- `input_patch_size`: Patch size for input
- `input_patch_stride`: Stride between patches
- `output_patch_size`: Patch size for output
- `quantiles`: List of quantile levels
- `use_reg_token = False`: Whether to use [REG] special token
- `use_arcsinh = False`: Apply arcsinh transform
- `max_output_patches = 1`: Max patches in single forward pass
- `time_encoding_scale`: Normalization for time indices

---

## Key Innovations vs Chronos-1

### 1. Universal Forecasting
- **Chronos-1**: Univariate only
- **Chronos-2**: Univariate + multivariate + covariate-informed

### 2. Group Attention Mechanism
- Novel cross-series attention via `group_ids`
- Enables flexible task configuration in single batch
- Zero-shot multivariate and covariate support

### 3. Patch-Based Architecture
- More efficient than token-based quantization
- Reduces sequence length while preserving information
- Direct multi-step forecasting (vs. autoregressive sampling)

### 4. Enhanced Input Processing
- Time encoding for temporal awareness
- Instance normalization with optional arcsinh
- Explicit handling of missing values via masks

### 5. Flexible Fine-Tuning
- Full fine-tuning support
- LoRA (Low-Rank Adaptation) support for efficient adaptation
- Can fine-tune on task-specific data with custom context/prediction lengths

---

## Inference Pipeline

**Location**: `src/chronos/chronos2/pipeline.py:442-635`

### Standard Prediction Flow

1. **Input Conversion**:
   - Convert various input formats to `Chronos2Dataset`
   - Supports: tensors, arrays, lists, dicts with covariates
   - Handles ragged sequences via padding

2. **Batch Processing**:
   - Configurable `batch_size` (default: 256)
   - Note: batch_size = number of time series (including targets + covariates)
   - Effective task count may be lower for multivariate inputs

3. **Prediction Modes**:
   - **Independent**: Each time series/task predicted separately (default)
   - **Cross-learning**: `cross_learning=True` enables information sharing across entire batch
     - Sets all `group_ids = 0`
     - Useful when series are related and have limited history
     - Optimal batch size ~100 (based on pretraining)

4. **Long-Horizon Strategy**:
   - If `prediction_length <= max_output_patches * output_patch_size`: Single forward pass
   - Otherwise: Autoregressive unrolling with quantile interpolation

5. **Output Format**:
   - List of tensors: `(n_variates, n_quantiles, prediction_length)`
   - One tensor per task in input

### Supported Input Formats

**Shape**: `(n_variates, history_length)` where n_variates = number of variables.

1. **Tensors**: `torch.randn(32, 3, 100)` - 32 tasks, 3 variates each, 100 timesteps
2. **Lists**: Mixed univariate/multivariate sequences
3. **Dicts with covariates**: `{"target": ..., "past_covariates": {...}, "future_covariates": {...}}`

### DataFrame Interface

**Location**: `pipeline.py:802-925`

```python
pipeline.predict_df(
    df=context_df,              # Historical data (long format)
    future_df=test_df,          # Future covariates (optional)
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",            # Can be list for multivariate
)
```

Returns DataFrame with predictions and quantiles per time series.

---

## Training & Fine-Tuning

**Location**: `pipeline.py:96-356`

### Fine-Tuning Modes

**1. Full Fine-Tuning** (`finetune_mode="full"`):
- Updates all model parameters
- Learning rate: `1e-6` (default)
- Best for sufficient data and compute

**2. LoRA Fine-Tuning** (`finetune_mode="lora"`):
- Low-rank adaptation of attention layers
- Default targets: Q, K, V, O projections + output embedding
- Learning rate: `1e-5` (recommended)
- Memory efficient, faster training

### Fine-Tuning Parameters

```python
finetuned_pipeline = pipeline.fit(
    inputs=train_data,
    prediction_length=24,
    validation_inputs=val_data,      # Optional
    finetune_mode="lora",             # or "full"
    lora_config=None,                 # or custom LoraConfig
    context_length=2048,              # Max context
    learning_rate=1e-5,
    num_steps=1000,
    batch_size=256,
    output_dir="./finetuned",
)
```

### Adaptive Context/Prediction Length
- Model's `context_length` updated if fine-tuned with longer context
- Model's `max_output_patches` updated if fine-tuned with longer horizon
- Config saved with updated values

---

## Advanced Features

### 1. Embeddings Extraction
**Location**: `pipeline.py:1075-1148`

```python
embeddings, loc_scales = pipeline.embed(
    inputs=time_series,
    batch_size=256,
    context_length=2048,
)
# Returns: (n_variates, num_patches + 2, d_model) per task
# +2 for [REG] token and masked output patch
```

### 2. FEV Benchmark Integration
**Location**: `pipeline.py:997-1073`

- Native support for `fev` (Forecasting Evaluation) library
- Handles fev.Task evaluation windows
- Optional fine-tuning on first window
- Returns predictions in fev-compatible format

### 3. Cross-Learning
- Enable with `cross_learning=True` in predict methods
- All inputs in batch share information (single group)
- Most effective when:
  - Individual series have limited context
  - Series are related/similar
  - Batch size ~100 (as in pretraining)

---

## Implementation Details

### Special Tokens
- `[PAD]` (id=0): Padding/missing values
- `[REG]` (id=1, optional): Registration token for aggregation

### Numerical Stability
- Instance normalization in FP32
- RoPE computations in FP32
- Mixed precision training supported (bf16/fp16)
- TF32 enabled on Ampere+ GPUs

### Efficiency Optimizations
- Patch-based processing reduces sequence length
- SDPA backend for faster attention
- Optional TF32 for matrix operations
- Fused AdamW optimizer when available

### Masking Strategy
- Time attention mask: Standard causal/padding mask
- Group-time mask: Outer product of group and time masks
- Future covariate mask: Indicates known vs. unknown values

---

## Code Structure

```
src/chronos/chronos2/
├── config.py           # Model configuration classes
├── model.py            # Core Chronos2Model implementation
├── layers.py           # Attention, FFN, RoPE layers
├── pipeline.py         # Inference and fine-tuning pipeline
├── dataset.py          # Data loading and preprocessing
├── trainer.py          # HuggingFace Trainer integration
└── __init__.py         # Package exports
```

### Key Classes

- `Chronos2Model`: Main model (extends `PreTrainedModel`)
- `Chronos2Pipeline`: User-facing API for inference/fine-tuning
- `Chronos2Encoder`: Stack of encoder blocks
- `Chronos2EncoderBlock`: Single encoder layer (Time + Group + FFN)
- `TimeSelfAttention`: Temporal attention with RoPE
- `GroupSelfAttention`: Cross-series attention
- `MHA`: Multi-head attention implementation
- `Chronos2Dataset`: Dataset handling various input formats

---

## Usage Recommendations

### When to Use Chronos-2
- ✅ Need covariates (past-only or future-known)
- ✅ Multivariate forecasting
- ✅ Best accuracy on diverse tasks
- ✅ Fine-tuning on custom data
- ✅ Moderate prediction lengths (<2048 steps)

### When to Use Chronos-Bolt
- ✅ Speed critical
- ✅ Resource constrained (memory/compute)
- ✅ Simple univariate tasks
- ✅ Very long prediction horizons

### When to Use Original Chronos
- ✅ Legacy compatibility
- ✅ Largest model size needed (710M)
- ✅ Research on language model approaches

---

## References

- Paper: "Chronos: Learning the Language of Time Series" (arXiv:2403.07815)
- Technical Report: "Chronos-2: From Univariate to Universal Forecasting" (arXiv:2510.15821)
- HuggingFace: https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444
- Repository: https://github.com/amazon-science/chronos-forecasting

---

**Last Updated**: December 2025
**Chronos-2**: Oct 2025, 120M params, 2048 context, Transformer encoder
