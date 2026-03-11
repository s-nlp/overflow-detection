# Hiddens comparison

Notebooks for **hidden-state saturation analysis**: analyzing and comparing neural activation patterns of xRAG tokens with baseline token types (context, non-xRAG) across model layers. This analysis reveals how xRAG tokens differ in their representational properties and saturation characteristics.

---

## Overview

This module investigates whether xRAG tokens exhibit different activation patterns compared to regular context tokens. The analysis focuses on three key saturation metrics computed per layer:
- **Hoyer sparsity index** — Measures activation sparsity (ratio of L1/L2 norms)
- **Spectral entropy** — Quantifies diversity of singular values in activation space
- **Excess kurtosis** — Captures tail heaviness and outliers in activation distributions

The pipeline compares xRAG token activations against multiple baselines (full context, minimal context, non-xRAG tokens) to identify distinctive patterns.

---

## Contents

### `getting_context_features.ipynb`

Extracts hidden-state saturation metrics for different token types by running the model with various context configurations.

**Purpose:**
- Isolate context-specific effects by comparing full context vs. minimal "No" context
- Compute per-layer saturation metrics for all token positions
- Generate baseline comparisons: first context token, mean context tokens, mean non-xRAG tokens

**Configuration (top cells):**
```python
INPUT_SAMPLES_PATH = "path/to/samples.jsonl"  # Input samples with xRAG predictions
OUTPUT_DIR = "path/to/output/"                # Where to save extracted features
XRAG_DIR = "xRAG/"                            # xRAG model implementation directory
CUDA_DEVICE = "cuda:0"                        # GPU device
```

**Process:**
1. Load xRAG model and input samples
2. For each sample, run model with:
   - Full context (original)
   - Minimal context (single "No" token)
3. Extract hidden states at all layers
4. Compute per-layer metrics: Hoyer sparsity, spectral entropy, excess kurtosis
5. Calculate statistics (mean, std) across layers

**Output:**
JSONL file in `OUTPUT_DIR` with structure:
```json
{
  "id": "sample_identifier",
  "context_features": {
    "hoyer_mean": [...],           // Per-layer means
    "hoyer_std": [...],            // Per-layer stds
    "spectral_entropy_mean": [...],
    "spectral_entropy_std": [...],
    "excess_kurtosis_mean": [...],
    "excess_kurtosis_std": [...]
  },
  "first_context_token_features": {...},
  "mean_context_tokens_features": {...},
  "mean_non_xrag_tokens_features": {...},
  "single_no_context_features": {...}
}
```

**Metrics explained:**
- **Hoyer index**: H = (√n - ||x||₁/||x||₂) / (√n - 1), range [0,1], higher = sparser
- **Spectral entropy**: H = -Σ(σᵢ/Σσ)·log(σᵢ/Σσ), computed on singular values of hidden states
- **Excess kurtosis**: Fourth moment statistic, positive values indicate heavy tails

### `adding_context_features.ipynb`

Merges context features from the previous step into the main features dataset.

**Purpose:**
- Combine context saturation metrics with existing xRAG features
- Enable unified analysis of all feature types in downstream tasks
- Align datasets by sample ID for proper matching

**Configuration (top cells):**
```python
CONTEXT_FEATURES_PATH = "path/to/context_features.jsonl"  # Output from step 1
MAIN_FEATURES_PATH = "path/to/main_features.jsonl"        # Base features dataset
OUTPUT_MERGED_PATH = "path/to/merged_features.jsonl"      # Final output
```

**Process:**
1. Load context features JSONL
2. Load main features JSONL
3. Merge by matching `id` field
4. Validate all matches succeeded
5. Save merged dataset

**Output:**
Merged JSONL file combining:
- Original xRAG features (embeddings, attention patterns, etc.)
- Context saturation metrics (all token type comparisons)
- Metadata (labels, predictions, metrics)

**Token type comparisons available:**
- `mean_non_xrag_tokens` — Average over all non-xRAG tokens in sequence
- `mean_context_tokens` — Average over context tokens only
- `first_context_token` — First token in context (position-sensitive)
- `single_no_context` — Minimal context baseline (single "No" token)

### `visualize_xrag_vs_others.ipynb`

Comprehensive visualization comparing xRAG tokens with baseline token types across datasets and metrics.

**Purpose:**
- Visualize distribution differences between xRAG and baseline tokens
- Generate figures (histograms, boxplots)
- Compute statistical significance tests
- Create comparison tables summarizing key findings

**Configuration (top cells):**
```python
DATA_PATH_TEMPLATE = "path/to/{dataset}_merged_features.jsonl"  # Template with {dataset}
FIGURES_OUTPUT_DIR = "figures/"                                  # Output directory
```

**Supported datasets:**
- `squad` — SQuAD (Stanford Question Answering Dataset)
- `triviaqa` — TriviaQA
- `hotpotqa` — HotpotQA (multi-hop reasoning)

**Process:**
1. Load merged features for all datasets
2. For each dataset and metric combination:
   - Extract xRAG token features
   - Extract baseline token features (context, non-xRAG)
   - Compute distributions and statistics
   - Generate comparative histograms
3. Create summary tables with mean/std comparisons
4. Statistical tests (t-test, KS-test) for significance

**Output:**

**Figures** (saved to `FIGURES_OUTPUT_DIR`):
- `{dataset}_{metric}_comparison.png` — Histogram overlays for each metric
- `{dataset}_all_metrics_boxplot.png` — Multi-panel boxplot comparisons
- `combined_heatmap.png` — Cross-dataset metric comparison heatmap

**In-notebook tables:**
- Metric summary table (mean ± std for each token type)
- Statistical significance table (p-values)
- Layer-wise progression tables (how metrics change across layers)

**Visualization types:**
- **Histograms**: Overlaid distributions of xRAG vs. baseline tokens
- **Boxplots**: Quartile comparisons across token types and datasets
- **Heatmaps**: Metric values aggregated by dataset and layer
- **Line plots**: Layer-wise metric trajectories
