# Features comparison

Notebooks for **context-feature extraction** and **overflow-prediction comparison** in the xRAG overflow-detection pipeline. Compares the predictive power of different feature types: saturation metrics, context features, attention patterns, and linear probes on hidden states.

---

## Overview

This module evaluates which features best predict xRAG token overflow by comparing:
- **Pre-compression features**: Context-level metrics (perplexity, length, gzip compression)
- **Saturation features**: Hidden-state saturation metrics from hiddens_comparison
- **Attention features**: Attention pattern analysis
- **Linear probes**: Direct classification on hidden states

The goal is to determine whether simple, interpretable features can match or exceed the performance of end-to-end learned classifiers.

---

## Contents

### `getting_additional_context_features.ipynb`

Extracts context-level features that capture text complexity and compressibility.

**Purpose:**
- Compute linguistic and statistical features at the context level
- Measure text complexity via perplexity and compression metrics
- Merge with existing features for comprehensive analysis

**Configuration (second cell):**
```python
CUDA_DEVICE = "cuda:0"                           # GPU device
XRAG_DIR = "xRAG/"                               # xRAG model directory
FEATURES_PATH = "path/to/base_features.jsonl"    # Input features
CONTEXT_PATH = "path/to/contexts.jsonl"          # Context texts
OUTPUT_PATH = "path/to/output_features.jsonl"    # Merged output
MODEL_PATH = "model_name_or_path"                # Language model for perplexity
MAX_LENGTH = 512                                 # Max sequence length
```

**Extracted features:**

1. **Perplexity** — Language model perplexity on context text
   - Measures how "surprising" the context is to the model
   - Lower perplexity = more predictable text
   - Computed using a pretrained language model

2. **Token length** — Number of tokens in context
   - Raw token count after tokenization
   - Captures context size

3. **Character length** — Number of characters in context
   - Raw character count
   - Alternative length measure

4. **Gzip bits-per-character** — Compression ratio
   - Measures text compressibility using gzip
   - Formula: `(len(gzip(text)) * 8) / len(text)`
   - Lower BPC = more compressible/repetitive text
   - Higher BPC = more random/complex text

**Process:**
1. Load base features and context texts
2. For each sample:
   - Compute perplexity using language model
   - Count tokens and characters
   - Compute gzip compression ratio
3. Merge new features with base features by ID
4. Save merged dataset

**Output:**
JSONL file with added fields:
```json
{
  "id": "sample_id",
  "context_perplexity": 45.2,
  "context_token_length": 234,
  "context_char_length": 1456,
  "context_gzip_bpc": 4.73,
  ... // existing features
}
```

**Feature interpretation:**
- **High perplexity** → Complex, unexpected text → More likely to cause overflow?
- **Long context** → More information to process → Higher overflow risk?
- **High gzip BPC** → Less redundant text → Harder to compress → More complex?

### `visualize_features_classification.ipynb`

Compares logistic regression performance across different feature types to determine which best predicts overflow.

**Purpose:**
- Train logistic regression classifiers on different feature combinations
- Compare with linear probes on hidden states (from probing_experiments)
- Generate ROC-AUC comparison tables across datasets
- Identify which feature types are most predictive

**Configuration (first cell):**
```python
PROBING_BASE_PATH = "path/to/probing_results/"                     # Probing experiment results
FEATURES_PATH_TEMPLATE = "path/to/{dataset}_features.jsonl"        # Features for each dataset
```

**Feature groups compared:**

1. **Pre-compression** (Context features)
   - Perplexity, token length, char length, gzip BPC
   - From `getting_additional_context_features.ipynb`

2. **Saturation features**
   - Hoyer index, spectral entropy, excess kurtosis
   - From `hiddens_comparison/getting_context_features.ipynb`

3. **Attention features**
   - Attention pattern statistics
   - Attention weight distributions

4. **Linear probes** (Baseline)
   - Direct classification on hidden states
   - Results from `probing_experiments/`
   - Different stages: preproj, postproj, mid, last

**Analysis approach:**
- **Logistic regression** with L2 regularization for feature-based methods
- **Stratified 5-fold cross-validation** for all experiments
- **ROC-AUC** as primary evaluation metric
- **Statistical significance testing** between methods
