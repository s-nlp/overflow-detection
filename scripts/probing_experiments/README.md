# Probing experiments

**Classifier probing** on xRAG (and optionally query) embeddings to predict token overflow. Trains multiple classifier types on fixed embedding combinations to evaluate overflow detection performance across different feature stages.

---

## Overview

This module evaluates how well different classifier architectures can predict xRAG token overflow using embeddings from various model stages. It supports two experimental setups:
- **With-query**: Combines xRAG embeddings with query embeddings
- **No-query**: Uses only xRAG embeddings (baseline)

The experiments test 4 classifier types with different embedding stage combinations (pre-projection, post-projection, middle layers, last layer).

---

## Contents

### `run_probing_experiments.py`

Main training script that runs probing experiments with stratified k-fold cross-validation.

**Key features:**
- Runs two experiment blocks by default (with-query and no-query)
- Tests 4 classifier architectures: LinearProbeTorch, MLPProbeTorch, MLPSCLProbeTorch, LinearProbe (sklearn)
- Two feature combination settings:
  - **Setting 1**: Pre/post-projection stages (`preproj`, `postproj`)
  - **Setting 2**: Middle/last layer stages (`mid`, `last`)
- Computes multiple metrics: AUC-ROC, accuracy, F1, average precision
- Saves detailed results including per-fold metrics and aggregated statistics

**Usage:**
```bash
python run_probing_experiments.py \
    --data_path /path/to/vectors_probing.pt \
    --output_dir ./results \
    --experiment_name probing \
    --cv_folds 5 \
    --device 0
```

**Options:**
- `--with_query_only` — Run only with-query experiments (Setting 1 & 2)
- `--no_query_only` — Run only no-query experiments
- `--cv_folds N` — Number of cross-validation folds (default: 5)
- `--device N` — CUDA device index (default: 0)

**Input format:**
PyTorch `.pt` file containing:
- `ids` — Sample identifiers
- `labels` — Binary overflow labels
- Embedding keys: `preproj`, `postproj`, `mid`, `last` (xRAG embeddings)
- Optional query keys: `preproj_q`, `postproj_q`, `mid_q`, `last_q`

**Output:**
JSON files in `output_dir`:
- `probing_results_setting1_*.json` — Setting 1 (preproj/postproj) with query
- `probing_results_setting2_*.json` — Setting 2 (mid/last) with query  
- `probing_results_xrag_*_no_query*.json` — No-query baseline experiments

Each JSON contains:
- Aggregated metrics (mean ± std across folds)
- Per-fold detailed results
- Feature combination and classifier configuration

### `visualize_classifiers_performance.py`

Visualization script that generates comparison figures across datasets and feature stages.

**Features:**
- Compares LLM-independent (preproj/postproj) vs LLM-dependent (mid/last) features
- Side-by-side comparison across multiple datasets
- Separate plots for different classifier architectures
- Saves both PNG and PDF formats

**Usage:**
```bash
python visualize_classifiers_performance.py \
    --base_path /path/to/results \
    --datasets mistral_squad mistral_trivia mistral_hotpot \
    --output_dir ./figures \
    --experiment_name probing
```

**Output:**
- `classifier_performance_comparison.png` — Raster figure
- `classifier_performance_comparison.pdf` — Vector figure

### Supporting modules

| Module | Purpose |
|--------|---------|
| **models.py** | Classifier implementations: `LinearProbe` (sklearn), `LinearProbeTorch`, `MLPProbeTorch` (single hidden layer), `MLPSCLProbeTorch` (MLP with supervised contrastive learning) |
| **data_loader.py** | Functions to load and validate `.pt` probing data format |
| **utils.py** | Utility functions including `set_seed()` for reproducibility |

---

## Classifier architectures

### LinearProbeTorch
- Single linear layer with L1/L2 regularization
- Configurable normalization, early stopping
- Default: λ₂=500, λ₁=100, 150 epochs

### MLPProbeTorch
- Two-layer MLP (input → hidden → output)
- Hidden dimension: 1024
- L1/L2 regularization, batch normalization
- Default: 150 epochs with early stopping

### MLPSCLProbeTorch
- MLP with supervised contrastive learning
- Combines classification loss with contrastive objective
- Temperature-scaled contrastive weight: 0.3
- Helps learn discriminative representations

### LinearProbe (sklearn)
- Logistic regression baseline
- Fast, no deep learning required
- Regularization: C=0.00001

---

## Evaluation metrics

All experiments report:
- **AUC-ROC** — Primary metric for overflow detection performance
- **Accuracy** — Overall classification accuracy
- **F1 Score** — Harmonic mean of precision and recall
- **Average Precision** — Area under precision-recall curve

Results include mean and standard deviation across k folds.

---

## Quick start

```bash
# 1. Run full experiments (with-query + no-query)
python run_probing_experiments.py \
    --data_path /app/overflow-detection/scripts/data_preprocessing/runs/trivia_7b/probe/vectors.pt \
    --output_dir /app/overflow-detection/scripts/data_preprocessing/runs/trivia_7b/results 

# 2. Run only with-query experiments
python run_probing_experiments.py \
    --data_path data/mistral_trivia_vectors_probing.pt \
    --output_dir results/mistral_trivia/ \
    --with_query_only

# 3. Generate comparison figures
python visualize_classifiers_performance.py \
    --base_path /app/overflow-detection/scripts/data_preprocessing/runs/ \
    --datasets trivia_7b/results \
    --output_dir figures/
```

---

## Expected workflow

1. **Prepare data** — Extract embeddings from xRAG model into `.pt` format
2. **Run probing** — Execute `run_probing_experiments.py` for each dataset
3. **Analyze results** — Examine JSON outputs for metric comparisons
4. **Visualize** — Generate figures with `visualize_classifiers_performance.py`
5. **Compare** — Evaluate which embedding stages and classifiers best predict overflow
