## Data preprocessing

End-to-end **data preparation and overflow labeling** for xRAG experiments. This folder turns raw QA datasets (SQuAD v2, HotpotQA, TriviaQA) into:

- Standardized `samples.jsonl` with \(`id`, `question`, `answer`, `background`, `task_type`\)
- Overflow-labeled `results.jsonl` with baseline/xRAG predictions
- Probing-ready features and vectors (`features.jsonl`, `vectors.pt`)

---

## Overview

This module prepares QA datasets for the xRAG overflow-detection and probing pipeline by:

- **Loading raw datasets** HuggingFace `datasets`
- **Normalizing format** into a canonical open-QA schema
- **Running baseline vs xRAG generation** to label overflow
- **Extracting embeddings and latent metrics** for probing

All scripts write into an experiment-specific run directory, typically under `runs/<dataset>/...`.

---

## Contents

- **`build_samples.py`**  
  - Builds canonical `samples.jsonl` for different datasets.
  - Supports:
    - `squad_v2` via HuggingFace `datasets`
    - `hotpotqa` via `hotpotqa/hotpot_qa` (distractor setting, supporting-fact windows)
    - `triviaqa` via local xRAG-style eval JSONL + retrieval outputs
  - **Output record format**:
    - `id`: dataset-specific id  
    - `question`: string  
    - `answer`: list of answer strings  
    - `background`: list of context strings (usually length 1)  
    - `task_type`: `"open_qa"`

- **`run_pipeline.py`**  
  - Single **entry point** for aligned overflow experiments.
  - Either:
    - Builds `samples.jsonl` using `build_samples.py`, or  
    - Uses an existing `samples.jsonl` you pass in.
  - Then calls `overflow_pipeline_xrag.run_overflow_pipeline` to run:
    - **Baseline** (full-context prompt) generation
    - **xRAG** generation (compressed-context via background embeddings)
    - Overflow labeling (`overflow_label`) and metric logging.

- **`overflow_pipeline_xrag.py`**  
  - Notebook-aligned implementation of the overflow pipeline, using:
    - `src.eval.run_eval.prepare_prompts` for prompt construction
    - xRAG model (`XMistralForCausalLM`) for baseline and xRAG generation
    - SFR retriever for background embeddings when needed
  - Can cache background embeddings to `.pt` and reuse them.
  - Writes final per-sample rows with:
    - `baseline_pred`, `baseline_substring_match`
    - Optionally `xrag_pred`, `xrag_substring_match`, `overflow_label`
    - Optional `xrag_metrics` (latent metrics on xRAG tokens)

- **`probe_pipeline.py`**  
  - Takes:
    - `samples.jsonl`
    - Background embedding cache (`ctx2embed`) from the overflow pipeline
  - Runs **instrumentation** on the xRAG model to extract:
    - Projection-stage embeddings (`preproj`, `postproj`)
    - Hidden states at a mid layer and last layer (`mid`, `last`)
    - Query hidden states (`mid_q`, `last_q`)
    - Saturation metrics, attention stats, and other scalars
  - Saves:
    - `features.jsonl`: one JSON per sample with scalar metrics and metadata
    - `vectors.pt`: PyTorch tensor file with all vectors/labels for probing

- **`projection_metrics.py`**  
  - Implements low-level hooks to:
    - Run xRAG model with given background embeddings
    - Capture hidden states at different layers
    - Compute attention statistics around xRAG tokens
    - Extract projection-stage and query vectors

- **`metrics.py`**  
  - Metrics for saturation and attention analysis:
    - Hoyer sparsity, spectral entropy, kurtosis, group metrics
    - Aggregation helpers for attention statistics.

- **`data_utils.py`**  
  - Simple helpers for reading/writing JSONL and data utilities used across the pipelines.

- **`llm_utils.py`**  
  - Lightweight wrappers around xRAG model generation for:
    - Baseline generation (`generate_baseline_via_xrag`)
    - xRAG generation with latent metrics (`generate_xrag_with_latent_metrics`)

---

## Typical workflows

### 1. Build canonical samples

Use `build_samples.py` when you want to inspect or reuse `samples.jsonl` independently.

```bash
cd scripts/data_preprocessing

# SQuAD v2
python build_samples.py \
  --data squad_v2 \
  --split validation \
  --out_jsonl runs/squad/samples.jsonl \
  --max_samples 100

# HotpotQA (supporting-fact window)
python build_samples.py \
  --data hotpotqa \
  --split validation \
  --hotpot_window 1 \
  --out_jsonl runs/hotpot/samples.jsonl \
  --max_samples 100

# TriviaQA (xRAG eval layout)
python build_samples.py \
  --data triviaqa \
  --data_root /app/xRAG/data \
  --retrieval_topk 1 \
  --out_jsonl runs/triviaqa/samples.jsonl \
  --max_samples 100
```

### 2. Run overflow pipeline (baseline + xRAG)

Use `run_pipeline.py` as the **one-click** entry point. It can build samples for you or use existing ones.

```bash
cd scripts/data_preprocessing

# Build SQuAD samples + run baseline + xRAG in one go
python run_pipeline.py \
  --data squad_v2 \
  --out_dir runs/squad \
  --mode both \
  --max_samples 100 \
  --model_name_or_path /app/models/xrag-7b \
  --model_type 'mistral' \
  --retriever_name_or_path /app/models/xrag_embed \
  --device "cuda:0"

# Or reuse an existing samples.jsonl
python run_pipeline.py \
  --samples_jsonl runs/squad/samples.jsonl \
  --out_dir runs/squad \
  --mode both \
  --model_name_or_path /app/models/xrag-7b \
  --model_type 'mistral' \
  --retriever_name_or_path /app/models/xrag_embed \
  --device "cuda:0"
```

**Main outputs:**

- `runs/<dataset>/samples.jsonl` – canonical samples
- `runs/<dataset>/results.jsonl` – baseline/xRAG predictions + overflow labels
- `runs/<dataset>/embeds/background_embeds.pt` – batched background embeddings cache

### 3. Extract probing features and vectors

After you have overflow-labeled data and embedding cache, run the probing pipeline:

```bash
python probe_pipeline.py \
  --samples_jsonl /app/xlong/scripts/data_preprocessing/runs/squad/samples.jsonl \
  --results_jsonl /app/xlong/scripts/data_preprocessing/runs/squad/results.jsonl \
  --ctx2embed /app/xlong/scripts/data_preprocessing/runs/squad/embeds/background_embeds.pt \
  --out_dir /app/xlong/scripts/data_preprocessing/runs/squad/probe \
  --model_name_or_path /app/models/xrag-7b \
  --model_type 'mistral' \
  --retriever_name_or_path /app/models/xrag_embed \
  --device cuda:0 \
  --max_samples 100 \
  --mid_layer_index 16 \
  --save_every 200 \
  --project_questions \
  --recompute_question_embeds \
  --save_vectors_pt
```

**Final outputs used for classifier probing:**

- `runs/<dataset>/probe_m16/features.jsonl` – scalar metrics + metadata
- `runs/<dataset>/probe_m16/vectors.pt` – tensors:
  - `ids`, `labels` (overflow labels)
  - `preproj`, `postproj`, `mid`, `last`
  - `mid_q`, `last_q`
  - Optionally `preproj_q`, `postproj_q` if question projection is enabled

These files are the expected input to the probing experiments in `scripts/probing_experiments/`.

---
