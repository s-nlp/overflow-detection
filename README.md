# <img src="assets/logo_overflow.png" height="40" alt="logo" /> Detecting Overflow in Compressed Token Representations for Retrieval-Augmented Generation

[![arXiv](https://img.shields.io/badge/arXiv-2602.12235-b31b1b.svg)](https://arxiv.org/abs/2602.12235)

This repository accompanies the paper. It studies **overflow** in soft compression for RAG, specifically in the [xRAG](https://github.com/ZhuohanGu/xRAG) setting — the failure mode where a compressed `<xRAG>` token loses enough information to cause a wrong answer. We evaluate overflow detection stage-wise across the compression pipeline, supporting pre-LLM gating decisions (re-retrieve, re-chunk, or fall back to full context) and a consistent way to compare compression modules by checking *where* overflow first becomes detectable.

## Contents

- [What is overflow?](#what-is-overflow)
- [Setup](#setup)
- [Quickstart](#quickstart-end-to-end)
- [Citation](#citation)

## What is overflow?

We study **token overflow** in projector-based soft compression for RAG: `Retriever Embeddings  →  Compression Projector  →  LLM Generator`.

**Overflow** is compression-induced task failure: the model succeeds on the full context ($\mathcal{T}_i^{\text{ref}} = 1$), but fails when that context is replaced by its compressed representation ($\mathcal{T}_i(\mathbf{C}_i) = 0$).

$$\mathcal{O}_i = \mathbf{1}\left(\mathcal{T}_i^{\text{ref}} = 1 \land \mathcal{T}_i(\mathbf{C}_i) = 0\right)$$

> Context → Compression → ⚠️ **Overflow** ⚠️ → Wrong answer

## Setup

### Prerequisites

You need two pre-trained models:

| Model | Role | Where to get it? |
|---|---|---|
| LLM with Projector | Generates baseline and xRAG answers and provides hidden states | [xRAG repo](https://github.com/ZhuohanGu/xRAG) |
| Retriever | Encodes background passages into xRAG embeddings | [Salesforce/SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral) |

### Install

```bash
pip install .
```

This installs dependencies for the full pipeline under `scripts/`. The [`scripts/xRAG/`](scripts/xRAG/) submodule has its own setup — see its upstream docs.

## Quickstart

The main probing pipeline has four stages. Each stage writes output files that the next stage reads.

### Step 1 — Build canonical QA samples

```bash
cd scripts/data_preprocessing

python build_samples.py \
  --data triviaqa \
  --split validation \
  --out_jsonl runs/triviaqa/samples.jsonl \
  --max_samples 100
```

Output: `runs/triviaqa/samples.jsonl` — records with `id`, `question`, `answer`, `background`, `task_type`.

### Step 2 — Run baseline vs. xRAG and label overflow

```bash
python run_pipeline.py \
  --samples_jsonl runs/triviaqa/samples.jsonl \
  --out_dir runs/triviaqa \
  --mode both \
  --model_name_or_path /path/to/xrag-7b \
  --retriever_name_or_path /path/to/xrag_embed \
  --device cuda:0
```

Outputs:
- `runs/triviaqa/results.jsonl` — per-sample predictions + `overflow_label`
- `runs/triviaqa/embeds/background_embeds.pt` — cached background embeddings

### Step 3 — Extract probing vectors

```bash
python probe_pipeline.py \
  --samples_jsonl runs/triviaqa/results.jsonl \
  --ctx2embed runs/triviaqa/embeds/background_embeds.pt \
  --out_dir runs/triviaqa/probe_m16 \
  --model_name_or_path /path/to/xrag-7b \
  --device cuda:0 \
  --mid_layer_index 16 \
  --project_questions \
  --save_vectors_pt
```

Outputs:
- `runs/triviaqa/probe_m16/vectors.pt` — tensors keyed by stage (`preproj`, `postproj`, `mid`, `last`, `mid_q`, `last_q`) + `labels`
- `runs/triviaqa/probe_m16/features.jsonl` — scalar saturation metrics per sample

> **Alternative:** [`scripts/feature_collection/run_feature_extraction.ipynb`](scripts/feature_collection/run_feature_extraction.ipynb) runs Steps 2–3 in a single notebook pass and also adds context-level features (perplexity, length, gzip) to `features.jsonl`.

### Step 4 — Train and evaluate overflow detectors

```bash
cd ../probing_experiments

python run_probing_experiments.py \
  --data_path ../data_preprocessing/runs/triviaqa/probe_m16/vectors.pt \
  --output_dir results/triviaqa_probe_m16 \
  --cv_folds 5 \
  --device 0
```

This trains lightweight classifiers on all embedding stage combinations with stratified 5-fold CV, and writes JSON result files with AUC-ROC, F1, and average precision.

## Citation

```bibtex
@misc{belikova2026detectingoverflowcompressedtoken,
  title        = {Detecting Overflow in Compressed Token Representations for Retrieval-Augmented Generation},
  author       = {Julia Belikova and Danila Rozhevskii and Dennis Svirin and Konstantin Polev and Alexander Panchenko},
  year         = {2026},
  eprint       = {2602.12235},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2602.12235},
}
```
