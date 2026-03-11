## Scripts overview and end-to-end pipeline

High-level overview for the **xRAG overflow** project lives in this `scripts` directory. It connects:

- **Data preparation** (`data_preprocessing/`)
- **xRAG model training + internals** (`xRAG/`)
- **Overflow classifier probing** (`probing_experiments/`)
- **Hidden-state / saturation analysis** (`hiddens_comparison/`)
- **Context-feature comparison** (`features_comparison/`)

This README focuses on how the pieces fit together and where to look for details.

---

## Directory map (top level)

- **`data_preprocessing/`** – Build canonical samples, run baseline vs xRAG overflow pipeline, and extract probing features.  
  See `data_preprocessing/README.md`.

- **`xRAG/`** – Third-party xRAG implementation: dense retriever training, xRAG LLM training, and evaluation.  
  See `xRAG/XRAG_INTERNAL_GUIDE.md` (internal guide) and upstream docs.

- **`probing_experiments/`** – Classifier probing on embeddings (`preproj`, `postproj`, `mid`, `last`, query variants) to detect overflow.  
  See `probing_experiments/README.md`.

- **`hiddens_comparison/`** – Notebooks for hidden-state saturation analysis and context feature comparison.  
  See `hiddens_comparison/README.md`.

- **`features_comparison/`** – Notebooks for context-feature extraction and overflow-prediction comparison (e.g. perplexity, length, gzip; ROC-AUC vs probes).  
  See `features_comparison/README.md`.

---

## End-to-end pipeline

The intended workflow from raw data to analysis is:

1. **Prepare canonical samples**  
   Use `data_preprocessing/build_samples.py` (or `data_preprocessing/run_pipeline.py` with `--data ...`) to create:
   - `scripts/data_preprocessing/runs/<dataset>/samples.jsonl`

2. **Run overflow pipeline (baseline vs xRAG)**  
   Use `data_preprocessing/run_pipeline.py` to:
   - Generate baseline and xRAG answers
   - Label overflow cases where baseline is correct but xRAG is not
   - Cache background embeddings if xRAG is enabled  
   Outputs:
   - `runs/<dataset>/results.jsonl`
   - `runs/<dataset>/embeds/background_embeds.pt`

3. **Extract probing features and vectors**  
   Use `data_preprocessing/probe_pipeline.py` to run the xRAG model and save:
   - `runs/<dataset>/probe_*/features.jsonl` – scalar metrics and metadata
   - `runs/<dataset>/probe_*/vectors.pt` – tensors for classifier probing

4. **Train overflow detection probes**  
   Use `probing_experiments/run_probing_experiments.py` to train:
   - Linear/MLP/SCL probes on the `vectors.pt` file
   - Evaluate AUC-ROC and other metrics across feature combinations  
   See `probing_experiments/README.md` for exact commands and expected formats.

5. **Analyze and visualize**  
   - `probing_experiments/visualize_classifiers_performance.py` – classifier performance comparisons  
   - `hiddens_comparison` notebooks – hidden-state and saturation analysis: `getting_context_features.ipynb` → `adding_context_features.ipynb` → `visualize_xrag_vs_others.ipynb`.
   - `features_comparison` notebooks – context features and overflow-prediction comparison:  `getting_additional_context_features.ipynb` → `visualize_features_classification.ipynb`. 

---

## Quick links

- **Data preprocessing**: `data_preprocessing/README.md`
- **xRAG internals and training**: `xRAG/XRAG_INTERNAL_GUIDE.md`
- **Probing experiments**: `probing_experiments/README.md`
- **Hidden-state comparison**: `hiddens_comparison/README.md`
- **Features comparison**: `features_comparison/README.md`

Use this README as a map: start from the stage you care about (data prep, probing, or analysis) and follow the linked READMEs for concrete commands and configuration details.

