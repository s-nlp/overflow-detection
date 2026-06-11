
"""
llm_utils.py

Thin wrappers that reuse xRAG's notebook-style helpers (llm_for_open_generation / prepare_prompts)
while still computing latent metrics over <xRAG> tokens.

Design goals:
- Do NOT implement custom prompt templates here.
- Assume prompts are produced by xRAG (e.g., prepare_prompts), so baseline vs xRAG
  differs only by whether retrieval_embeds is passed to the LLM generation.
- Provide optional latent-metrics extraction for xRAG prompts by running a forward pass
  and summarizing hidden states at <xRAG> token positions.

You must ensure xRAG is importable (either installed as a package, or its repo root is on PYTHONPATH).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import os, sys

# Your project-local metric summarizer
from metrics import summarize_xrag_latents

# xRAG imports (same pattern as the notebook: reuse existing utilities)
HERE = os.path.dirname(os.path.abspath(__file__))
XRAG_DIR = os.path.abspath(os.path.join(HERE, "..", "xRAG"))

sys.path.insert(0, XRAG_DIR)
from src.eval.utils import get_substring_match_score  # optional but handy for callers
from src.language_modeling.utils import XRAG_TOKEN  # the token string used by the model
from src.eval.run_eval import (
    llm_for_open_generation,
)

def _ensure_pad_token(tok) -> None:
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token_id = tok.eos_token_id


@torch.no_grad()
def generate_baseline_via_xrag(
    prompts: Sequence[str],
    llm,
    tok,
    **gen_kwargs: Any,
) -> List[str]:
    """
    Baseline generation using xRAG's llm_for_open_generation, but WITHOUT retrieval_embeds.
    Prompts should already contain the real background text (not <xRAG> placeholders).
    """
    #_ensure_pad_token(tok)
    # We intentionally do not pass retrieval_embeds.
    outputs = llm_for_open_generation(
        prompts=prompts,
        llm=llm,
        llm_tokenizer=tok,
        retrieval_embeds=None,   # None for non-xRAG modes
        batch_size=4,
        enable_progress_bar=True,
        #bad_words_ids=[[x_id]],  # don't let model re-emit XRAG token
    )
    return outputs


@torch.no_grad()
def generate_xrag_with_latent_metrics(
    prompts: Sequence[str],
    retrieval_embeds: torch.Tensor,
    llm,
    tok,
    max_new_tokens: int = 64,
    scale: float = 1.0,
    **gen_kwargs: Any,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    xRAG generation using xRAG's llm_for_open_generation, WITH retrieval_embeds,
    plus latent metrics over <xRAG> prompt token positions.

    Assumptions:
    - prompts contain exactly N occurrences of XRAG_TOKEN for each sample, where N matches
      retrieval_embeds.shape[1] (if batched) or the per-sample segment count.
    - retrieval_embeds is either:
        [B, N, H]  (preferred)
      or [N, H] for a single-sample call (we'll unsqueeze to [1, N, H]).
    """

    # ---- 1) Generate answers via xRAG helper ----
    outputs = llm_for_open_generation(
        prompts=prompts,
        llm=llm,
        llm_tokenizer=tok,
        retrieval_embeds=retrieval_embeds,   # None for non-xRAG modes
        batch_size=4,
        enable_progress_bar=True,
        #bad_words_ids=[[x_id]],  # don't let model re-emit XRAG token
    )

    # ---- 2) Compute latent metrics for each prompt via forward pass ----
    metrics_list: List[Dict[str, Any]] = []
    # for i, prompt in enumerate(prompts):
    #     inp = tok(prompt, return_tensors="pt")
    #     inp = {k: v.to(dev) for k, v in inp.items()}

    #     hs_out = llm(
    #         **inp,
    #         retrieval_embeds=re[i : i + 1],
    #         output_hidden_states=True,
    #         use_cache=False,
    #         return_dict=True,
    #     )
    #     last_hs = hs_out.hidden_states[-1][0]  # [seq_len, hidden]
    #     inp_ids = inp["input_ids"][0]          # [seq_len]

    #     xrag_mask = (inp_ids == x_id)
    #     if xrag_mask.any():
    #         xrag_vecs = last_hs[xrag_mask]     # [N_xrag, hidden]
    #     else:
    #         xrag_vecs = last_hs.new_zeros((0, last_hs.size(-1)))

    #     metrics_list.append(summarize_xrag_latents(xrag_vecs))

    return outputs, metrics_list


