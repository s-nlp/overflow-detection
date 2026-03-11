import torch
import numpy as np
import math
from typing import Dict
from transformers import AutoTokenizer, AutoModel

import re
import json


def _normalize_answer_text_field(obj):
    """
    Robustly extract a single gold answer string from what is stored in rec["answer_text"].

    Handles:
      - already a plain string: "2003"
      - a JSON-like list string: "['2003']" or '["2003"]'
      - a real list: ["2003", "three"]
    """
    if obj is None:
        return ""

    # If it's already a list (e.g. loaded from JSONL as array)
    if isinstance(obj, list):
        return str(obj[0]) if obj else ""

    # If it's a string, try to parse list-like syntax
    if isinstance(obj, str):
        s = obj.strip()
        # e.g. "['2003']" or '["2003"]'
        if s.startswith("[") and s.endswith("]"):
            try:
                # try real JSON first
                parsed = json.loads(s)
            except json.JSONDecodeError:
                # fallback: replace single quotes with double quotes
                try:
                    parsed = json.loads(s.replace("'", '"'))
                except json.JSONDecodeError:
                    parsed = []
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
            else:
                return ""
        # otherwise treat as single string
        return s

    # Fallback
    return str(obj)


def _normalize_text_for_metric(s: str) -> str:
    """Lowercase, remove articles, punctuation, extra whitespace."""
    s = s.lower()
    # optional: strip articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # keep only letters / digits
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> int:
    """Brittle EM, kept mostly for debugging / secondary reporting."""
    if not gold:
        return 0
    return int(_normalize_text_for_metric(pred) == _normalize_text_for_metric(gold))


def squad_f1(pred: str, gold: str) -> float:
    """
    SQuAD-style F1 for a single gold answer.
    Works with longer generative answers (sentences).
    """
    if not gold:
        return 0.0

    pred_tokens = _normalize_text_for_metric(pred).split()
    gold_tokens = _normalize_text_for_metric(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def hoyer_sparsity(vec: torch.Tensor) -> float:
    """
    vec: [d] (1D tensor)
    """
    v = vec.float()
    d = v.numel()
    l1 = v.abs().sum()
    l2 = torch.sqrt((v ** 2).sum() + 1e-12)
    if d <= 1:
        return 0.0
    return float((math.sqrt(d) - (l1 / (l2 + 1e-12))) / (math.sqrt(d) - 1.0 + 1e-12))

def excess_kurtosis(vec: torch.Tensor) -> float:
    """
    Excess kurtosis over normal.
    """
    v = vec.float()
    m = v.mean()
    centered = v - m
    var = (centered ** 2).mean() + 1e-12
    m4 = (centered ** 4).mean()
    k = m4 / (var ** 2) - 3.0
    return float(k)


def spectral_entropy(vec: torch.Tensor) -> float:
    """
    Treat vec as a 1D signal, compute entropy of power spectrum.
    """
    v = vec.float()
    # real FFT
    F = torch.fft.rfft(v)
    power = (F.real ** 2 + F.imag ** 2)
    power_sum = power.sum()
    if power_sum <= 0:
        return 0.0
    p = power / power_sum
    # avoid log(0)
    p = torch.clamp(p, min=1e-12)
    H = -(p * p.log()).sum()
    return float(H)

def summarize_xrag_latents(xrag_vecs: torch.Tensor) -> Dict[str, float]:
    """
    xrag_vecs: [N_xrag, d]
    Returns aggregate metrics over N_xrag tokens.
    """
    if xrag_vecs.numel() == 0:
        return {
            "xrag_mean_l2": 0.0,
            "xrag_max_l2": 0.0,
            "xrag_mean_hoyer": 0.0,
            "xrag_max_hoyer": 0.0,
            "xrag_mean_kurtosis": 0.0,
            "xrag_max_kurtosis": 0.0,
            "xrag_mean_spectral_entropy": 0.0,
            "xrag_max_spectral_entropy": 0.0,
            "xrag_num_tokens": 0,
        }

    norms = torch.linalg.vector_norm(xrag_vecs, dim=-1)  # [N]
    hoyers = torch.tensor([hoyer_sparsity(v) for v in xrag_vecs], device=xrag_vecs.device)
    kurts = torch.tensor([excess_kurtosis(v) for v in xrag_vecs], device=xrag_vecs.device)
    ents  = torch.tensor([spectral_entropy(v) for v in xrag_vecs], device=xrag_vecs.device)

    return {
        "xrag_mean_l2": float(norms.mean()),
        "xrag_max_l2": float(norms.max()),
        "xrag_mean_hoyer": float(hoyers.mean()),
        "xrag_max_hoyer": float(hoyers.max()),
        "xrag_mean_kurtosis": float(kurts.mean()),
        "xrag_max_kurtosis": float(kurts.max()),
        "xrag_mean_spectral_entropy": float(ents.mean()),
        "xrag_max_spectral_entropy": float(ents.max()),
        "xrag_num_tokens": int(xrag_vecs.shape[0]),
    }

def match_metric(pred: str, gold: str) -> int:
    """
    Match (M): 1 if normalized gold answer appears in normalized prediction, else 0.
    Good for verbose LLM outputs where EM is too brittle.
    """
    if not gold:
        return 0

    pred_n = _normalize_text_for_metric(pred)
    gold_n = _normalize_text_for_metric(gold)

    if not pred_n or not gold_n:
        return 0

    return int(gold_n in pred_n)

def basic_norms(x: torch.Tensor) -> dict:
    x = x.flatten()
    return {
        "l2": float(torch.linalg.norm(x, ord=2).item()),
        "l1": float(torch.linalg.norm(x, ord=1).item()),
        "linf": float(torch.linalg.norm(x, ord=float("inf")).item()),
        "mean": float(x.mean().item()),
        "std": float(x.float().std(unbiased=False).item()),
        "max_abs": float(x.abs().max().item()),
    }
    
def compute_saturation_metrics(x: torch.Tensor) -> dict:
    return {
        **basic_norms(x),
        "hoyer": hoyer_sparsity(x),
        "spec_entropy": spectral_entropy(x, normalize=True),
        "excess_kurtosis": excess_kurtosis(x),
    }

def _hoyer_batch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # X: [T, D]
    X = X.float()
    T, D = X.shape
    l1 = X.abs().sum(dim=1)
    l2 = torch.sqrt((X * X).sum(dim=1) + eps)
    denom = (math.sqrt(D) - 1.0 + eps)
    return (math.sqrt(D) - (l1 / l2)) / denom

def _spectral_entropy_batch(X: torch.Tensor, eps: float = 1e-12, normalize: bool = True) -> torch.Tensor:
    # X: [T, D]
    X = X.float()
    F = torch.fft.rfft(X, dim=1).abs()                 # [T, F]
    P = F / (F.sum(dim=1, keepdim=True) + eps)
    H = -(P * (P + eps).log()).sum(dim=1)              # [T]
    if normalize:
        Hmax = math.log(P.shape[1] + eps)
        return H / Hmax
    return H

def _excess_kurtosis_batch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    X = X.float()
    mu = X.mean(dim=1, keepdim=True)
    v = X - mu
    var = (v * v).mean(dim=1) + eps
    m4 = (v ** 4).mean(dim=1)
    return m4 / (var ** 2) - 3.0

def _l2_batch(X: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(X.float(), dim=1)

def _agg(x: torch.Tensor, prefix: str) -> dict:
    x = x.detach().cpu()
    if x.numel() == 0:
        return {f"{prefix}_n": 0, f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0, f"{prefix}_p95": 0.0}

    return {
        f"{prefix}_n": int(x.numel()),
        f"{prefix}_first": float(x[0]),
        f"{prefix}_last": float(x[-1]),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_max": float(x.max()),
        f"{prefix}_p95": float(torch.quantile(x, 0.95)),
    }

@torch.no_grad()
def compute_group_saturation_metrics(
    hs: torch.Tensor,            # [seq, d] on device
    xrag_pos: torch.Tensor,      # [n_xrag] on device
) -> dict:
    xrag_pos = torch.unique(xrag_pos)
    seq_len = hs.shape[0]
    mask = torch.ones(seq_len, dtype=torch.bool, device=hs.device)
    mask[xrag_pos] = False

    hs_x = hs[xrag_pos]      # [n_xrag, d]
    hs_n = hs[mask]          # [n_non, d]

    out = {
        "seq_len": int(seq_len),
        "n_xrag": int(hs_x.shape[0]),
        "n_nonxrag": int(hs_n.shape[0]),
    }

    # per-token scalars
    def per_token(H):
        if H.numel() == 0:
            return None
        return {
            "l2": _l2_batch(H),
            "hoyer": _hoyer_batch(H),
            "spec_entropy": _spectral_entropy_batch(H),
            "excess_kurtosis": _excess_kurtosis_batch(H),
        }

    mx = per_token(hs_x)
    mn = per_token(hs_n)

    for name in ["l2", "hoyer", "spec_entropy", "excess_kurtosis"]:
        out.update(_agg(mx[name] if mx else torch.tensor([], device="cpu"), f"xrag_{name}"))
        out.update(_agg(mn[name] if mn else torch.tensor([], device="cpu"), f"nonxrag_{name}"))

    # ratios (often useful in plots)
    out["ratio_nonxrag_to_xrag_l2_mean"] = float(out["nonxrag_l2_mean"] / (out["xrag_l2_mean"] + 1e-9))
    out["ratio_nonxrag_to_xrag_entropy_mean"] = float(out["nonxrag_spec_entropy_mean"] / (out["xrag_spec_entropy_mean"] + 1e-9))
    out["ratio_nonxrag_to_xrag_hoyer_mean"] = float(out["nonxrag_hoyer_mean"] / (out["xrag_hoyer_mean"] + 1e-9))

    return out

def aggregate_attention_stats(attn_stats, layer_id: int):
    """
    attn_stats: list[dict] returned by get_xrag_attention_stats
    layer_id: which layer to aggregate (already normalized index)
    Returns small scalar summary over heads for that layer.
    """
    rows = [r for r in attn_stats if r["layer"] == layer_id]
    if len(rows) == 0:
        return {}

    keys = [
        "xrag_out_to_xrag_mean",
        "xrag_out_to_nonxrag_mean",
        "nonxrag_in_to_xrag_mean",
        "xrag_in_share",
        "xrag_out_entropy",
    ]

    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float32)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_max"]  = float(vals.max())
        out[f"{k}_min"]  = float(vals.min())
        out[f"{k}_std"]  = float(vals.std())
    # extra: ratio “XRAG attends XRAG vs nonXRAG”
    out["xrag_out_xrag_vs_nonx_ratio_mean"] = float(
        (np.array([r["xrag_out_to_xrag_mean"] for r in rows]) /
         (np.array([r["xrag_out_to_nonxrag_mean"] for r in rows]) + 1e-9)).mean()
    )
    return out
    
def compute_saturation_metrics(x: torch.Tensor) -> dict:
    return {
        **basic_norms(x),
        "hoyer": hoyer_sparsity(x),
        "spec_entropy": spectral_entropy(x),
        "excess_kurtosis": excess_kurtosis(x),
    }

def _hoyer_batch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # X: [T, D]
    X = X.float()
    T, D = X.shape
    l1 = X.abs().sum(dim=1)
    l2 = torch.sqrt((X * X).sum(dim=1) + eps)
    denom = (math.sqrt(D) - 1.0 + eps)
    return (math.sqrt(D) - (l1 / l2)) / denom

def _spectral_entropy_batch(X: torch.Tensor, eps: float = 1e-12, normalize: bool = True) -> torch.Tensor:
    # X: [T, D]
    X = X.float()
    F = torch.fft.rfft(X, dim=1).abs()                 # [T, F]
    P = F / (F.sum(dim=1, keepdim=True) + eps)
    H = -(P * (P + eps).log()).sum(dim=1)              # [T]
    if normalize:
        Hmax = math.log(P.shape[1] + eps)
        return H / Hmax
    return H

def _excess_kurtosis_batch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    X = X.float()
    mu = X.mean(dim=1, keepdim=True)
    v = X - mu
    var = (v * v).mean(dim=1) + eps
    m4 = (v ** 4).mean(dim=1)
    return m4 / (var ** 2) - 3.0

def _l2_batch(X: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(X.float(), dim=1)

def _agg(x: torch.Tensor, prefix: str) -> dict:
    x = x.detach().cpu()
    if x.numel() == 0:
        return {f"{prefix}_n": 0, f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0, f"{prefix}_p95": 0.0}

    return {
        f"{prefix}_n": int(x.numel()),
        f"{prefix}_first": float(x[0]),
        f"{prefix}_last": float(x[-1]),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_max": float(x.max()),
        f"{prefix}_p95": float(torch.quantile(x, 0.95)),
    }

@torch.no_grad()
def compute_group_saturation_metrics(
    hs: torch.Tensor,            # [seq, d] on device
    xrag_pos: torch.Tensor,      # [n_xrag] on device
) -> dict:
    xrag_pos = torch.unique(xrag_pos)
    seq_len = hs.shape[0]
    mask = torch.ones(seq_len, dtype=torch.bool, device=hs.device)
    mask[xrag_pos] = False

    hs_x = hs[xrag_pos]      # [n_xrag, d]
    hs_n = hs[mask]          # [n_non, d]

    out = {
        "seq_len": int(seq_len),
        "n_xrag": int(hs_x.shape[0]),
        "n_nonxrag": int(hs_n.shape[0]),
    }

    # per-token scalars
    def per_token(H):
        if H.numel() == 0:
            return None
        return {
            "l2": _l2_batch(H),
            "hoyer": _hoyer_batch(H),
            "spec_entropy": _spectral_entropy_batch(H),
            "excess_kurtosis": _excess_kurtosis_batch(H),
        }

    mx = per_token(hs_x)
    mn = per_token(hs_n)

    for name in ["l2", "hoyer", "spec_entropy", "excess_kurtosis"]:
        out.update(_agg(mx[name] if mx else torch.tensor([], device="cpu"), f"xrag_{name}"))
        out.update(_agg(mn[name] if mn else torch.tensor([], device="cpu"), f"nonxrag_{name}"))

    # ratios (often useful in plots)
    out["ratio_nonxrag_to_xrag_l2_mean"] = float(out["nonxrag_l2_mean"] / (out["xrag_l2_mean"] + 1e-9))
    out["ratio_nonxrag_to_xrag_entropy_mean"] = float(out["nonxrag_spec_entropy_mean"] / (out["xrag_spec_entropy_mean"] + 1e-9))
    out["ratio_nonxrag_to_xrag_hoyer_mean"] = float(out["nonxrag_hoyer_mean"] / (out["xrag_hoyer_mean"] + 1e-9))

    return out

def aggregate_attention_stats(attn_stats, layer_id: int):
    """
    attn_stats: list[dict] returned by get_xrag_attention_stats
    layer_id: which layer to aggregate (already normalized index)
    Returns small scalar summary over heads for that layer.
    """
    rows = [r for r in attn_stats if r["layer"] == layer_id]
    if len(rows) == 0:
        return {}

    keys = [
        "xrag_out_to_xrag_mean",
        "xrag_out_to_nonxrag_mean",
        "nonxrag_in_to_xrag_mean",
        "xrag_in_share",
        "xrag_out_entropy",
    ]

    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float32)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_max"]  = float(vals.max())
        out[f"{k}_min"]  = float(vals.min())
        out[f"{k}_std"]  = float(vals.std())
    # extra: ratio “XRAG attends XRAG vs nonXRAG”
    out["xrag_out_xrag_vs_nonx_ratio_mean"] = float(
        (np.array([r["xrag_out_to_xrag_mean"] for r in rows]) /
         (np.array([r["xrag_out_to_nonxrag_mean"] for r in rows]) + 1e-9)).mean()
    )
    return out
