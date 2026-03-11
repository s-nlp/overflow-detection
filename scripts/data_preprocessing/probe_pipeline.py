#!/usr/bin/env python3
"""
probe_pipeline.py

Run instrumentation (hooks + metrics + attention stats) on a set of canonical samples,
saving incremental JSONL features and final vectors.pt.

Usage example:
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
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
import numpy as np
import gc

# Project-specific helpers (adjust import paths if needed)
from projection_metrics import (  # your module: get_xrag_states_with_projection, get_xrag_attention_stats, take_first_token_vec, prompt_xrag_mistral, format_one_example
    get_xrag_states_with_projection,
    get_xrag_attention_stats,
    extract_vector,
)
from metrics import (  # your module: compute_saturation_metrics, compute_group_saturation_metrics, aggregate_attention_stats
    compute_saturation_metrics,
    compute_group_saturation_metrics,
    aggregate_attention_stats,
)
from overflow_pipeline_xrag import (
    _read_jsonl, 
    _pack_single_text_embeds, 
    _load_retriever,
    _load_embed_cache, 
    _save_embed_cache
)

# ---------------------------
# I/O helpers
# ---------------------------

def load_ctx2embed(path: str):
    """
    Support JSON (id -> list) or .pt/.pth files saved as:
      - dict {id: [vals,...], ...}  OR
      - torch.save({"meta":..., "embeds": Tensor, "ids": [...]}) OR
      - torch.save({"id_list": [...], "embeds": Tensor})
    Returns either Python dict (JSON) or a dict with keys "ids" and "embeds" (Tensor on CPU float32).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        payload = torch.load(path, map_location="cpu")
        # If mapping id->list
        if isinstance(payload, dict) and all(not isinstance(v, torch.Tensor) for v in payload.values()):
            # assume id->list
            return payload
        # If saved as {"meta":..., "embeds": Tensor, "ids": [...]}
        if isinstance(payload, dict) and "embeds" in payload:
            embeds = payload["embeds"]
            ids = payload.get("ids") or payload.get("id_list") or payload.get("meta", {}).get("ids")
            if ids is None:
                # fallback numeric ids if nothing provided
                ids = [str(i) for i in range(embeds.shape[0])]
            return {"ids": [str(i) for i in ids], "embeds": embeds.float()}
        # raw tensor
        if isinstance(payload, torch.Tensor):
            ids = [str(i) for i in range(payload.shape[0])]
            return {"ids": ids, "embeds": payload.float()}
        raise ValueError("Unsupported ctx2embed format at: " + path)

def load_ctx2embed_normalized(path: str) -> Dict[str, Any]:
    """
    Loads ctx2embed data from .pt/.pth (preferred) or .json legacy.
    Returns a canonical dict mapping id (str) -> list[list[float]] (list of segment vectors).
    Also returns meta if present.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if path.endswith(".json"):
        # legacy JSON mapping id->list or id->[ [seg1], [seg2], ... ]
        import json
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # normalize: ensure every value is a list-of-vectors
        out = {}
        for k, v in raw.items():
            arr = np.array(v)
            # collapse extra dims conservatively:
            if arr.ndim == 1:
                out[str(k)] = [arr.astype(float).tolist()]
            else:
                # collapse leading singleton dims then reshape to (-1, H)
                H = arr.shape[-1]
                out[str(k)] = arr.reshape(-1, H).astype(float).tolist()
        return out, None

    # .pt / .pth loader
    payload = torch.load(path, map_location="cpu")
    # If payload was saved as {"embeds": Tensor, "meta": {...}}
    if isinstance(payload, dict) and "embeds" in payload:
        embeds = payload["embeds"].float().cpu()   # [B, maxN, H]
        meta = payload.get("meta", {})
        ids = [str(x) for x in meta.get("ids", [])]
        lengths = meta.get("lengths", None)

        if lengths is None:
            # If lengths not present, try to infer per-id lengths by scanning for zero rows.
            # This is fragile (zeros may be valid) so prefer writing lengths in meta.
            B, maxN, H = embeds.shape
            inferred = []
            for i in range(B):
                row = embeds[i]  # [maxN, H]
                # detect "empty" rows as all zeros
                nonzero_mask = ~(torch.all(row == 0.0, dim=-1))
                nseg = int(nonzero_mask.sum().item())
                inferred.append(nseg)
            lengths = inferred

        out = {}
        for i, _id in enumerate(ids):
            nseg = int(lengths[i]) if i < len(lengths) else embeds.shape[1]
            segs = embeds[i, :nseg, :].numpy().astype(float).tolist()
            out[str(_id)] = segs
        return out, meta

    # If payload is a raw tensor saved (N,H) or (B,N,H)
    if isinstance(payload, torch.Tensor):
        t = payload.float().cpu()
        if t.ndim == 2:
            # assume B x H mapping where ids will be numeric
            ids = [str(i) for i in range(t.shape[0])]
            out = {ids[i]: [t[i].numpy().astype(float).tolist()] for i in range(t.shape[0])}
            return out, {"ids": ids, "lengths":[1]*t.shape[0]}
        elif t.ndim == 3:
            # B x maxN x H
            B, maxN, H = t.shape
            ids = [str(i) for i in range(B)]
            out = {}
            for i in range(B):
                # default lengths = maxN
                segs = t[i].numpy().reshape(-1, H).astype(float).tolist()
                out[ids[i]] = segs
            return out, {"ids": ids, "lengths":[maxN]*B}
    raise ValueError("Unsupported ctx2embed payload format at: " + path)

def write_jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------------------
# Resume helper
# ---------------------------
def read_existing_ids(features_path: Path) -> set:
    if not features_path.exists():
        return set()
    ids = set()
    with features_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                if "id" in j:
                    ids.add(str(j["id"]))
            except Exception:
                continue
    return ids

# ---------------------------
# Question embed utilities
# ---------------------------

def load_question_embeds(path: str) -> Tuple[List[str], torch.Tensor]:
    """
    Load question embeddings saved as JSON (id -> list) or .pt saved payload.
    Returns (ids_list, tensor [N, H] on CPU float32).
    """
    data = load_ctx2embed(path)
    if isinstance(data, dict) and "embeds" in data and "ids" in data:
        ids = [str(x) for x in data["ids"]]
        embeds = data["embeds"].float().cpu()
        return ids, embeds
    # if pure dict id->list (json loaded)
    if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()) and not ("embeds" in data):
        ids = list(data.keys())
        rows = [torch.tensor(data[k], dtype=torch.float32) for k in ids]
        return ids, torch.stack(rows, dim=0).float().cpu()
    raise ValueError("Unsupported question embeds format: " + path)

def reorder_embeds_to_ids(ids_target: List[str], ids_src: List[str], embeds_src: torch.Tensor) -> torch.Tensor:
    """
    Reorder src embeddings (ids_src order) to match ids_target order.
    Raises KeyError if some ids are missing.
    Returns tensor [len(ids_target), H] on CPU float32.
    """
    id_to_idx = {str(k): i for i, k in enumerate(ids_src)}
    out_rows = []
    for tid in ids_target:
        tid_s = str(tid)
        if tid_s not in id_to_idx:
            raise KeyError(f"id {tid_s} not found in question embeds (source size {len(ids_src)})")
        out_rows.append(embeds_src[id_to_idx[tid_s]])
    return torch.stack(out_rows, dim=0).float().cpu()

def project_question_preembeds(
    model,
    preembeds: torch.Tensor,   # [N, H], CPU float
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 64,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Project question pre-embeds through model.projector in batches.
    Temporarily moves projector to device and returns CPU tensor [N, D_out] float32.
    """
    if preembeds.numel() == 0:
        return preembeds

    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    projector = model.projector
    proj_device_before = next(projector.parameters()).device
    orig_requires_grad = any(p.requires_grad for p in projector.parameters())

    N = int(preembeds.shape[0])
    out_list = []
    try:
        projector.to(device)
        projector.eval()
        with torch.inference_mode():
            for i in range(0, N, batch_size):
                b = preembeds[i : i + batch_size].to(device=device, dtype=dtype)
                out = projector(b)
                out_list.append(out.detach().to("cpu"))
    finally:
        # try moving back; if it fails, we at least free cuda memory
        try:
            projector.to(proj_device_before)
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(out_list, dim=0).float().cpu()

# Inference helper
def projector_forward(model, x: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """
    Runs model.projector on x, moving x to projector's device.
    Returns CPU float16 vector(s).
    """
    proj = model.projector
    proj_device = next(proj.parameters()).device
    with torch.inference_mode():
        y = proj(x.to(device=proj_device, dtype=dtype))
    return y.detach().to("cpu", dtype=torch.float16)

# ---------------------------
# Main driver
# ---------------------------

def run_probe_pipeline(
    samples_jsonl: str,
    results_jsonl: str,
    ctx2embed_path: str,
    model,
    model_type: str,
    tokenizer,
    retriever_name_or_path,
    device: str = "cuda",
    out_dir: str = "runs/probe",
    mid_layer_index: int = 16,
    sae = None,
    sae_layer_index_for_hidden_states: int = 16,
    save_vectors_pt: bool = True,
    save_full_sae_vectors: bool = False,
    max_samples: Optional[int] = None,
    save_every: int = 500,
    resume: bool = True,
    question_embeds_cache: str = None,
    recompute_question_embeds: bool = False,
    embed_batch_size: int = 16,
    embed_max_len: int = 512,
    question_embeds: Optional[str] = None,
    project_questions: bool = False,
    projector_device: str = "cpu",
    projector_dtype: str = "bfloat16",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "features.jsonl"
    vectors_path = out_dir / "vectors.pt"
    meta_path = out_dir / "probe_meta.json"

    samples = _read_jsonl(samples_jsonl)
    results = _read_jsonl(results_jsonl)
    ctx2embed_map, meta = load_ctx2embed_normalized(ctx2embed_path)  # may be dict id->list OR {"ids":..., "embeds":Tensor}

    q_embeds = None
    q_meta = None
    
    if question_embeds_cache is not None or project_questions:
        # decide cache path
        if question_embeds_cache is None:
            question_embeds_cache = str(Path(out_dir) / "embeds" / "question_embeds.pt")
    
        if (not recompute_question_embeds): #and os.path.exists(question_embeds_cache):
            q_embeds, q_meta = _load_embed_cache(question_embeds_cache)  # float32 CPU, shape [B,1,H]
            if q_embeds.shape[0] != len(samples):
                raise ValueError(f"Question embed cache B mismatch: {q_embeds.shape[0]} vs {len(samples)}")
        else:
            if retriever_name_or_path is None:
                raise ValueError("Need --retriever_name_or_path to compute question embeds.")
            print("Loading retriever for question embeds...")
            retriever, retr_tok = _load_retriever(retriever_name_or_path)
    
            questions = [str(s.get("question","")).strip() for s in samples]
            q_embeds, q_meta = _pack_single_text_embeds(
                samples=samples,
                texts=questions,
                embed_max_len=embed_max_len,
                embed_batch_size=embed_batch_size,
                embed_device=device,
                retriever=retriever,
                retr_tok=retr_tok,
            )
            _save_embed_cache(question_embeds_cache, q_embeds, q_meta)
    
            del retriever, retr_tok
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # resume: skip already done ids
    done_ids = read_existing_ids(features_path) if resume else set()
    
    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    XRAG_DIR = os.path.abspath(os.path.join(HERE, "..", "xRAG"))
    sys.path.insert(0, XRAG_DIR)
    from transformers import AutoTokenizer
    from src.model import XMistralForCausalLM, XMixtralForCausalLM  # or your LLM wrapper
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding_side="left", add_eos_token=False, use_fast=False)
    if model_type == 'mistral':
        model = XMistralForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto", attn_implementation="eager")
    else:
        model = XMixtralForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto", attn_implementation="eager")
    model.to(device)
    model.eval()

    # run lists
    ids_list: List[str] = []
    labels_list: List[int] = []
    preproj_list = []
    postproj_list = []
    mid_list = []
    last_list = []
    mid_q_list = []
    last_q_list = []
    preproj_q_list = []
    postproj_q_list = []
    sae_xrag_vecs = []
    sae_other_mean_vecs = []

    n_ok = 0
    n_fail = 0
    processed = 0

    start_time = time.time()

    # iterate and map to results for overlfow label
    results_by_id = {str(r["id"]): r for r in results}
    
    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        if max_samples is not None and i >= max_samples:
            break
        sid = str(sample.get("id", ""))
        if sid in done_ids:
            tqdm.write(f"[skip] {sid} already in {features_path.name}")
            continue

        result = results_by_id.get(sid)
        if result is None:
            tqdm.write(f"[warn] no result for {sid}")
            continue

        try:
            # --- call the main instrumentation functions
            out = get_xrag_states_with_projection(
                sample=sample, model=model, tokenizer=tokenizer,
                ctx2embed=ctx2embed_map,
                mid_layer_index=mid_layer_index, task_type="open_qa",
                use_rag=True, device=device
            )
            # attention stats (optional but useful)
            attn_stats = get_xrag_attention_stats(
                sample=sample, model=model, tokenizer=tokenizer,
                ctx2embed=ctx2embed_map, layer_indices=[mid_layer_index, -1],
                device=device
            )

            # take vectors 
            preproj = extract_vector(out["retrieval_embeds_preproj"], "retrieval_embeds_preproj", mode="first")
            postproj = extract_vector(out["proj_out_postproj"], "proj_out_postproj", mode="first")
            mid = out["mid_xrag"][0].detach().cpu()
            last = out["last_xrag"][0].detach().cpu()
            mid_q = out["mid_q"][0].detach().cpu()
            last_q = out["last_q"][0].detach().cpu()

            # compute scalar metrics
            row = {
                "id": sample.get("id"),
                "gold_answer": sample.get("answer", sample.get("gold_answer", [])),
                "task_type": sample.get("task_type", "open_qa"),
                "input_len": int(out["input_len"]),
                "xrag_pos": int(out["token_indices"][0].item()) if hasattr(out["token_indices"], "shape") and out["token_indices"].numel() > 0 else -1,
                "mid_layer_index": out["mid_layer_index"],
                "last_layer_index": out["last_layer_index"],
                "preproj_metrics": compute_saturation_metrics(preproj),
                "postproj_metrics": compute_saturation_metrics(postproj),
                "mid_metrics": compute_saturation_metrics(mid),
                "last_metrics": compute_saturation_metrics(last),
                "mid_group_metrics": out.get("mid_group_metrics", {}),
                "last_group_metrics": out.get("last_group_metrics", {}),
                "attn_mid": aggregate_attention_stats(attn_stats, layer_id=mid_layer_index),
                "attn_last": aggregate_attention_stats(attn_stats, layer_id=max([r["layer"] for r in attn_stats]) if len(attn_stats) else out["last_layer_index"]),
                "overflow_label": result.get("overflow_label", None),
            }

            # append JSONL row (incremental safe write)
            write_jsonl_append(features_path, row)

            # collect vectors
            if save_vectors_pt:
                #print("collecting vectors...")
                ids_list.append(sid)
                labels_list.append(int(result.get("overflow_label", 0) if result.get("overflow_label", None) is not None else 0))
                preproj_list.append(preproj.to(torch.float16))
                postproj_list.append(postproj.to(torch.float16))
                mid_list.append(mid.to(torch.float16))
                last_list.append(last.to(torch.float16))
                mid_q_list.append(mid_q.to(torch.float16))
                last_q_list.append(last_q.to(torch.float16))

                if q_embeds is not None:
                    pre_q = q_embeds[i, 0].to(torch.float16)   # q_embeds is CPU already
                    preproj_q_list.append(pre_q)
            
                    if project_questions:
                        post_q = projector_forward(model, pre_q, dtype=torch.bfloat16)
                        postproj_q_list.append(post_q)

            n_ok += 1
            processed += 1

            # periodic checkpoint save for vectors
            if save_vectors_pt and (processed % save_every == 0):
                # build checkpoint dict
                ck = {
                    "ids": ids_list,
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                    "preproj": torch.stack(preproj_list, dim=0),
                    "postproj": torch.stack(postproj_list, dim=0),
                    "mid": torch.stack(mid_list, dim=0),
                    "last": torch.stack(last_list, dim=0),
                    "mid_q": torch.stack(mid_q_list, dim=0),
                    "last_q": torch.stack(last_q_list, dim=0),
                }
                # include question vectors if computed
                if q_embeds is not None:
                    ck["preproj_q"] = torch.stack(preproj_q_list, dim=0)
                    if project_questions:
                        ck["postproj_q"] = torch.stack(postproj_q_list, dim=0)
        
                tmp_path = vectors_path.with_suffix(".tmp.pt")
                torch.save(ck, tmp_path)
                tmp_path.replace(vectors_path)
                tqdm.write(f"[checkpoint] saved {vectors_path} after {processed} samples")

        except Exception as e:
            n_fail += 1
            tqdm.write(f"[error] id={sid} exception: {repr(e)}")
            write_jsonl_append(features_path, {"id": sid, "error": str(e)})
            continue

    # final save vectors
    if save_vectors_pt and len(ids_list) > 0:
        print("final data save preparation...")
        final_data = {
            "ids": ids_list,
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "preproj": torch.stack(preproj_list, dim=0),
            "postproj": torch.stack(postproj_list, dim=0),
            "mid": torch.stack(mid_list, dim=0),
            "last": torch.stack(last_list, dim=0),
            "mid_q": torch.stack(mid_q_list, dim=0),
            "last_q": torch.stack(last_q_list, dim=0),
        }
        if q_embeds is not None:
            final_data["preproj_q"] = torch.stack(preproj_q_list, dim=0)
            if project_questions:
                final_data["postproj_q"] = torch.stack(postproj_q_list, dim=0)
        
        torch.save(final_data, vectors_path)
        print("[done] wrote vectors:", vectors_path)

    # write meta
    meta = {"n_ok": n_ok, "n_fail": n_fail, "time_s": time.time() - start_time, "n_processed": processed}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("FINISHED. ok:", n_ok, "fail:", n_fail)

    print(len(ids_list), len(preproj_list), len(preproj_q_list))
    if project_questions:
        print(len(postproj_q_list))

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_jsonl", required=True)
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--ctx2embed", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument(
        "--model_type",
        type=str,
        default="mistral",
        choices=["mistral", "mixtral"],
        help="Which XRAG model architecture to load"
    )
    ap.add_argument("--retriever_name_or_path", type=str, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mid_layer_index", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--save_vectors_pt", action="store_true")
    ap.add_argument("--resume", action="store_true")
    # question options
    ap.add_argument("--question_embeds_cache", type=str, default=None, help="Where to save/load question embeds (.pt). Default: <out_dir>/embeds/question_embeds.pt")
    ap.add_argument("--recompute_question_embeds", action="store_true")
    ap.add_argument("--embed_batch_size", type=int, default=16)
    ap.add_argument("--embed_max_len", type=int, default=512)
    
    ap.add_argument("--question_embeds", type=str, default=None, help="Path to question embeds (.json or .pt) keyed by sample id")
    ap.add_argument("--project_questions", action="store_true", help="Run model.projector on question preembeds and store postproj_q")
    ap.add_argument("--projector_device", type=str, default="cpu", help="Device to run projector on (cpu or cuda)")
    ap.add_argument("--projector_dtype", type=str, default="bfloat16", help="dtype for projector (bfloat16 or float16)")

    args = ap.parse_args()

    # load model / tokenizer here - adapt to your loader
    #model.eval()

    run_probe_pipeline(
        samples_jsonl=args.samples_jsonl,
        results_jsonl=args.results_jsonl,
        ctx2embed_path=args.ctx2embed,
        model=args.model_name_or_path,
        model_type=args.model_type,
        tokenizer=args.model_name_or_path,
        retriever_name_or_path = args.retriever_name_or_path,
        device=args.device,
        out_dir=args.out_dir,
        mid_layer_index=args.mid_layer_index,
        sae=None,
        save_vectors_pt=args.save_vectors_pt,
        max_samples=args.max_samples,
        save_every=args.save_every,
        resume=args.resume,
        question_embeds_cache=args.question_embeds_cache,
        recompute_question_embeds = args.recompute_question_embeds,
        embed_batch_size = args.embed_batch_size,
        embed_max_len = args.embed_max_len,
        question_embeds=args.question_embeds,
        project_questions=args.project_questions,
        projector_device=args.projector_device,
        projector_dtype=args.projector_dtype,
    )

if __name__ == "__main__":
    main()

