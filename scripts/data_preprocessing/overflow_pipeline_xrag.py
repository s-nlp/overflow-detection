
"""
overflow_pipeline_xrag.py

Notebook-aligned pipeline:
- Uses xRAG's prepare_prompts to build prompts (no custom prompt templates here)
- Uses xRAG's llm_for_open_generation for generation
- Computes latent metrics over <xRAG> tokens for xRAG runs
- Supports optional "only baseline-correct" filtering before labeling overflow

Expected sample format (canonical):
  { "id": ..., "question": str, "background": [str, ...], "answer": [str, ...] }
Your existing Hotpot preprocessor produces "gold_answer"; we accept that too.
"""
    
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from tqdm.auto import tqdm
import gc

import torch
from transformers import AutoTokenizer
from rich.progress import track

from llm_utils import generate_baseline_via_xrag, generate_xrag_with_latent_metrics

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
XRAG_DIR = os.path.abspath(os.path.join(HERE, "..", "xRAG"))

sys.path.insert(0, XRAG_DIR)
    
from src.eval.utils import get_substring_match_score
from src.model import SFR, XMistralForCausalLM, XMixtralForCausalLM
import src.eval.run_eval as run_eval
from src.eval.run_eval import (
    prepare_prompts,
    llm_for_open_generation,
)
from src.language_modeling.utils import XRAG_TOKEN, get_retrieval_embeds
from src.eval.utils import (
    stop_sequences_criteria,
    get_substring_match_score,
    eval_fact_checking,
    eval_truthfulqa,
    keyword_extraction_with_tfidf,
)
# Retrieval embedding helper: prefer xRAG's, fallback to local embedding
try:
    from src.retrieval.utils import prepare_retrieval_embeds  # type: ignore
except Exception:
    prepare_retrieval_embeds = None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _ensure_xrag_imports(xrag_dir: Optional[str]) -> None:
    """
    Make sure `src.*` from xRAG is importable.
    Prefer:
      1) explicit --xrag_dir
      2) env XRAG_DIR
      3) ./xRAG
      4) ./third_party/xRAG
    """
    import sys

    candidates: List[str] = []
    if xrag_dir:
        candidates.append(xrag_dir)
    env_dir = os.environ.get("XRAG_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates += ["./xRAG", "./third_party/xRAG"]

    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            # Only need one that works; don't spam sys.path.
            break


def _load_retriever(retriever_name_or_path: str) -> Tuple[SFR, Any]:
    retr_tok = AutoTokenizer.from_pretrained(retriever_name_or_path, use_fast=False)
    retriever = SFR.from_pretrained(
        retriever_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    retriever.eval()
    return retriever, retr_tok

from tqdm.auto import tqdm
import torch

def _embed_with_fallback(
    flat_texts,
    retriever,
    retr_tok,
    embed_max_len: int = 256,
    batch_size: int = 16,
    device: str = "cuda:0",
    show_progress: bool = True,
):
    retriever.eval()#.to(device)

    outs = []
    total = len(flat_texts)
    it = range(0, total, batch_size)
    if show_progress:
        it = tqdm(it, total=(total + batch_size - 1) // batch_size, desc="Embedding passages")

    for i in it:
        batch = flat_texts[i:i + batch_size]
        inp = retr_tok(
            batch,
            max_length=embed_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inp = {k: v.to(device) for k, v in inp.items()}

        with torch.inference_mode():
            E = retriever.get_doc_embedding(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
            )

        outs.append(E.detach().to("cpu", dtype=torch.float16))
        del inp, E

        # Optional: show GPU mem in the progress bar

    return torch.cat(outs, dim=0)

def _pack_embeds(
    samples: List[Dict[str, Any]],
    backgrounds: List[List[str]],
    embed_max_len: int,
    embed_batch_size: int,
    embed_device: str,
    retriever: SFR,
    retr_tok: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Embed all background passages, then pack into [B, maxN, H] (CPU fp32).
    Returns (packed_embeds, meta)
    """
    flat_backgrounds: List[str] = []
    offsets: List[Tuple[int, int]] = []
    start = 0
    for bg_list in backgrounds:
        flat_backgrounds.extend(bg_list)
        end = start + len(bg_list)
        offsets.append((start, end))
        start = end

    E = _embed_with_fallback(
        flat_backgrounds,
        retriever,
        retr_tok,
        embed_max_len=embed_max_len,
        batch_size=embed_batch_size,
        device=embed_device,
    )
    H = int(E.shape[-1])
    maxN = max((e - s) for (s, e) in offsets) if offsets else 0
    B = len(samples)

    packed = torch.zeros((B, maxN, H), dtype=torch.float32)
    for i, (s, e) in enumerate(offsets):
        segs = E[s:e]
        packed[i, : segs.shape[0]] = segs

    meta = {
        "ids": [s["id"] for s in samples],
        "B": B,
        "maxN": maxN,
        "H": H,
        "embed_max_len": embed_max_len,
        "note": "Embeddings are aligned to the prompts' `backgrounds` order from prepare_prompts.",
    }
    return packed, meta


def _pack_single_text_embeds(
    samples: List[Dict[str, Any]],
    texts: List[str],
    *,
    embed_max_len: int,
    embed_batch_size: int,
    embed_device: str,
    retriever,
    retr_tok,
    desc: str = "Embedding texts",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Embed one string per sample, return packed embeds [B, 1, H] on CPU float32,
    plus meta with ids/lengths.
    """
    assert len(samples) == len(texts), "texts must align 1:1 with samples"

    E = _embed_with_fallback(
        flat_texts=texts,
        retriever=retriever,
        retr_tok=retr_tok,
        embed_max_len=embed_max_len,
        batch_size=embed_batch_size,
        device=embed_device,
        show_progress=True,
    )  # [B, H] on CPU fp16

    if E.ndim != 2:
        raise RuntimeError(f"Expected E as [B, H], got {tuple(E.shape)}")

    B, H = E.shape
    packed = E.float().unsqueeze(1)  # [B, 1, H]

    meta = {
        "ids": [str(s["id"]) for s in samples],
        "lengths": [1] * B,
        "B": B,
        "maxN": 1,
        "H": H,
        "embed_max_len": embed_max_len,
        "note": "Question embeddings aligned to samples order; packed as [B,1,H].",
    }
    return packed, meta


def _save_embed_cache(path: str, embeds: torch.Tensor, meta: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        # Store fp16 on CPU to save space; expand to fp32 later if needed.
        "embeds": embeds.to(dtype=torch.float16, device="cpu"),
    }
    torch.save(payload, path)


def _load_embed_cache(path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    embeds = payload["embeds"]
    meta = payload.get("meta", {})
    if not isinstance(embeds, torch.Tensor):
        raise ValueError("Embed cache missing 'embeds' tensor.")
    return embeds.float(), meta


def run_overflow_pipeline(
    *,
    samples_jsonl: str,
    out_jsonl: str,
    model_name_or_path: str,
    model_type: str,
    retriever_name_or_path: Optional[str],
    mode: str = "both",  # baseline | xrag | both
    embed_max_len: int = 512,
    retrieval_embed_length: int = 1,
    n_shot: int = 0,
    chat_format: str = "mistral",
    max_new_tokens: int = 32,
    only_baseline_correct: bool = False,
    embed_cache_path: Optional[str] = None,
    recompute_embeds: bool = False,
    device: str = "cuda:0",
) -> None:
    """
    Main pipeline entry.
    If mode includes xrag, we need retrieval embeddings. We will load from cache if available,
    otherwise compute them (requires retriever).
    """
    
    if mode not in {"baseline", "xrag", "both"}:
        raise ValueError(f"Unknown mode={mode}")

    #_ensure_xrag_imports(xrag_dir)

    samples = _read_jsonl(samples_jsonl)
    
    # --- load tokenizer (cheap) ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side = 'left',
        add_eos_token=True, 
        use_fast=False,
    )
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- build baseline prompts (no placeholder tokens) ---
    def create_prompt_with_mistral_chat_format(messages, tokenizer):
        formatted_text = ""
        for message in messages:
            if message['role'] == 'user':
                formatted_text += "[INST] " + message['content'] + " [/INST]"
            elif message['role'] == 'assistant':
                formatted_text += message['content'] + tokenizer.eos_token
            else:
                raise ValueError("Only 'user' and 'assistant' roles are supported.")
        return formatted_text
    
    chat_format = create_prompt_with_mistral_chat_format if chat_format == "mistral" else None
    
    baseline_prompts, _baseline_backgrounds = prepare_prompts(
        dev_data=None,
        test_data=samples,
        task_type="open_qa",
        tokenizer=tokenizer,
        n_shot=n_shot,
        use_rag=True,
        retrieval_embed_length=0,
        chat_format=chat_format,
    )

    # --- build xRAG prompts + retrieval embeds if needed ---
    xrag_prompts: Optional[List[str]] = None
    retrieval_embeds_batched: Optional[torch.Tensor] = None
    embed_meta: Optional[Dict[str, Any]] = None

    if mode in {"xrag", "both"}:
        if retriever_name_or_path is None:
            raise ValueError("xrag mode requested but retriever_name_or_path is None")

        xrag_prompts, backgrounds = prepare_prompts(
            dev_data=None,
            test_data=samples,
            task_type="open_qa",
            tokenizer=tokenizer,
            n_shot=n_shot,
            use_rag=True,
            retrieval_embed_length=retrieval_embed_length,
            chat_format=chat_format,
        )

        # Decide cache path
        cache_path = embed_cache_path
        if cache_path is None:
            # default next to output file
            out_dir = str(Path(out_jsonl).parent)
            cache_path = os.path.join(out_dir, "embeds", "background_embeds.pt")

        # Load or compute
        if (not recompute_embeds) and os.path.exists(cache_path):
            retrieval_embeds_batched, embed_meta = _load_embed_cache(cache_path)
            # Very light sanity check
            if retrieval_embeds_batched.shape[0] != len(samples):
                raise ValueError(
                    f"Embed cache B mismatch: {retrieval_embeds_batched.shape[0]} vs {len(samples)}. "
                    f"Cache={cache_path}"
                )
        else:
            # Compute with retriever, then free retriever before LLM load
            print("Loading retriever...")
            retriever, retr_tok = _load_retriever(retriever_name_or_path)
            retrieval_embeds_batched, embed_meta = _pack_embeds(
                samples=samples,
                backgrounds=backgrounds,
                embed_max_len=embed_max_len,
                embed_batch_size = 16,
                embed_device=device,
                retriever=retriever,
                retr_tok=retr_tok,
            )
            _save_embed_cache(cache_path, retrieval_embeds_batched, embed_meta)

            # Free retriever from GPU/VRAM before loading LLM
            del retriever, retr_tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- load LLM (after retriever is freed, if applicable) ---
    print("Loading LLM...")
    if model_type == 'mistral':
        llm = XMistralForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        llm = XMixtralForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        
    llm.eval()

    assert XRAG_TOKEN in tokenizer.get_vocab()
    llm.set_xrag_token_id(tokenizer.convert_tokens_to_ids(XRAG_TOKEN))
    
    run_eval.tokenizer = tokenizer
        
    # --- generate ---
    print(baseline_prompts[:2])
    baseline_outs: List[str] = []
    if mode in {"baseline", "both"}:
        baseline_outs = llm_for_open_generation(
            llm=llm,
            llm_tokenizer=tokenizer,
            prompts=baseline_prompts,
            retrieval_embeds=None,   # None for non-xRAG modes
            batch_size=4,
            enable_progress_bar=True,
        )

    xrag_outs: List[str] = []
    xrag_metrics: List[Dict[str, Any]] = []
    if mode in {"xrag", "both"}:
        assert xrag_prompts is not None and retrieval_embeds_batched is not None
        xrag_outs = llm_for_open_generation(
            llm=llm,
            llm_tokenizer=tokenizer,
            prompts=xrag_prompts,
            retrieval_embeds=retrieval_embeds_batched,   # None for non-xRAG modes
            batch_size=4,
            enable_progress_bar=True,
        )

    answers = []
    for s in samples:
        ans_list = s.get("answer") or s.get("answers") or []
        if isinstance(ans_list, str):
            ans_list = [ans_list]
        answers.append(ans_list)

    baseline_score_per_sample = None
    xrag_score_per_sample = None

    if baseline_outs:
        _, baseline_score_per_sample = get_substring_match_score(baseline_outs, answers)

    if xrag_outs:
        _, xrag_score_per_sample = get_substring_match_score(xrag_outs, answers)

    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        ans_list = s.get("answer") or s.get("answers") or []
        if isinstance(ans_list, str):
            ans_list = [ans_list]

        baseline_pred = baseline_outs[i] if baseline_outs else ""
        baseline_correct = float(baseline_score_per_sample[i]) if baseline_score_per_sample is not None else 0.0

        xrag_pred = xrag_outs[i] if xrag_outs else ""
        xrag_correct = float(xrag_score_per_sample[i]) if xrag_score_per_sample is not None else 0.0

        row: Dict[str, Any] = {
            "sample_idx": i,
            "id": s.get("id"),
            "question": s.get("question"),
            "answer": ans_list,
            "baseline_pred": baseline_pred,
            "baseline_substring_match": baseline_correct,
        }

        if mode in {"xrag", "both"}:
            row.update(
                {
                    "xrag_pred": xrag_pred,
                    "xrag_substring_match": xrag_correct,
                    "overflow_label": int((baseline_correct == 1.0) and (xrag_correct == 0.0)),
                    "xrag_metrics": xrag_metrics[i] if i < len(xrag_metrics) else {},
                }
            )

        if embed_meta is not None:
            row["embed_cache_meta"] = {
                "path": embed_cache_path,
                "B": embed_meta.get("B"),
                "maxN": embed_meta.get("maxN"),
                "H": embed_meta.get("H"),
            }

        rows.append(row)

    if only_baseline_correct and baseline_outs:
        rows = [r for r in rows if r["baseline_substring_match"] == 1.0]

    _write_jsonl(out_jsonl, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--retriever_name_or_path", type=str, default=None)
    ap.add_argument("--mode", type=str, default="both", choices=["baseline", "xrag", "both"])
    ap.add_argument("--xrag_dir", type=str, default=None)

    ap.add_argument("--embed_cache_path", type=str, default=None, help="Path to .pt cache for background embeddings")
    ap.add_argument("--recompute_embeds", action="store_true")

    ap.add_argument("--embed_max_len", type=int, default=512)
    ap.add_argument("--retrieval_embed_length", type=int, default=1)
    ap.add_argument("--n_shot", type=int, default=0)
    ap.add_argument("--chat_format", type=str, default="mistral")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--only_baseline_correct", action="store_true")
    ap.add_argument("--device", type=str, default="auto")

    args = ap.parse_args()
    run_overflow_pipeline(
        samples_jsonl=args.samples_jsonl,
        out_jsonl=args.out_jsonl,
        model_name_or_path=args.model_name_or_path,
        retriever_name_or_path=args.retriever_name_or_path,
        mode=args.mode,
        xrag_dir=args.xrag_dir,
        embed_cache_path=args.embed_cache_path,
        recompute_embeds=args.recompute_embeds,
        embed_max_len=args.embed_max_len,
        retrieval_embed_length=args.retrieval_embed_length,
        n_shot=args.n_shot,
        chat_format=args.chat_format,
        max_new_tokens=args.max_new_tokens,
        only_baseline_correct=args.only_baseline_correct,
        device=arg.device,
    )


if __name__ == "__main__":
    main()
