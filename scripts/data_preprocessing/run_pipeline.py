"""
Single entry point:
- optionally builds samples.jsonl (SQuAD v2 / HotpotQA / TriviaQA xRAG-style)
- runs notebook-aligned overflow pipeline (baseline + optional xRAG + metrics)

Typical (100 samples quick example with trivia dataset on Mistral xRAG model):
  CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --data triviaqa --out_dir runs/trivia_7b --mode both --only_baseline_correct --model_name_or_path /app/models/xrag-7b --model_type 'mistral' --retriever_name_or_path /app/models/xrag-embed --device "cuda:0"
Or (MoE xRAG model):
  CUDA_VISIBLE_DEVICES=0,1 python run_pipeline.py --data triviaqa --out_dir runs/trivia_moe --mode both --only_baseline_correct --model_name_or_path /app/models/xrag-moe --model_type 'mixtral' --retriever_name_or_path /app/models/xrag-embed --device "cuda:0"

python -c "import os; print(os.path.isdir('/app/xlong/scripts/xRAG'))"

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from data_utils import write_jsonl
from overflow_pipeline_xrag import run_overflow_pipeline  # type: ignore

import os, sys

def main() -> None:
    ap = argparse.ArgumentParser()

    # Either provide samples_jsonl directly, or build from dataset spec:
    ap.add_argument("--samples_jsonl", type=str, default=None)

    ap.add_argument("--data", type=str, default=None, choices=["squad_v2", "hotpotqa", "triviaqa"])
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--data_root", type=str, default="/app/xRAG/data")

    # Output
    ap.add_argument("--out_dir", type=str, required=True)

    # Models
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument(
        "--model_type",
        type=str,
        default="mistral",
        choices=["mistral", "mixtral"],
        help="Which XRAG model architecture to load"
    )
    ap.add_argument("--retriever_name_or_path", type=str, default=None)
    ap.add_argument("--mode", type=str, default="both", choices=["baseline", "xrag", "both"])
    ap.add_argument("--device", type=str, default="auto")

    # Embedding cache
    ap.add_argument("--embed_cache_path", type=str, default=None)
    ap.add_argument("--recompute_embeds", action="store_true")

    # Prompt / gen
    ap.add_argument("--n_shot", type=int, default=0)
    ap.add_argument("--chat_format", type=str, default="mistral")
    ap.add_argument("--retrieval_embed_length", type=int, default=1)
    ap.add_argument("--embed_max_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--hotpot_window", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=None)

    ap.add_argument("--only_baseline_correct", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve samples.jsonl
    samples_path = args.samples_jsonl
    if samples_path is None:
        if args.data is None:
            raise SystemExit("Provide --samples_jsonl or specify --data to build samples.")
        samples_path = str(out_dir / "samples.jsonl")
        # build samples
        from build_samples import (
            build_squad_v2, build_hotpotqa, build_triviaqa_xrag_style
        )
        from data_utils import write_jsonl

        if args.data == "squad_v2":
            rows = build_squad_v2(args.split, args.max_samples)
        elif args.data == "hotpotqa":
            rows = build_hotpotqa(args.split, args.hotpot_window, args.max_samples)
        else:
            _, rows = build_triviaqa_xrag_style(
                data_name="triviaqa",
                data_root=args.data_root,
                retrieval_prefix="colbertv2",
                retrieval_topk=[1],
                tf_idf_topk=0,
                retriever_name_or_path=args.retriever_name_or_path,
                max_samples=args.max_samples,
            )
        write_jsonl(samples_path, rows)
        print(f"[run_pipeline] wrote {len(rows)} samples -> {samples_path}")

    out_jsonl = str(out_dir / "results.jsonl")
    run_overflow_pipeline(
        samples_jsonl=samples_path,
        out_jsonl=out_jsonl,
        model_name_or_path=args.model_name_or_path,
        model_type=args.model_type,
        retriever_name_or_path=args.retriever_name_or_path,
        mode=args.mode,
        embed_cache_path=args.embed_cache_path,
        recompute_embeds=args.recompute_embeds,
        embed_max_len=args.embed_max_len,
        retrieval_embed_length=args.retrieval_embed_length,
        n_shot=args.n_shot,
        chat_format=args.chat_format,
        max_new_tokens=args.max_new_tokens,
        only_baseline_correct=args.only_baseline_correct,
        device=args.device,
    )

    print(f"[done] wrote: {out_jsonl}")


if __name__ == "__main__":
    main()
