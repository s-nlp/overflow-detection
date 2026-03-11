"""
Build canonical samples JSONL for xRAG-style pipelines.

Supported datasets:
- squad_v2: downloads via datasets
- hotpotqa: downloads via datasets (hotpot_qa, distractor) and extracts supporting-fact windows
- triviaqa: reads xRAG eval-style jsonl files from a local data root

Outputs a JSONL where each line is a canonical sample:
{
  "id": ...,
  "question": ...,
  "answer": [...],
  "background": [...],
  "task_type": "open_qa"
}
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from rich.progress import track

from data_utils import write_jsonl

def build_squad_v2(split: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("squad_v2")
    data = ds[split]
    out: List[Dict[str, Any]] = []
    for ex in track(data, description="[bold]building squad_v2 samples..."):
        # SQuAD v2: skip unanswerable (answers["text"] empty)
        ans_texts = ex.get("answers", {}).get("text", []) or []
        if len(ans_texts) == 0:
            continue
        out.append({
            "id": ex.get("id"),
            "question": ex.get("question", "").strip(),
            "answer": [ans_texts[0].strip()],
            "background": [ex.get("context", "").strip()],
            "task_type": "open_qa",
        })
    if max_samples is not None:
        out = out[:max_samples]
    return out

def _sent_window(sentences: List[str], sent_idx: int, window: int) -> List[str]:
    s = max(0, sent_idx - window)
    e = min(len(sentences), sent_idx + window + 1)
    return [sentences[i] for i in range(s, e)]

def extract_background_hotpot(facts, context, noise_left: int = 1, noise_right: int = 1) -> str:
    background = []
    for title, id_sentence in zip(facts["title"], facts["sent_id"]):
        # find article index (assumes title present)
        article_index = context["title"].index(title)
        left_border = max(0, id_sentence - noise_left)
        # +1 because Python slice end is exclusive
        right_border = min(id_sentence + noise_right, len(context["sentences"][article_index]) - 1) + 1
        background.extend(context["sentences"][article_index][left_border:right_border])
    # dedupe while preserving first encounter order (same as original)
    deduped = list(dict.fromkeys(background))
    return " ".join(deduped)


def prepare_hotpot_dataset(dataset_split, noise_left: int = 1, noise_right: int = 1):
    """
    dataset_split: the HuggingFace Dataset split (not a dict).
    Keeps same output fields as your old script (gold_answer, single-string background in a list).
    """
    result_dataset = []
    for sample in track(dataset_split, description="[bold sea_green3]building hotpotqa samples…"):
        sample_id = sample["id"]
        question = sample["question"]
        answer = sample["answer"]
        background = extract_background_hotpot(
            facts=sample["supporting_facts"],
            context=sample["context"],
            noise_left=noise_left,
            noise_right=noise_right,
        )
        result_dataset.append(
            {
                "id": sample_id,
                "question": question.strip(),
                "background": [background.strip()],
                "gold_answer": answer,
                "task_type": "open_qa",
            }
        )
    return result_dataset


def build_hotpotqa(split: str, hotpot_window: int = 1, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads hotpotqa/distractor, uses prepare_hotpot_dataset (old-format),
    then converts to canonical format:
      - answer -> list
      - background -> list (already single-string list)
    """
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    split_ds = ds[split]

    # prepare_hotpot_dataset expects a split object; it returns old-format rows
    old_rows = prepare_hotpot_dataset(split_ds, noise_left=hotpot_window, noise_right=hotpot_window)

    # convert to canonical
    out: List[Dict[str, Any]] = []
    for r in track(old_rows, description="[bold]converting hotpot rows to canonical..."):
        ans = r.get("gold_answer", None)
        # ensure answer is a list
        if isinstance(ans, list):
            answer_list = [a for a in ans if a is not None]
        elif ans is None:
            answer_list = []
        else:
            answer_list = [ans]
        bg = r.get("background", [])
        # background should already be a list with a single long string: keep as-is
        out.append({
            "id": r.get("id"),
            "question": r.get("question", "").strip(),
            "answer": answer_list,
            "background": bg if isinstance(bg, list) else [bg],
            "task_type": r.get("task_type", "open_qa"),
        })

    if max_samples is not None:
        out = out[:max_samples]
    return out


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    import json
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_triviaqa_xrag_style(
    data_name: str,
    data_root: str,
    retrieval_prefix: str = "colbertv2",
    retrieval_topk: Optional[List[int]] = None,
    tf_idf_topk: int = 0,
    retriever_name_or_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Mirrors your notebook snippet:
    - loads /eval/{data_name}/test.jsonl
    - if retrieval_topk provided, injects backgrounds from retrieval jsonl
    - optional TF-IDF keyword extraction (requires xRAG keyword function)
    - optional "passage: " prefix for ColBERTv2
    """
    dev_data = None

    test_path = os.path.join(data_root, "eval", data_name, "test.jsonl")
    test_data = _read_jsonl(test_path)

    # normalize field names into canonical
    # many xRAG eval jsonls already have id/question/answer/background etc; we map if needed
    for ex in test_data:
        if "answer" not in ex and "answers" in ex:
            # sometimes stored as list under answers
            ex["answer"] = ex["answers"]

    if retrieval_topk is not None and len(retrieval_topk) > 0:
        # retrieval file
        ret_path = os.path.join(
            data_root, "eval", data_name, "retrieval", retrieval_prefix, "test.jsonl"
        )
        test_retrieval = _read_jsonl(ret_path)
        assert len(test_retrieval) == len(test_data), "retrieval and test length mismatch"

        # retrieval_topk are 1-indexed in your snippet usage; convert to 0-index here
        ranks = [r - 1 for r in retrieval_topk]
        for i in range(len(test_data)):
            test_data[i]["background"] = [test_retrieval[i]["topk"][rk]["text"] for rk in ranks]

        if tf_idf_topk and tf_idf_topk > 0:
            # import keyword extraction from xRAG to avoid duplication (same as your notebook)
            from xRAG.src.utils import keyword_extraction_with_tfidf
            documents = [x["background"][0] for x in test_data]
            keywords = keyword_extraction_with_tfidf(documents, topk=tf_idf_topk)
            for i in range(len(test_data)):
                test_data[i]["background"] = [keywords[i]]

        if retriever_name_or_path is not None and retriever_name_or_path.lower() == "colbertv2":
            for i in range(len(test_data)):
                test_data[i]["background"] = ["passage: " + x for x in test_data[i]["background"]]

    # canonicalize
    out: List[Dict[str, Any]] = []
    for ex in test_data[: (max_samples if max_samples is not None else len(test_data))]:
        _id = ex.get("id", ex.get("_id", ex.get("qid", "")))
        q = ex.get("question", ex.get("query", ""))
        ans = ex.get("answer", ex.get("answers", ex.get("gold_answer", "")))
        bg = ex.get("background", ex.get("contexts", []))
        # ensure list types
        if not isinstance(ans, list):
            ans = [ans] if ans is not None else []
        if not isinstance(bg, list):
            bg = [bg] if bg is not None else []
        out.append({
            "id": _id,
            "question": q,
            "answer": ans,
            "background": bg,
            "task_type": ex.get("task_type", "open_qa"),
        })

    return dev_data, out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["squad_v2", "hotpotqa", "triviaqa"])
    ap.add_argument("--split", default="validation", help="For squad_v2/hotpotqa: train/validation/test")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_samples", type=int, default=None)

    # hotpot-only
    ap.add_argument("--hotpot_window", type=int, default=1)

    # triviaqa-only (xRAG eval layout)
    ap.add_argument("--data_root", type=str, default="/app/xRAG/data")
    ap.add_argument("--retrieval_prefix", type=str, default="colbertv2")
    ap.add_argument("--retrieval_topk", type=int, nargs="*", default=None, help="1-indexed ranks, e.g., 1 2 3")
    ap.add_argument("--tf_idf_topk", type=int, default=0)
    ap.add_argument("--retriever_name_or_path", type=str, default=None)

    args = ap.parse_args()
    if args.data == "squad_v2":
        rows = build_squad_v2(args.split, args.max_samples)
    elif args.data == "hotpotqa":
        rows = build_hotpotqa(args.split, args.hotpot_window, args.max_samples)
    else:
        _, rows = build_triviaqa_xrag_style(
            data_name="triviaqa",
            data_root=args.data_root,
            retrieval_prefix=args.retrieval_prefix,
            retrieval_topk=args.retrieval_topk,
            tf_idf_topk=args.tf_idf_topk,
            retriever_name_or_path=args.retriever_name_or_path,
            max_samples=args.max_samples,
        )

    # write_jsonl(path, rows) — ensure order matches your data_utils.write_jsonl(out_path, rows)
    write_jsonl(args.out_jsonl, rows)
    print(f"Wrote {len(rows)} samples -> {args.out_jsonl}")

if __name__ == "__main__":
    main()