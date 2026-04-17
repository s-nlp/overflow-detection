#!/usr/bin/env python3
"""
Example:
python split_probe_dataset.py \
  --input_dir /app/overflow-detection/scripts/data_preprocessing/runs/hotpotqa_7b/probe \
  --out_dir /app/overflow-detection/scripts/data_preprocessing/runs/split_hotpotqa_7b/probe \
  --test_size 0.2 \
  --random_seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from sklearn.model_selection import train_test_split


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_rows(rows: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, Any]]:
    return [rows[i] for i in indices]


def select_vectors(vectors: Dict[str, Any], indices: List[int]) -> Dict[str, Any]:
    idx = torch.tensor(indices, dtype=torch.long)
    out: Dict[str, Any] = {}
    for k, v in vectors.items():
        if k == "ids":
            out[k] = [v[i] for i in indices]
        elif isinstance(v, torch.Tensor):
            if v.ndim == 0:
                out[k] = v
            else:
                out[k] = v.index_select(0, idx)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing features.jsonl and vectors.pt")
    ap.add_argument("--out_dir", required=True, help="Output directory for train/test split")
    ap.add_argument("--test_size", type=float, default=0.2, help="Fraction for test split")
    ap.add_argument("--random_seed", type=int, default=42, help="Random seed")
    ap.add_argument("--no_stratify", action="store_true", help="Disable stratification by labels")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    features_path = input_dir / "features.jsonl"
    vectors_path = input_dir / "vectors.pt"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing {vectors_path}")

    rows = read_jsonl(features_path)
    vectors = torch.load(vectors_path, map_location="cpu")

    if not isinstance(vectors, dict):
        raise ValueError("vectors.pt must contain a dict")
    if "ids" not in vectors or "labels" not in vectors:
        raise ValueError("vectors.pt must contain 'ids' and 'labels'")

    n_rows = len(rows)
    n_vecs = len(vectors["ids"])
    if n_rows != n_vecs:
        raise ValueError(f"features/vectors size mismatch: {n_rows} vs {n_vecs}")

    labels = vectors["labels"]
    if not isinstance(labels, torch.Tensor):
        raise ValueError("'labels' must be a torch.Tensor")
    y = labels.cpu().numpy()

    all_indices = list(range(n_rows))
    stratify = None if args.no_stratify else y

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify,
    )

    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)

    train_rows = select_rows(rows, train_idx)
    test_rows = select_rows(rows, test_idx)

    train_vectors = select_vectors(vectors, train_idx)
    test_vectors = select_vectors(vectors, test_idx)

    write_jsonl(train_dir / "features.jsonl", train_rows)
    write_jsonl(test_dir / "features.jsonl", test_rows)
    torch.save(train_vectors, train_dir / "vectors.pt")
    torch.save(test_vectors, test_dir / "vectors.pt")

    def label_rate(lbls: torch.Tensor) -> float:
        if lbls.numel() == 0:
            return 0.0
        return float(lbls.float().mean().item())

    meta = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "stratified": not args.no_stratify,
        "n_total": n_rows,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_positive_rate": label_rate(train_vectors["labels"]),
        "test_positive_rate": label_rate(test_vectors["labels"]),
        "vector_keys": sorted([k for k, v in vectors.items() if isinstance(v, torch.Tensor) or k == "ids"]),
    }

    with (out_dir / "split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote train features -> {train_dir / 'features.jsonl'}")
    print(f"Wrote train vectors  -> {train_dir / 'vectors.pt'}")
    print(f"Wrote test features  -> {test_dir / 'features.jsonl'}")
    print(f"Wrote test vectors   -> {test_dir / 'vectors.pt'}")
    print(f"Wrote split meta     -> {out_dir / 'split_meta.json'}")
    print(f"Train/Test sizes: {len(train_idx)}/{len(test_idx)}")
    print(f"Train/Test positive rate: {meta['train_positive_rate']:.3f}/{meta['test_positive_rate']:.3f}")


if __name__ == "__main__":
    main()
