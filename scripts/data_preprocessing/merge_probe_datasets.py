#!/usr/bin/env python3
"""
merge_probe_datasets.py

Merge multiple probe_pipeline outputs into one combined probe dataset.

Expected per-input directory structure:
  <run_dir>/
    features.jsonl
    vectors.pt

Output:
  <out_dir>/
    features.jsonl
    vectors.pt
    merge_meta.json
/app/overflow-detection/scripts/data_preprocessing/runs/hotpotqa_moe/probe \
Example:
python merge_probe_datasets.py \
  --inputs /app/overflow-detection/scripts/data_preprocessing/runs/split_squad_moe/probe/train \
           /app/overflow-detection/scripts/data_preprocessing/runs/split_trivia_moe/probe/train \
           /app/overflow-detection/scripts/data_preprocessing/runs/split_hotpotqa_moe/probe/train \
  --out_dir /app/overflow-detection/scripts/data_preprocessing/runs/split_combined_moe/probe/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


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


def infer_dataset_name(run_dir: Path) -> str:
    if run_dir.parent.name:
        return run_dir.parent.name
    return run_dir.name


def detect_vector_keys(data: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for k, v in data.items():
        if k in {"ids", "labels"}:
            continue
        if isinstance(v, torch.Tensor):
            keys.append(k)
    return keys


def validate_compatible_shapes(
    seen_shapes: Dict[str, Tuple[int, ...]],
    dataset_name: str,
    tensor_dict: Dict[str, torch.Tensor],
) -> None:
    for k, t in tensor_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        shape = tuple(t.shape)
        if len(shape) == 0:
            continue
        feature_shape = shape[1:]
        if k not in seen_shapes:
            seen_shapes[k] = feature_shape
        else:
            if seen_shapes[k] != feature_shape:
                raise ValueError(
                    f"Incompatible tensor shape for key '{k}' in dataset '{dataset_name}': "
                    f"got feature shape {feature_shape}, expected {seen_shapes[k]}"
                )


def add_metadata_to_feature_rows(
    rows: List[Dict[str, Any]],
    dataset_name: str,
    source_run: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        src_id = row.get("id")
        global_id = f"{dataset_name}::{src_id}"
        new_row = dict(row)
        new_row["dataset"] = dataset_name
        new_row["source_run"] = source_run
        new_row["source_id"] = src_id
        new_row["global_id"] = global_id
        out.append(new_row)
    return out


def merge_one_run(
    run_dir: Path,
    dataset_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if dataset_name is None:
        dataset_name = infer_dataset_name(run_dir)

    features_path = run_dir / "features.jsonl"
    vectors_path = run_dir / "vectors.pt"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.jsonl in {run_dir}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing vectors.pt in {run_dir}")

    feature_rows = read_jsonl(features_path)
    feature_rows = add_metadata_to_feature_rows(feature_rows, dataset_name, str(run_dir))

    vectors = torch.load(vectors_path, map_location="cpu")
    if not isinstance(vectors, dict):
        raise ValueError(f"vectors.pt in {run_dir} is not a dict")

    if "ids" not in vectors or "labels" not in vectors:
        raise ValueError(f"vectors.pt in {run_dir} must contain 'ids' and 'labels'")

    ids = [str(x) for x in vectors["ids"]]
    global_ids = [f"{dataset_name}::{x}" for x in ids]
    vectors["ids"] = global_ids

    summary = {
        "dataset": dataset_name,
        "run_dir": str(run_dir),
        "n_features_rows": len(feature_rows),
        "n_vectors": len(global_ids),
        "vector_keys": detect_vector_keys(vectors),
    }
    return feature_rows, vectors, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of probe run directories to merge",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Where to write merged features.jsonl / vectors.pt / merge_meta.json",
    )
    ap.add_argument(
        "--dataset_names",
        nargs="*",
        default=None,
        help="Optional explicit dataset names, same order as --inputs",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dirs = [Path(x) for x in args.inputs]
    if args.dataset_names is not None and len(args.dataset_names) not in {0, len(input_dirs)}:
        raise ValueError("--dataset_names must be omitted or have the same length as --inputs")

    all_feature_rows: List[Dict[str, Any]] = []
    merged_ids: List[str] = []
    merged_labels: List[torch.Tensor] = []
    tensor_buckets: Dict[str, List[torch.Tensor]] = {}
    seen_feature_shapes: Dict[str, Tuple[int, ...]] = {}
    per_dataset_meta: List[Dict[str, Any]] = []

    for idx, run_dir in enumerate(input_dirs):
        dataset_name = None
        if args.dataset_names:
            dataset_name = args.dataset_names[idx]

        feature_rows, vectors, summary = merge_one_run(run_dir, dataset_name)
        per_dataset_meta.append(summary)

        all_feature_rows.extend(feature_rows)

        ids = vectors["ids"]
        labels = vectors["labels"]

        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"'labels' in {run_dir} must be a torch.Tensor")

        merged_ids.extend(ids)
        merged_labels.append(labels.cpu())

        tensor_payload = {
            k: v.cpu() for k, v in vectors.items()
            if isinstance(v, torch.Tensor) and k not in {"labels"}
        }
        validate_compatible_shapes(seen_feature_shapes, summary["dataset"], tensor_payload)

        for k, v in tensor_payload.items():
            tensor_buckets.setdefault(k, []).append(v)

    merged_vectors: Dict[str, Any] = {
        "ids": merged_ids,
        "labels": torch.cat(merged_labels, dim=0) if merged_labels else torch.empty(0, dtype=torch.long),
    }

    for k, tensors in tensor_buckets.items():
        merged_vectors[k] = torch.cat(tensors, dim=0)

    write_jsonl(out_dir / "features.jsonl", all_feature_rows)
    torch.save(merged_vectors, out_dir / "vectors.pt")

    merge_meta = {
        "inputs": [str(x) for x in input_dirs],
        "datasets": [m["dataset"] for m in per_dataset_meta],
        "n_total_features_rows": len(all_feature_rows),
        "n_total_vectors": len(merged_ids),
        "n_by_dataset": {
            m["dataset"]: {
                "n_features_rows": m["n_features_rows"],
                "n_vectors": m["n_vectors"],
            }
            for m in per_dataset_meta
        },
        "vector_keys": list(tensor_buckets.keys()),
        "feature_shapes": {k: list(v) for k, v in seen_feature_shapes.items()},
    }

    with (out_dir / "merge_meta.json").open("w", encoding="utf-8") as f:
        json.dump(merge_meta, f, indent=2)

    print(f"Wrote merged features -> {out_dir / 'features.jsonl'}")
    print(f"Wrote merged vectors  -> {out_dir / 'vectors.pt'}")
    print(f"Wrote merge meta      -> {out_dir / 'merge_meta.json'}")


if __name__ == "__main__":
    main()
