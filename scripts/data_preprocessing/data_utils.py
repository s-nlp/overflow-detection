"""
Small JSONL helpers + canonical sample validation.
Canonical sample format:
{
  "id": str|int,
  "question": str,
  "answer": List[str],
  "background": List[str],
  "task_type": "open_qa"
}
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def coerce_to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def normalize_sample(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    if "answer" in out:
        out["answer"] = coerce_to_list(out["answer"])
    elif "gold_answer" in out:
        out["answer"] = coerce_to_list(out["gold_answer"])
    else:
        out["answer"] = []
    out["background"] = coerce_to_list(out.get("background", []))
    out.setdefault("task_type", "open_qa")
    # required fields sanity
    if "id" not in out:
        raise ValueError("Sample missing 'id'")
    if "question" not in out:
        raise ValueError(f"Sample {out.get('id')} missing 'question'")
    return out
