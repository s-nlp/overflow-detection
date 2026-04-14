#!/usr/bin/env python3
"""
python evaluate_best_probe.py \
  --data_path /app/overflow-detection/scripts/data_preprocessing/runs/hotpotqa_moe/probe/vectors.pt \
  --probe_prefix /app/overflow-detection/scripts/data_preprocessing/runs/merged2_moe/results/best_probe_with_query_probing \
  --device 0 \
  --output_json /app/overflow-detection/scripts/data_preprocessing/runs/hotpotqa_moe/results/hotpot_test_metrics.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from data_loader import load_probing_data
from models import LinearProbe, LinearProbeTorch, MLPProbeTorch, MLPSCLProbeTorch


def build_X(data, xrag_combo, query_combo):
    parts = []
    if xrag_combo:
        parts.append(np.hstack([data[feat].numpy() for feat in xrag_combo]))
    if query_combo:
        parts.append(np.hstack([data[feat].numpy() for feat in query_combo]))
    if not parts:
        raise ValueError("At least one of xrag_combo or query_combo must be non-empty")
    return np.hstack(parts)


def load_metadata(probe_prefix: Path):
    meta_path = Path(str(probe_prefix) + "_metadata.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def make_probe(classifier_name, config):
    if classifier_name == "LinearProbeTorch":
        return LinearProbeTorch(**config)
    if classifier_name == "MLPProbeTorch":
        return MLPProbeTorch(**config)
    if classifier_name == "MLPSCLProbeTorch":
        return MLPSCLProbeTorch(**config)
    if classifier_name == "LinearProbe":
        return LinearProbe(**config)
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def materialize_torch_probe(probe, input_dim):
    for name in ["_build_model", "build_model", "_init_model", "init_model"]:
        if hasattr(probe, name):
            getattr(probe, name)(input_dim)
            return
    raise RuntimeError(
        "Torch probe wrapper has no known builder method. "
        "Add one of: _build_model(input_dim), build_model(input_dim), _init_model(input_dim), init_model(input_dim)"
    )


def load_saved_probe(probe_prefix: Path, metadata, input_dim, device):
    classifier = metadata["classifier"]
    config = metadata["config"]
    artifact_type = metadata["artifact_type"]

    model_pkl = Path(str(probe_prefix) + "_model.pkl")
    scaler_pkl = Path(str(probe_prefix) + "_scaler.pkl")
    weights_pt = Path(str(probe_prefix) + "_weights.pt")

    if artifact_type == "pickle_model":
        with open(model_pkl, "rb") as f:
            probe = pickle.load(f)
        scaler = None
        if scaler_pkl.exists():
            with open(scaler_pkl, "rb") as f:
                scaler = pickle.load(f)
        return probe, scaler

    if artifact_type in {"torch_model_state_dict", "torch_state_dict"}:
        probe = make_probe(classifier, config)
        materialize_torch_probe(probe, input_dim)

        state = torch.load(weights_pt, map_location=device)
        if hasattr(probe, "model"):
            probe.model.load_state_dict(state)
            probe.model.to(device)
            probe.model.eval()
        else:
            probe.load_state_dict(state)
            probe.to(device)
            probe.eval()
        return probe, None

    raise ValueError(f"Unknown artifact_type: {artifact_type}")


def predict_with_probe(probe, scaler, X):
    if scaler is not None:
        X_eval = scaler.transform(X)
        y_proba = probe.predict_proba(X_eval)
        y_pred = probe.predict(X_eval)
        return y_pred, y_proba

    y_proba = probe.predict_proba(X)
    y_pred = probe.predict(X)
    return y_pred, y_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Path to Hotpot probe vectors.pt")
    ap.add_argument("--probe_prefix", required=True, help="Saved best probe prefix")
    ap.add_argument("--device", default="cpu", help="cpu or CUDA id, e.g. 0")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"

    # load Hotpot probe data
    data = load_probing_data(args.data_path)

    # load saved best probe metadata
    probe_prefix = Path(args.probe_prefix)
    metadata = load_metadata(probe_prefix)

    xrag_features = tuple(metadata.get("xrag_features", []))
    query_features = tuple(metadata.get("query_features", []))

    # verify required keys exist in Hotpot vectors
    missing = [k for k in list(xrag_features) + list(query_features) if k not in data]
    if missing:
        raise ValueError(f"Hotpot data is missing required feature keys for this probe: {missing}")

    X = build_X(data, xrag_features, query_features)
    y = data["labels"].numpy()

    probe, scaler = load_saved_probe(
        probe_prefix=probe_prefix,
        metadata=metadata,
        input_dim=X.shape[1],
        device=device,
    )

    y_pred, y_proba = predict_with_probe(probe, scaler, X)

    result = {
        "probe_prefix": str(probe_prefix),
        "test_data_path": args.data_path,
        "classifier": metadata["classifier"],
        "xrag_features": list(xrag_features),
        "query_features": list(query_features),
        "n_samples": int(len(y)),
        "overflow_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, y_proba[:, 1])),
        "pr_auc": float(average_precision_score(y, y_proba[:, 1])),
        "f1": float(f1_score(y, y_pred)),
        "accuracy": float(accuracy_score(y, y_pred)),
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()