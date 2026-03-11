"""
Run probing experiments for overflow detection.

Runs two experiment blocks:
1. With-query: xRAG+query feature combinations (Setting 1: preproj/postproj; Setting 2: mid/last)
   with 4 classifiers (LinearProbeTorch, MLPProbeTorch, MLPSCLProbeTorch, LinearProbe).
2. No-query: xRAG-only features, 3 classifiers (LinearProbeTorch, MLPProbeTorch, LinearProbe).

Use --with_query_only or --no_query_only to run only one block.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_probing_data
from models import LinearProbe, LinearProbeTorch, MLPProbeTorch, MLPSCLProbeTorch
from utils import set_seed

warnings.filterwarnings("ignore")

# --- With-query: fixed xRAG+query combinations ---
SETTING1_COMBOS = [
    (("postproj",), ("postproj_q",)),
    (("preproj",), ("preproj_q",)),
    (("preproj", "postproj"), ("preproj_q", "postproj_q")),
]
SETTING2_COMBOS = [
    (("mid",), ("mid_q",)),
    (("last",), ("last_q",)),
    (("mid", "last"), ("mid_q", "last_q")),
]

WITH_QUERY_TEST_CONFIGS = [
    (
        "linear_torch",
        "LinearProbeTorch",
        {
            "l2_lambda": 500.0,
            "l1_lambda": 100.0,
            "epochs": 150,
            "batch_size": 256,
            "normalize": True,
            "verbose": False,
            "early_stopping_patience": 20,
            "random_state": 42,
        },
    ),
    (
        "mlp_torch",
        "MLPProbeTorch",
        {
            "l2_lambda": 500.0,
            "l1_lambda": 100.0,
            "epochs": 50,
            "batch_size": 256,
            "hidden_dim": 1024,
            "normalize": True,
            "verbose": False,
            "early_stopping_patience": 20,
            "random_state": 42,
        },
    ),
    (
        "mlp_scl_torch",
        "MLPSCLProbeTorch",
        {
            "l2_lambda": 500.0,
            "l1_lambda": 100.0,
            "epochs": 50,
            "batch_size": 256,
            "hidden_dim": 1024,
            "normalize": True,
            "verbose": False,
            "early_stopping_patience": 20,
            "random_state": 42,
            "contrastive_weight": 0.3,
            "contrastive_temperature": 0.07,
        },
    ),
    ("linear_sklearn", "LinearProbe", {"C": 0.00001}),
]

NO_QUERY_TEST_CONFIGS = [
    (
        "linear_torch",
        "LinearProbeTorch",
        {
            "l2_lambda": 500.0,
            "l1_lambda": 100.0,
            "epochs": 150,
            "batch_size": 256,
            "normalize": True,
            "verbose": False,
            "early_stopping_patience": 20,
            "random_state": 42,
        },
    ),
    (
        "mlp_torch",
        "MLPProbeTorch",
        {
            "l2_lambda": 500.0,
            "l1_lambda": 100.0,
            "epochs": 50,
            "batch_size": 256,
            "hidden_dim": 1024,
            "normalize": True,
            "verbose": False,
            "early_stopping_patience": 20,
            "random_state": 42,
        },
    ),
    ("linear_sklearn", "LinearProbe", {"C": 0.00001}),
]

NO_QUERY_XRAG_COMBOS = [
    ("preproj",),
    ("postproj",),
    ("preproj", "postproj"),
    ("mid",),
    ("last",),
    ("mid", "last"),
]


def _print_results(results):
    """Print formatted results table."""
    print("\n" + "-" * 80)
    print(f"{'Experiment':<40} {'AUC':<8} {'PR-AUC':<8} {'F1':<8}")
    print("-" * 80)
    sorted_results = sorted(
        results.items(), key=lambda x: x[1].get("auc", 0), reverse=True
    )
    for name, result in sorted_results:
        auc = result.get("auc", 0)
        pr_auc = result.get("pr_auc", 0)
        f1 = result.get("f1", 0)
        print(f"{name:<40} {auc:<8.3f} {pr_auc:<8.3f} {f1:<8.3f}")
    print("-" * 80)
    print(f"\nTotal experiments: {len(results)}")
    if results:
        print(f"Best AUC: {max(r.get('auc', 0) for r in results.values()):.3f}")


def _make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def build_X(data, xrag_combo, query_combo):
    """Build feature matrix from xRAG and query combos (either can be empty)."""
    parts = []
    if xrag_combo:
        parts.append(np.hstack([data[feat].numpy() for feat in xrag_combo]))
    if query_combo:
        parts.append(np.hstack([data[feat].numpy() for feat in query_combo]))
    if not parts:
        raise ValueError(
            "At least one of xrag_combo or query_combo must be non-empty"
        )
    return np.hstack(parts)


def combo_name(xrag_combo, query_combo):
    """Experiment name segment for a combination."""
    xrag_str = "+".join(xrag_combo) if xrag_combo else "none"
    query_str = "+".join(query_combo) if query_combo else "none"
    return f"{xrag_str}_with_{query_str}"


def run_one_fold(
    X_train_cv,
    X_test_cv,
    y_train_cv,
    y_test_cv,
    config_name,
    classifier_type,
    config,
):
    """Run one CV fold for a given classifier; return test and train metrics."""
    if classifier_type in ["LinearProbeTorch", "MLPProbeTorch", "MLPSCLProbeTorch"]:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_cv,
            y_train_cv,
            test_size=0.2,
            random_state=42,
            stratify=y_train_cv,
        )

    if classifier_type == "LinearProbeTorch":
        probe = LinearProbeTorch(**config)
        probe.fit(
            X_train_split,
            y_train_split,
            X_val=X_val_split,
            y_val=y_val_split,
        )
    elif classifier_type == "MLPProbeTorch":
        probe = MLPProbeTorch(**config)
        probe.fit(
            X_train_split,
            y_train_split,
            X_val=X_val_split,
            y_val=y_val_split,
        )
    elif classifier_type == "MLPSCLProbeTorch":
        probe = MLPSCLProbeTorch(**config)
        probe.fit(
            X_train_split,
            y_train_split,
            X_val=X_val_split,
            y_val=y_val_split,
        )
    elif classifier_type == "LinearProbe":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_cv)
        X_test_scaled = scaler.transform(X_test_cv)
        probe = LinearProbe(**config)
        probe.fit(X_train_scaled, y_train_cv)

    if classifier_type == "LinearProbe":
        y_test_pred = probe.predict(X_test_scaled)
        y_test_proba = probe.predict_proba(X_test_scaled)
        X_train_scaled_full = scaler.transform(X_train_cv)
        y_train_pred = probe.predict(X_train_scaled_full)
        y_train_proba = probe.predict_proba(X_train_scaled_full)
    else:
        y_test_pred = probe.predict(X_test_cv)
        y_test_proba = probe.predict_proba(X_test_cv)
        y_train_pred = probe.predict(X_train_cv)
        y_train_proba = probe.predict_proba(X_train_cv)

    test_auc = roc_auc_score(y_test_cv, y_test_proba[:, 1])
    test_pr = average_precision_score(y_test_cv, y_test_proba[:, 1])
    test_f1 = f1_score(y_test_cv, y_test_pred)
    test_acc = accuracy_score(y_test_cv, y_test_pred)
    train_auc = roc_auc_score(y_train_cv, y_train_proba[:, 1])
    train_pr = average_precision_score(y_train_cv, y_train_proba[:, 1])
    train_f1 = f1_score(y_train_cv, y_train_pred)
    train_acc = accuracy_score(y_train_cv, y_train_pred)

    return (
        (test_auc, test_pr, test_f1, test_acc),
        (train_auc, train_pr, train_f1, train_acc),
    )


def run_one_experiment_with_query(
    data, xrag_combo, query_combo, setting_prefix, probe_type, cv_folds, results
):
    """Run all classifiers for one (xrag_combo, query_combo) in with-query mode."""
    X = build_X(data, xrag_combo, query_combo)
    y = data["labels"].numpy()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    name_seg = combo_name(xrag_combo, query_combo)

    for config_name, classifier_type, config in WITH_QUERY_TEST_CONFIGS:
        exp_name = f"{setting_prefix}_{name_seg}_{config_name}"
        print(f"\n  Testing: {exp_name}")
        print(
            f"    xRAG: {xrag_combo or 'none'}, Query: {query_combo or 'none'}, Features: {X.shape[1]}, Classifier: {classifier_type}"
        )

        cv_aucs, cv_pr_aucs, cv_f1s, cv_accs = [], [], [], []
        cv_train_aucs, cv_train_pr_aucs, cv_train_f1s, cv_train_accs = (
            [],
            [],
            [],
            [],
        )

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]

            (test_metrics, train_metrics) = run_one_fold(
                X_train_cv,
                X_test_cv,
                y_train_cv,
                y_test_cv,
                config_name,
                classifier_type,
                config,
            )
            (test_auc, test_pr, test_f1, test_acc) = test_metrics
            (train_auc, train_pr, train_f1, train_acc) = train_metrics

            cv_aucs.append(test_auc)
            cv_pr_aucs.append(test_pr)
            cv_f1s.append(test_f1)
            cv_accs.append(test_acc)
            cv_train_aucs.append(train_auc)
            cv_train_pr_aucs.append(train_pr)
            cv_train_f1s.append(train_f1)
            cv_train_accs.append(train_acc)

            if fold == 0:
                print(
                    f"      Fold 1: Test AUC = {cv_aucs[-1]:.3f}, Train AUC = {cv_train_aucs[-1]:.3f}"
                )

        result = {
            "name": exp_name,
            "probe_type": probe_type,
            "classifier": classifier_type,
            "auc": float(np.mean(cv_aucs)),
            "auc_std": float(np.std(cv_aucs)),
            "auc_scores": cv_aucs,
            "pr_auc": float(np.mean(cv_pr_aucs)),
            "pr_auc_std": float(np.std(cv_pr_aucs)),
            "f1": float(np.mean(cv_f1s)),
            "f1_std": float(np.std(cv_f1s)),
            "accuracy": float(np.mean(cv_accs)),
            "accuracy_std": float(np.std(cv_accs)),
            "train_auc": float(np.mean(cv_train_aucs)),
            "train_auc_std": float(np.std(cv_train_aucs)),
            "train_pr_auc": float(np.mean(cv_train_pr_aucs)),
            "train_pr_auc_std": float(np.std(cv_train_pr_aucs)),
            "train_f1": float(np.mean(cv_train_f1s)),
            "train_f1_std": float(np.std(cv_train_f1s)),
            "train_accuracy": float(np.mean(cv_train_accs)),
            "train_accuracy_std": float(np.std(cv_train_accs)),
            "xrag_features": list(xrag_combo),
            "query_features": list(query_combo),
            "n_features": X.shape[1],
            "config": config,
            "n_folds": cv_folds,
        }
        results[exp_name] = result
        print(
            f"      Mean Test AUC: {result['auc']:.3f} ± {result['auc_std']:.3f}, Train AUC: {result['train_auc']:.3f} ± {result['train_auc_std']:.3f}"
        )


def save_setting_results(
    output_path, setting_name, setting_prefix, results_dict, experiment_name, cv_folds
):
    """Save results for a setting to JSON."""
    setting_results = {
        k: v for k, v in results_dict.items() if k.startswith(setting_prefix)
    }
    if not setting_results:
        return
    serializable = {
        k: {key: _make_serializable(val) for key, val in v.items()}
        for k, v in setting_results.items()
    }
    out_file = output_path / f"probing_results_{setting_name}_{experiment_name}.json"
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(
        f"\n✓ {setting_name} results saved to {out_file} ({len(setting_results)} experiments)"
    )


def run_with_query(data_path, output_dir, experiment_name, cv_folds):
    """Run with-query experiments (fixed xRAG+query combinations, 4 classifiers)."""
    print("=" * 80)
    print("RUNNING PROBING EXPERIMENTS (WITH QUERY)")
    print(f"CV folds: {cv_folds}")
    print(
        "\nSetting 1: preproj+postproj, preproj_q+postproj_q, preproj+preproj_q+postproj+postproj_q"
    )
    print("Setting 2: mid+mid_q, last+last_q, mid+mid_q+last+last_q")
    print(
        "\nClassifiers: LinearProbeTorch, MLPProbeTorch, MLPSCLProbeTorch, LinearProbe"
    )
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    data = load_probing_data(data_path)
    print(
        f"Loaded {len(data['ids'])} samples, overflow rate: {data['labels'].float().mean():.3f}"
    )

    results = {}

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Setting 1 (preproj/postproj)")
    print("=" * 80)
    for xrag_combo, query_combo in SETTING1_COMBOS:
        all_keys = list(xrag_combo) + list(query_combo)
        if not all(k in data for k in all_keys):
            print(f"  Skip {combo_name(xrag_combo, query_combo)}: missing keys in data")
            continue
        run_one_experiment_with_query(
            data,
            xrag_combo,
            query_combo,
            setting_prefix="setting1",
            probe_type="setting1_prepost_projection",
            cv_folds=cv_folds,
            results=results,
        )
    save_setting_results(
        output_path, "setting1", "setting1_", results, experiment_name, cv_folds
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Setting 2 (mid/last)")
    print("=" * 80)
    for xrag_combo, query_combo in SETTING2_COMBOS:
        all_keys = list(xrag_combo) + list(query_combo)
        if not all(k in data for k in all_keys):
            print(f"  Skip {combo_name(xrag_combo, query_combo)}: missing keys in data")
            continue
        run_one_experiment_with_query(
            data,
            xrag_combo,
            query_combo,
            setting_prefix="setting2",
            probe_type="setting2_extended_features",
            cv_folds=cv_folds,
            results=results,
        )
    save_setting_results(
        output_path, "setting2", "setting2_", results, experiment_name, cv_folds
    )

    print("\n" + "=" * 80)
    print("WITH-QUERY EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"Total experiments: {len(results)}")
    return results


def run_no_query(data_path, output_dir, experiment_name, cv_folds):
    """Run no-query experiments (xRAG only, 3 classifiers)."""
    print("=" * 80)
    print("RUNNING PROBING EXPERIMENTS (NO QUERY FEATURES)")
    print(f"Using {cv_folds}-fold cross-validation")
    print("Features: xRAG (context) only — no query features")
    print("=" * 80)
    print("\nClassifiers: LinearProbeTorch, MLPProbeTorch, LinearProbe")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    data = load_probing_data(data_path)
    print(f"Loaded {len(data['ids'])} samples")
    print(f"Overflow rate: {data['labels'].float().mean():.3f}")

    results = {}

    def save_setting_no_query(setting_name, setting_prefix, results_dict):
        setting_results = {
            k: v for k, v in results_dict.items() if k.startswith(setting_prefix)
        }
        if not setting_results:
            return
        serializable = {
            k: {key: _make_serializable(val) for key, val in v.items()}
            for k, v in setting_results.items()
        }
        out_file = output_path / f"probing_results_{setting_name}_{experiment_name}.json"
        with open(out_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n✓ {setting_name} results saved to {out_file}")

    print("\n" + "=" * 80)
    print("EXPERIMENTS: xRAG only (no query)")
    print("=" * 80)

    xrag_combos = [
        c for c in NO_QUERY_XRAG_COMBOS if all(feat in data for feat in c)
    ]
    print(f"Available in data: {xrag_combos}")

    if not xrag_combos:
        print("No xRAG combinations available in data.")
        return results

    y = data["labels"].numpy()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for xrag_combo in xrag_combos:
        X = np.hstack([data[feat].numpy() for feat in xrag_combo])

        for config_name, classifier_type, config in NO_QUERY_TEST_CONFIGS:
            exp_name = f"xrag_{'+'.join(xrag_combo)}_no_query_{config_name}"
            print(f"\n  Testing: {exp_name}")
            print(
                f"    xRAG: {xrag_combo}, Features: {X.shape[1]}, Classifier: {classifier_type}"
            )

            cv_aucs, cv_pr_aucs, cv_f1s, cv_accs = [], [], [], []
            cv_train_aucs, cv_train_pr_aucs, cv_train_f1s, cv_train_accs = (
                [],
                [],
                [],
                [],
            )

            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train_cv, X_test_cv = X[train_idx], X[test_idx]
                y_train_cv, y_test_cv = y[train_idx], y[test_idx]

                (test_metrics, train_metrics) = run_one_fold(
                    X_train_cv,
                    X_test_cv,
                    y_train_cv,
                    y_test_cv,
                    config_name,
                    classifier_type,
                    config,
                )
                (test_auc, test_pr, test_f1, test_acc) = test_metrics
                (train_auc, train_pr, train_f1, train_acc) = train_metrics

                cv_aucs.append(test_auc)
                cv_pr_aucs.append(test_pr)
                cv_f1s.append(test_f1)
                cv_accs.append(test_acc)
                cv_train_aucs.append(train_auc)
                cv_train_pr_aucs.append(train_pr)
                cv_train_f1s.append(train_f1)
                cv_train_accs.append(train_acc)

                if fold == 0:
                    print(
                        f"      Fold 1: Test AUC = {cv_aucs[-1]:.3f}, Train AUC = {cv_train_aucs[-1]:.3f}"
                    )

            result = {
                "name": exp_name,
                "probe_type": "xrag_no_query",
                "classifier": classifier_type,
                "auc": float(np.mean(cv_aucs)),
                "auc_std": float(np.std(cv_aucs)),
                "auc_scores": cv_aucs,
                "pr_auc": float(np.mean(cv_pr_aucs)),
                "pr_auc_std": float(np.std(cv_pr_aucs)),
                "f1": float(np.mean(cv_f1s)),
                "f1_std": float(np.std(cv_f1s)),
                "accuracy": float(np.mean(cv_accs)),
                "accuracy_std": float(np.std(cv_accs)),
                "train_auc": float(np.mean(cv_train_aucs)),
                "train_auc_std": float(np.std(cv_train_aucs)),
                "train_pr_auc": float(np.mean(cv_train_pr_aucs)),
                "train_pr_auc_std": float(np.std(cv_train_pr_aucs)),
                "train_f1": float(np.mean(cv_train_f1s)),
                "train_f1_std": float(np.std(cv_train_f1s)),
                "train_accuracy": float(np.mean(cv_train_accs)),
                "train_accuracy_std": float(np.std(cv_train_accs)),
                "xrag_features": list(xrag_combo),
                "query_features": [],
                "n_features": X.shape[1],
                "config": config,
                "n_folds": cv_folds,
            }
            results[exp_name] = result
            print(
                f"      Mean Test AUC: {result['auc']:.3f} ± {result['auc_std']:.3f}, Train AUC: {result['train_auc']:.3f} ± {result['train_auc_std']:.3f}"
            )

    _print_results(results)
    save_setting_no_query("xrag", "xrag_", results)

    # Combined JSON
    combined_path = output_path / f"probing_results_{experiment_name}_no_query_combined.json"
    serializable = {
        k: {key: _make_serializable(val) for key, val in v.items()}
        for k, v in results.items()
    }
    with open(combined_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")

    # Statistical tests (no_query only)
    _run_no_query_statistics(results, output_path, experiment_name)

    return results


def _run_no_query_statistics(results, output_path, experiment_name):
    """Paired t-tests and Friedman test for no_query experiments."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (NO QUERY)")
    print("=" * 80)

    def paired_ttest(scores_a, scores_b):
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        if len(scores_a) != len(scores_b):
            return {"p_value": 1.0, "t_statistic": 0, "mean_difference": 0}
        t_stat, p_value = stats.ttest_rel(
            scores_a, scores_b, alternative="two-sided"
        )
        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "significant_at_0.001": p_value < 0.001,
        }

    def cohens_d(scores_a, scores_b):
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        n_a, n_b = len(scores_a), len(scores_b)
        var_a, var_b = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
        pooled_std = np.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )
        d = mean_diff / pooled_std if pooled_std > 0 else 0
        abs_d = abs(d)
        interpretation = (
            "Negligible"
            if abs_d < 0.2
            else "Small"
            if abs_d < 0.5
            else "Medium"
            if abs_d < 0.8
            else "Large"
        )
        return {
            "cohens_d": d,
            "absolute_d": abs_d,
            "interpretation": interpretation,
            "mean_difference": mean_diff,
        }

    classifier_names = ["LinearProbeTorch", "MLPProbeTorch", "LinearProbe"]
    feature_combinations = {}
    for exp_name, result in results.items():
        xrag_str = "+".join(result["xrag_features"])
        classifier = result["classifier"]
        auc_scores = result["auc_scores"]
        if xrag_str not in feature_combinations:
            feature_combinations[xrag_str] = {}
        feature_combinations[xrag_str][classifier] = {
            "auc_scores": auc_scores,
            "mean_auc": result["auc"],
        }

    complete = {
        k: v
        for k, v in feature_combinations.items()
        if all(clf in v for clf in classifier_names)
    }

    if not complete:
        print("No complete feature combinations with all 3 classifiers.")
        return

    aggregated_scores = {clf: [] for clf in classifier_names}
    for combo_data in complete.values():
        for clf in classifier_names:
            aggregated_scores[clf].extend(combo_data[clf]["auc_scores"])

    comparisons = [
        ("LinearProbeTorch", "MLPProbeTorch"),
        ("LinearProbeTorch", "LinearProbe"),
        ("MLPProbeTorch", "LinearProbe"),
    ]
    comparison_results = []
    for clf_a, clf_b in comparisons:
        scores_a = aggregated_scores[clf_a]
        scores_b = aggregated_scores[clf_b]
        ttest_result = paired_ttest(scores_a, scores_b)
        effect_result = cohens_d(scores_a, scores_b)
        sig = (
            "***"
            if ttest_result["significant_at_0.001"]
            else "**"
            if ttest_result["significant_at_0.01"]
            else "*"
            if ttest_result["significant_at_0.05"]
            else "ns"
        )
        comparison_results.append({
            "comparison": f"{clf_a} vs {clf_b}",
            "mean_diff": ttest_result["mean_difference"],
            "cohens_d": effect_result["cohens_d"],
            "p_value": ttest_result["p_value"],
            "significance": sig,
        })

    matched_scores = {
        clf: [complete[k][clf]["mean_auc"] for k in complete]
        for clf in classifier_names
    }
    friedman_stat, friedman_p = stats.friedmanchisquare(
        matched_scores["LinearProbeTorch"],
        matched_scores["MLPProbeTorch"],
        matched_scores["LinearProbe"],
    )

    stats_results = {
        "overall_performance": {
            clf: {
                "mean_auc": float(np.mean(aggregated_scores[clf])),
                "std_auc": float(np.std(aggregated_scores[clf])),
                "n_scores": len(aggregated_scores[clf]),
            }
            for clf in classifier_names
        },
        "pairwise_comparisons": comparison_results,
        "friedman_test": {
            "statistic": float(friedman_stat),
            "p_value": float(friedman_p),
            "n_combinations": len(complete),
            "significant": bool(friedman_p < 0.05),
        },
        "n_feature_combinations": len(complete),
    }
    stats_path = output_path / f"probing_results_{experiment_name}_no_query_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"✓ Statistical comparison saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run probing experiments for overflow detection"
    )
    parser.add_argument(
        "--with_query_only",
        action="store_true",
        help="Run only with-query experiments (Setting 1 + 2, 4 classifiers)",
    )
    parser.add_argument(
        "--no_query_only",
        action="store_true",
        help="Run only no-query experiments (xRAG only, 3 classifiers)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to probing data .pt file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: parent of data_path)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="probing",
        help="Suffix for output JSON files",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device (e.g. 0 or 6)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if args.with_query_only and args.no_query_only:
        parser.error("Cannot use both --with_query_only and --no_query_only")
    run_both = not args.with_query_only and not args.no_query_only

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    set_seed(args.random_seed)
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(args.data_path).parent
    )

    if run_both or args.with_query_only:
        run_with_query(
            data_path=args.data_path,
            output_dir=str(output_dir),
            experiment_name=args.experiment_name,
            cv_folds=args.cv_folds,
        )
    if run_both or args.no_query_only:
        run_no_query(
            data_path=args.data_path,
            output_dir=str(output_dir),
            experiment_name=args.experiment_name,
            cv_folds=args.cv_folds,
        )


if __name__ == "__main__":
    main()
