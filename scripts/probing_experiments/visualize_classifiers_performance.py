#!/usr/bin/env python3
"""
Figure comparing classifier performance across datasets and feature stages.
Expects JSON results from run_probing_experiments.py (with-query block: setting1/setting2).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.titlesize"] = 14


def load_results(base_path, dataset, setting, experiment_name="probing"):
    """Load results for a specific dataset and setting."""
    file_path = Path(base_path) / dataset / f"probing_results_{setting}_{experiment_name}.json"
    with open(file_path, "r") as f:
        return json.load(f)


def get_classifier_type(exp_name):
    """Extract classifier type from experiment name."""
    if exp_name.endswith("_linear_torch"):
        return "Linear"
    elif exp_name.endswith("_mlp_torch"):
        return "MLP"
    elif exp_name.endswith("_mlp_scl_torch"):
        return "MLP (SCL)"
    elif exp_name.endswith("_linear_sklearn"):
        return "Linear (SK)"
    return None


def match_feature_combination(exp_name, xrag_features, query_features):
    """Check if experiment matches the specified feature combination."""
    if "setting1_" in exp_name:
        prefix = "setting1_"
    elif "setting2_" in exp_name:
        prefix = "setting2_"
    else:
        return False

    name = exp_name.replace(prefix, "")
    parts = name.split("_with_")
    if len(parts) != 2:
        return False

    exp_xrag, exp_query = parts

    for classifier in [
        "_linear_torch",
        "_mlp_torch",
        "_mlp_scl_torch",
        "_linear_sklearn",
    ]:
        exp_query = exp_query.replace(classifier, "")

    return exp_xrag == xrag_features and exp_query == query_features


def extract_classifier_results(
    results, xrag_features, query_features, metric="auc"
):
    """Extract results for all classifiers for a given feature combination."""
    classifier_results = {}

    for exp_name, exp_data in results.items():
        if match_feature_combination(
            exp_name, xrag_features, query_features
        ):
            classifier_type = get_classifier_type(exp_name)
            if classifier_type:
                classifier_results[classifier_type] = {
                    "mean": exp_data[metric],
                    "std": exp_data[f"{metric}_std"],
                }

    return classifier_results


def create_classifier_comparison_plot(
    base_path, datasets, output_dir="figures", experiment_name="probing"
):
    """Create figure comparing classifier performance across datasets and stages."""
    setting1_features = [
        ("Pre-Projections", "preproj", "preproj_q"),
        ("Post-Projections", "postproj", "postproj_q"),
        ("Pre+Post-Projections", "preproj+postproj", "preproj_q+postproj_q"),
    ]

    setting2_features = [
        ("Middle Hiddens", "mid", "mid_q"),
        ("Last Hiddens", "last", "last_q"),
        ("Mid+Last Hiddens", "mid+last", "mid_q+last_q"),
    ]

    dataset_labels = {
        "mistral_squad": "SQuADv2",
        "mistral_trivia": "TriviaQA",
        "mistral_hotpot": "HotpotQA",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        "Linear": "#5ED1FF",
        "MLP": "#FFAF5E",
        "MLP (SCL)": "#8238D9",
        "Linear (SK)": "#FF6B6B",
    }
    classifier_order = ["Linear", "MLP", "MLP (SCL)", "Linear (SK)"]
    metric = "auc"

    for col_idx, dataset in enumerate(datasets):
        ax1 = axes[0, col_idx]
        results1 = load_results(base_path, dataset, "setting1", experiment_name)

        feature_labels1 = []
        data_by_classifier1 = {clf: [] for clf in classifier_order}
        std_by_classifier1 = {clf: [] for clf in classifier_order}

        for label, xrag, query in setting1_features:
            feature_labels1.append(label)
            classifier_results = extract_classifier_results(
                results1, xrag, query, metric
            )
            for classifier in classifier_order:
                if classifier in classifier_results:
                    data_by_classifier1[classifier].append(
                        classifier_results[classifier]["mean"]
                    )
                    std_by_classifier1[classifier].append(
                        classifier_results[classifier]["std"]
                    )
                else:
                    data_by_classifier1[classifier].append(0)
                    std_by_classifier1[classifier].append(0)

        x1 = np.arange(len(feature_labels1))
        bar_width = 0.2

        for i, classifier in enumerate(classifier_order):
            offset = (i - 1.5) * bar_width
            ax1.bar(
                x1 + offset,
                data_by_classifier1[classifier],
                bar_width,
                yerr=std_by_classifier1[classifier],
                label=classifier if col_idx == 0 else "",
                color=colors[classifier],
                alpha=0.8,
                capsize=3,
                error_kw={"linewidth": 1},
            )

        ax1.set_xticks(x1)
        ax1.set_xticklabels(
            feature_labels1, fontsize=12, rotation=0, ha="center"
        )
        ax1.set_ylim(0.5, 0.80)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        if col_idx == 0:
            ax1.set_ylabel("ROC-AUC", fontweight="bold", fontsize=12)
            ax1.legend(loc="upper left", framealpha=0.95, fontsize=12)

        ax1.set_title(
            f"{dataset_labels.get(dataset, dataset)}",
            fontweight="bold",
            fontsize=14,
        )
        if col_idx == 0:
            ax1.text(
                -0.18,
                0.5,
                "LLM-Independent",
                transform=ax1.transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                ha="center",
                rotation=90,
            )

        ax2 = axes[1, col_idx]
        results2 = load_results(base_path, dataset, "setting2", experiment_name)

        feature_labels2 = []
        data_by_classifier2 = {clf: [] for clf in classifier_order}
        std_by_classifier2 = {clf: [] for clf in classifier_order}

        for label, xrag, query in setting2_features:
            feature_labels2.append(label)
            classifier_results = extract_classifier_results(
                results2, xrag, query, metric
            )
            for classifier in classifier_order:
                if classifier in classifier_results:
                    data_by_classifier2[classifier].append(
                        classifier_results[classifier]["mean"]
                    )
                    std_by_classifier2[classifier].append(
                        classifier_results[classifier]["std"]
                    )
                else:
                    data_by_classifier2[classifier].append(0)
                    std_by_classifier2[classifier].append(0)

        x2 = np.arange(len(feature_labels2))
        for i, classifier in enumerate(classifier_order):
            offset = (i - 1.5) * bar_width
            ax2.bar(
                x2 + offset,
                data_by_classifier2[classifier],
                bar_width,
                yerr=std_by_classifier2[classifier],
                label=classifier if col_idx == 0 else "",
                color=colors[classifier],
                alpha=0.8,
                capsize=3,
                error_kw={"linewidth": 1},
            )

        ax2.set_xticks(x2)
        ax2.set_xticklabels(
            feature_labels2, fontsize=12, rotation=0, ha="center"
        )
        ax2.set_xlabel("Feature Combination", fontweight="bold", fontsize=12)
        ax2.set_ylim(0.5, 0.80)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        if col_idx == 0:
            ax2.set_ylabel("ROC-AUC", fontweight="bold", fontsize=12)
            ax2.text(
                -0.18,
                0.5,
                "LLM-Dependent",
                transform=ax2.transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                ha="center",
                rotation=90,
            )

    plt.tight_layout(rect=[0.01, 0, 1, 0.99])

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    filename = output_path / "classifier_performance_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.savefig(filename.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {filename}")
    print(f"Saved: {filename.with_suffix('.pdf')}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize classifier performance from probing experiment results"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base directory containing one subdir per dataset (e.g. mistral_squad) with probing_results_*.json",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mistral_squad", "mistral_trivia", "mistral_hotpot"],
        help="Dataset subdirectory names under base_path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="probing",
        help="Experiment name used in result filenames (probing_results_<setting>_<experiment_name>.json)",
    )
    args = parser.parse_args()

    print("Creating classifier performance comparison figure...")
    create_classifier_comparison_plot(
        args.base_path,
        args.datasets,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    print(f"\n✓ Visualization saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
