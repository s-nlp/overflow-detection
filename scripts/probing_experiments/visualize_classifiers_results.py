#!/usr/bin/env python3
"""
Manual figure for best AUC / F1 / Accuracy from results table.

Fill values in MANUAL_RESULTS.
"""

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


DATASETS = ["TriviaQA", "SQuADv2", "HotpotQA", "Combined"]
MODELS = ["Mistral 7B", "Mixtral-8x7B"]
MODES = ["No query", "With query"]
METRICS = ["AUC", "F1", "Acc"]

# Fill manually:
# MANUAL_RESULTS[dataset][model][mode][metric] = value
MANUAL_RESULTS = {
    "TriviaQA": {
        "Mistral 7B": {
            "No query": {"AUC": 0.688, "F1": 0.119, "Acc": 0.803},
            "With query": {"AUC": 0.719, "F1": 0.106, "Acc": 0.905},
        },
        "Mixtral-8x7B": {
            "No query": {"AUC": 0.646, "F1": 0.008, "Acc": 0.905},
            "With query": {"AUC": 0.673, "F1": 0.000, "Acc": 0.905},
        },
    },

    "SQuADv2": {
        "Mistral 7B": {
            "No query": {"AUC": 0.640, "F1": 0.732, "Acc": 0.606},
            "With query": {"AUC": 0.696, "F1": 0.734, "Acc": 0.658},
        },
        "Mixtral-8x7B": {
            "No query": {"AUC": 0.649, "F1": 0.579, "Acc": 0.604},
            "With query": {"AUC": 0.692, "F1": 0.617, "Acc": 0.637},
        },
    },

    "HotpotQA": {
        "Mistral 7B": {
            "No query": {"AUC": 0.637, "F1": 0.651, "Acc": 0.605},
            "With query": {"AUC": 0.716, "F1": 0.682, "Acc": 0.659},
        },
        "Mixtral-8x7B": {
            "No query": {"AUC": 0.636, "F1": 0.601, "Acc": 0.595},
            "With query": {"AUC": 0.734, "F1": 0.683, "Acc": 0.672},
        },
    },

    "Combined": {
        "Mistral 7B": {
            "No query": {"AUC": 0.728, "F1": 0.556, "Acc": 0.674},
            "With query": {"AUC": 0.774, "F1": 0.639, "Acc": 0.709},
        },
        "Mixtral-8x7B": {
            "No query": {"AUC": 0.774, "F1": 0.554, "Acc": 0.730},
            "With query": {"AUC": 0.811, "F1": 0.609, "Acc": 0.765},
        },
    },
}


def create_manual_results_plot(output_dir="figures"):
    fig, axes = plt.subplots(2, 4, figsize=(26, 10), sharey=False)

    colors = {
        "AUC": "#5ED1FF",
        "F1": "#FFAF5E",
        "Acc": "#8238D9",
    }

    bar_width = 0.22
    x = np.arange(len(MODES))

    for col_idx, dataset in enumerate(DATASETS):
        for row_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]

            for metric_idx, metric in enumerate(METRICS):
                values = [
                    MANUAL_RESULTS[dataset][model][mode][metric]
                    for mode in MODES
                ]

                offset = (metric_idx - 1) * bar_width
                bars = ax.bar(
                    x + offset,
                    values,
                    bar_width,
                    label=metric if col_idx == 0 and row_idx == 0 else "",
                    color=colors[metric],
                    alpha=0.8,
                )

                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.015,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        rotation=90,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(MODES, fontsize=12)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            if row_idx == 0:
                ax.set_title(dataset, fontweight="bold", fontsize=14)

            if col_idx == 0:
                ax.set_ylabel("Score", fontweight="bold", fontsize=12)
                ax.text(
                    -0.22,
                    0.5,
                    model,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                    ha="center",
                    rotation=90,
                )

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", framealpha=0.95, fontsize=12)

    plt.tight_layout(rect=[0.01, 0, 1, 0.99])

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    png_path = output_path / "manual_best_results_comparison.png"
    pdf_path = output_path / "manual_best_results_comparison.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    create_manual_results_plot()