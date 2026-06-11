#!/usr/bin/env python3
'''
python results_table_latex.py --include-accuracy \
    trivia_7b_no_query_heldout_test.json \
    trivia_7b_with_query_heldout_test.json \
    trivia_moe_no_query_heldout_test.json \
    trivia_moe_with_query_heldout_test.json \
    hotpotqa_7b_no_query_heldout_test.json \
    hotpotqa_7b_with_query_heldout_test.json \
    hotpotqa_moe_no_query_heldout_test.json \
    hotpotqa_moe_with_query_heldout_test.json \
    squad_7b_no_query_heldout_test.json \
    squad_7b_with_query_heldout_test.json \
    squad_moe_no_query_heldout_test.json \
    squad_moe_with_query_heldout_test.json \
    combined_7b_no_query_heldout_test.json \
    combined_7b_with_query_heldout_test.json \
    combined_moe_no_query_heldout_test.json \
    combined_moe_with_query_heldout_test.json 
'''
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

DATASET_MAP = {
    "trivia": "TriviaQA",
    "hotpotqa": "HotpotQA",
    "squad": "SQuAD",
    "combined": "Combined",
}

MODEL_MAP = {
    "7b": "Mistral 7B",
    "moe": "Mixtral-8x7B",
}

MODE_MAP = {
    "no_query": "No query",
    "with_query": "With query",
}

DATASET_ORDER = ["trivia", "hotpotqa", "squad", "combined"]
MODE_ORDER = ["no_query", "with_query"]
MODEL_ORDER = ["7b", "moe"]

# Example:
# trivia_moe_with_query_heldout_test.json
FILENAME_RE = re.compile(
    r"^(?P<dataset>trivia|hotpotqa|squad|combined)_(?P<model>7b|moe)_(?P<mode>no_query|with_query).*\.json$"
)

DEFAULT_METRICS = [
    ("test_auc", "AUC"),
    ("test_f1", "F1"),
    ("test_n", "N"),
    ("test_positive", "Pos"),
]


def read_json_summary(path: Path) -> Dict[str, Any]:
    """
    Reads a JSON file and returns the first JSON object.
    If there are multiple lines, it uses the first non-empty valid JSON line.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

    raise ValueError(f"No valid JSON object found in file: {path}")


def parse_filename(path: Path) -> Tuple[str, str, str]:
    """
    Extract dataset, model, mode from filename.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Filename does not match expected pattern: {path.name}\n"
            f"Expected something like: trivia_moe_with_query_heldout_test.json"
        )
    return m.group("dataset"), m.group("model"), m.group("mode")


def format_value(metric_name: str, value: Any) -> str:
    if value is None:
        return "--"

    if metric_name in {"test_auc", "test_pr_auc", "test_f1", "test_accuracy"}:
        try:
            return f"{float(value):.3f}"
        except Exception:
            return str(value)

    if metric_name in {"test_n", "test_positive"}:
        try:
            return str(int(value))
        except Exception:
            return str(value)

    return str(value)


def collect_results(files: List[Path]) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Nested structure:
    results[dataset][mode][model] = metrics_dict
    """
    results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    for path in files:
        dataset, model, mode = parse_filename(path)
        metrics = read_json_summary(path)

        results.setdefault(dataset, {}).setdefault(mode, {})[model] = metrics

    return results


def generate_latex_table(
    results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    metrics: List[Tuple[str, str]],
    caption: str = "Probe performance across datasets, modes, and models.",
    label: str = "tab:probe_results",
) -> str:
    metric_labels = [label for _, label in metrics]
    n_metrics = len(metrics)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll" + "c" * n_metrics + "c" * n_metrics + "}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Mode} "
        + f"& \\multicolumn{{{n_metrics}}}{{c}}{{Mistral 7B}} "
        + f"& \\multicolumn{{{n_metrics}}}{{c}}{{Mixtral-8x7B}} \\\\"
    )
    lines.append(
        r"\cmidrule(lr){3-" + str(2 + n_metrics) + "}"
        + r"\cmidrule(lr){"
        + str(3 + n_metrics)
        + "-"
        + str(2 + 2 * n_metrics)
        + "}"
    )
    lines.append(
        " & & "
        + " & ".join(metric_labels)
        + " & "
        + " & ".join(metric_labels)
        + r" \\"
    )
    lines.append(r"\midrule")

    for dataset in DATASET_ORDER:
        dataset_label = DATASET_MAP[dataset]
        first_row_for_dataset = True

        for mode in MODE_ORDER:
            mode_label = MODE_MAP[mode]

            row = []
            if first_row_for_dataset:
                row.append(rf"\multirow{{2}}{{*}}{{{dataset_label}}}")
                first_row_for_dataset = False
            else:
                row.append("")

            row.append(mode_label)

            for model in MODEL_ORDER:
                model_metrics = results.get(dataset, {}).get(mode, {}).get(model, {})
                for metric_name, _ in metrics:
                    row.append(format_value(metric_name, model_metrics.get(metric_name)))

            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    # Remove the final extra midrule and replace with bottomrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)

def generate_best_metrics_latex_table(
    results,
    caption="Best probe statistics across datasets, modes, and models.",
    label="tab:best_probe_statistics",
):
    """
    Generates a compact LaTeX table with AUC / F1 / Accuracy
    for each dataset, model, and setup.

    Rows: Dataset + Model
    Columns: No-query AUC/F1/Acc, With-query AUC/F1/Acc
    """

    metrics = [
        ("test_auc", "AUC"),
        ("test_f1", "F1"),
        ("test_accuracy", "Acc"),
    ]

    lines = []
    lines.append(r"\begin{table*}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc|ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Model} "
        r"& \multicolumn{3}{c}{\textbf{No query}} "
        r"& \multicolumn{3}{c}{\textbf{With query}} \\"
    )
    lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
    lines.append(
        r" & & \textbf{AUC} & \textbf{F1} & \textbf{Acc} "
        r"& \textbf{AUC} & \textbf{F1} & \textbf{Acc} \\"
    )
    lines.append(r"\midrule")

    for dataset in DATASET_ORDER:
        dataset_label = DATASET_MAP[dataset]

        for model_idx, model in enumerate(MODEL_ORDER):
            model_label = MODEL_MAP[model]

            row = []
            row.append(rf"\multirow{{2}}{{*}}{{{dataset_label}}}" if model_idx == 0 else "")
            row.append(model_label)

            for mode in MODE_ORDER:
                model_metrics = results.get(dataset, {}).get(mode, {}).get(model, {})

                for metric_name, _ in metrics:
                    row.append(format_value(metric_name, model_metrics.get(metric_name)))

            lines.append(" & ".join(row) + r" \\")

        if dataset != DATASET_ORDER[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from experiment summary JSON files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="List of JSON files, e.g. trivia_moe_with_query_heldout_test.json",
    )
    parser.add_argument(
        "--include-pr-auc",
        action="store_true",
        help="Include PR-AUC in the output table.",
    )
    parser.add_argument(
        "--include-accuracy",
        action="store_true",
        help="Include accuracy in the output table.",
    )
    parser.add_argument(
        "--caption",
        default="Probe performance across datasets, modes, and models.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:probe_results",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/final_results_table.tex",
        help="Optional output .tex file. If omitted, prints to stdout.",
    )

    args = parser.parse_args()

    metrics = list(DEFAULT_METRICS)
    if args.include_pr_auc:
        metrics.insert(1, ("test_pr_auc", "PR-AUC"))
    if args.include_accuracy:
        # Put accuracy after F1
        insert_pos = 3 if args.include_pr_auc else 2
        metrics.insert(insert_pos, ("test_accuracy", "Acc"))

    files = [Path(f) for f in args.files]
    results = collect_results(files)

    # latex = generate_latex_table(
    #     results=results,
    #     metrics=metrics,
    #     caption=args.caption,
    #     label=args.label,
    # )


    best_metrics_latex = generate_best_metrics_latex_table(
    results=results,
    caption=(
        "Best probe statistics for overflow detection across datasets, "
        "models, and input setups. We report ROC-AUC, F1, and accuracy "
        "for no-query and with-query settings."
    ),
    label="tab:best_probe_statistics",
    )

    # if args.output:
    #     out_path = Path(args.output)
    #     out_path.write_text(latex, encoding="utf-8")
    #     print(f"Saved LaTeX table to {out_path}")
    # else:
    #     print(latex)
    
    best_out_path = Path("figures/best_probe_statistics_table.tex")
    best_out_path.write_text(best_metrics_latex, encoding="utf-8")
    print(f"Saved best-metrics LaTeX table to {best_out_path}")


if __name__ == "__main__":
    main()