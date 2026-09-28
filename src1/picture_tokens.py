import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "plot_data" / "reward_cost_comparison.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "comparison_plot_with_vlines_with_legend.pdf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot reward, token, query, and latency comparison from a JSON data file."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Input JSON data file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output PDF path.")
    return parser.parse_args()


def load_plot_data(path):
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    required = {"layouts", "methods", "method_labels", "colors", "metrics"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Missing required fields in {path}: {sorted(missing)}")
    return payload


def setup_style():
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def set_log2_yscale(ax):
    try:
        ax.set_yscale("log", base=2)
    except (TypeError, ValueError):
        ax.set_yscale("log", basey=2)


def main():
    args = parse_args()
    payload = load_plot_data(args.data)
    setup_style()

    layout_abbr = payload["layouts"]
    methods = payload["methods"]
    method_labels = payload["method_labels"]
    colors = payload["colors"]
    metric_configs = payload["metrics"]

    x = np.arange(len(layout_abbr)) * 0.42
    width = 0.11
    bar_offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width

    fig = plt.figure(figsize=(13.2, 12.0))
    gs = GridSpec(3, 2, height_ratios=[0.12, 1.0, 1.0], hspace=1.02, wspace=0.28)

    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis("off")
    legend_handles = [
        Patch(facecolor=colors[method], edgecolor="white", label=method_labels[method])
        for method in methods
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -4.0),
        ncol=len(methods),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.8,
        fontsize=18,
    )

    axs = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    for ax, cfg in zip(axs, metric_configs):
        metric = cfg["mean"]
        metric_max = cfg["max"]
        metric_min = cfg["min"]

        for method, offset in zip(methods, bar_offsets):
            bars = ax.bar(
                x + offset,
                metric[method],
                width,
                color=colors[method],
                edgecolor="white",
                linewidth=0.8,
                label=method_labels[method],
            )

            for i, bar in enumerate(bars):
                center = bar.get_x() + bar.get_width() / 2
                ax.vlines(center, metric_min[method][i], metric_max[method][i], color="black", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(layout_abbr, fontsize=17)
        ax.tick_params(axis="x", pad=6)
        ax.set_title(cfg["title"], pad=18, fontweight="semibold")
        ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if cfg.get("yscale") == "log2":
            set_log2_yscale(ax)
            ax.yaxis.set_major_locator(LogLocator(base=2))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

        ax.set_xlabel(cfg["caption"], labelpad=28, fontsize=22)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.965, bottom=0.095)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)


if __name__ == "__main__":
    main()
