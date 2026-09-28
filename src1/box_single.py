import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = Path(__file__).resolve().parent / "plot_data" / "human_rewards_long.csv"
SAVE_PATH = Path(__file__).resolve().parent / "comparison_plot_box.pdf"

# Original layout name -> abbreviated display name
LAYOUTS = [
    ("cramped room", "Cramped Rm."),
    ("coordination ring", "Coord. Ring"),
    ("counter circuit", "CT. Circuit"),
    ("asymmetric advantages", "Asymm. Adv."),
    ("forced coordination", "Forced Coord."),
]

AGENTS = [
    "Co-π-Tree",
    "Co-π-Tree-PI",
    "Co-π-Tree w/o P",
    "SP",
    "PBT",
    "FCP",
    "MEP",
    "COLE",
]
AGENT_LABELS = {
    "Co-π-Tree": r"Co-$\pi$-Tree",
    "Co-π-Tree-PI": r"Co-$\pi$-Tree-PI",
    "Co-π-Tree w/o P": r"Co-$\pi$-Tree w/o P",
}

METHOD_ALIASES = {
    "Co-π-Tree w/o P P": "Co-π-Tree w/o P",
}

# Co-π-Tree variants use one blue family with stronger light/dark contrast.
COLORS = [
    "#08306B",  # Co-π-Tree - dark blue
    "#2B8CBE",  # Co-π-Tree-PI - medium blue
    "#A6CEE3",  # Co-π-Tree w/o P - light blue
    "#1B9E77",  # SP - green
    "#7570B3",  # PBT - purple
    "#E7298A",  # FCP - magenta
    "#66A61E",  # MEP - olive
    "#666666",  # COLE - gray
]


REFERENCE_AGENT = "Co-π-Tree"
SIGNIFICANCE_AGENTS = [agent for agent in AGENTS if agent != REFERENCE_AGENT]


def mann_whitney_u_pvalue(sample_a, sample_b):
    """Two-sided Mann-Whitney U test with normal approximation and tie correction."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    n_a = len(sample_a)
    n_b = len(sample_b)
    combined = [(value, 0) for value in sample_a] + [(value, 1) for value in sample_b]
    combined.sort(key=lambda item: item[0])

    ranks = np.zeros(len(combined), dtype=float)
    tie_sizes = []
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0
        tie_sizes.append(j - i)
        i = j

    rank_sum_a = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 0)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    n_total = n_a + n_b
    mean_u = n_a * n_b / 2.0
    tie_term = sum(size ** 3 - size for size in tie_sizes)
    variance_u = n_a * n_b / 12.0 * (
        n_total + 1 - tie_term / (n_total * (n_total - 1))
    )
    if variance_u <= 0:
        return 1.0

    z = (abs(u - mean_u) - 0.5) / math.sqrt(variance_u)
    z = max(z, 0.0)
    return math.erfc(z / math.sqrt(2.0))


def holm_adjust(pvalue_entries):
    """Return Holm-Bonferroni adjusted p-values for (key, pvalue) pairs."""
    sorted_entries = sorted(pvalue_entries, key=lambda item: item[1])
    adjusted = {}
    running_max = 0.0
    total = len(sorted_entries)

    for rank, (key, pvalue) in enumerate(sorted_entries):
        corrected = min((total - rank) * pvalue, 1.0)
        running_max = max(running_max, corrected)
        adjusted[key] = running_max

    return adjusted


def significance_label(pvalue):
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def plot_significance_label(pvalue):
    label = significance_label(pvalue)
    return "n.s." if label == "ns" else label


def compute_significance(data, layouts):
    pvalue_entries = []
    raw_pvalues = {}

    for layout in layouts:
        reference_values = data[layout][REFERENCE_AGENT]
        for agent in SIGNIFICANCE_AGENTS:
            key = (layout, agent)
            pvalue = mann_whitney_u_pvalue(reference_values, data[layout][agent])
            raw_pvalues[key] = pvalue
            pvalue_entries.append((key, pvalue))

    adjusted_pvalues = holm_adjust(pvalue_entries)
    return raw_pvalues, adjusted_pvalues


def print_significance(raw_pvalues, adjusted_pvalues, layouts):
    print("Significance: Mann-Whitney U test, two-sided, Holm-adjusted")
    for layout in layouts:
        print(f"\n{layout}: {REFERENCE_AGENT} vs.")
        for agent in SIGNIFICANCE_AGENTS:
            key = (layout, agent)
            print(
                f"  {agent:18s} "
                f"raw p={raw_pvalues[key]:.4g}, "
                f"adj p={adjusted_pvalues[key]:.4g} "
                f"{significance_label(adjusted_pvalues[key])}"
            )


def draw_significance_bracket(ax, x_start, x_end, y, bracket_height, label, text_gap):
    ax.plot(
        [x_start, x_start, x_end, x_end],
        [y, y + bracket_height, y + bracket_height, y],
        color="#222222",
        linewidth=0.9,
        clip_on=False,
    )
    ax.text(
        (x_start + x_end) / 2.0,
        y + bracket_height + text_gap,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        clip_on=False,
    )


def add_significance_labels(ax, data, layouts, position_lookup, adjusted_pvalues):
    y_min = min(np.min(data[layout][agent]) for layout in layouts for agent in AGENTS)
    y_max = max(np.max(data[layout][agent]) for layout in layouts for agent in AGENTS)
    y_range = max(y_max - y_min, 1)
    base_gap = y_range * 0.045
    row_gap = y_range * 0.058
    bracket_height = y_range * 0.010
    text_gap = y_range * 0.001

    top_by_layout = {}
    for layout in layouts:
        top_by_layout[layout] = max(np.max(data[layout][agent]) for agent in AGENTS)
        reference_x = position_lookup[(layout, REFERENCE_AGENT)]

        for row, agent in enumerate(SIGNIFICANCE_AGENTS):
            label = plot_significance_label(adjusted_pvalues[(layout, agent)])
            comparison_x = position_lookup[(layout, agent)]
            y = top_by_layout[layout] + base_gap + row * row_gap
            draw_significance_bracket(
                ax,
                reference_x,
                comparison_x,
                y,
                bracket_height,
                label,
                text_gap,
            )

    needed_top = max(
        top_by_layout[layout]
        + base_gap
        + (len(SIGNIFICANCE_AGENTS) - 1) * row_gap
        + bracket_height
        + text_gap
        for layout in layouts
    )
    return needed_top


def load_data(csv_path):
    """Load raw rewards from the long-format CSV."""
    layout_name = {raw: display for raw, display in LAYOUTS}
    data = {display: {agent: [] for agent in AGENTS} for _, display in LAYOUTS}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"layout", "method", "reward"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            raw_layout = row["layout"]
            method = METHOD_ALIASES.get(row["method"], row["method"])
            if raw_layout not in layout_name or method not in AGENTS:
                continue
            data[layout_name[raw_layout]][method].append(int(row["reward"]))

    for layout in data:
        for agent in AGENTS:
            values = data[layout][agent]
            if len(values) != 20:
                raise ValueError(
                    f"Expected 20 rewards for {layout}/{agent} "
                    f"(2 role assignments x 10 runs), got {len(values)}."
                )
            data[layout][agent] = np.array(values)

    return data


def main():
    data = load_data(CSV_PATH)
    layouts = [display for _, display in LAYOUTS]

    # Slightly reduce the overall height
    fig, ax = plt.subplots(figsize=(12, 4.6))

    width = 0.11
    center_offset = (len(AGENTS) - 1) / 2

    box_data = []
    positions = []
    box_colors = []
    position_lookup = {}

    for i, layout in enumerate(layouts):
        for j, agent in enumerate(AGENTS):
            position = i + (j - center_offset) * width
            box_data.append(data[layout][agent])
            positions.append(position)
            box_colors.append(COLORS[j])
            position_lookup[(layout, agent)] = position

    bplot = ax.boxplot(
        box_data,
        positions=positions,
        widths=width * 0.85,
        patch_artist=True,
        showmeans=False,    # Do not show mean markers
        showfliers=False,   # Do not show outliers
    )

    # Set boxplot colors
    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        patch.set_linewidth(0.9)

    for element in ["whiskers", "caps", "medians"]:
        for item in bplot[element]:
            item.set_color("#222222")
            item.set_linewidth(0.9)

    ax.set_xticks(range(len(layouts)))
    ax.set_xticklabels(layouts, rotation=0, ha="center", fontsize=12)
    ax.set_ylabel("Episodic reward", fontsize=12)
    ax.tick_params(axis="y", labelsize=11, pad=4)

    ax.set_xlim(-0.55, len(layouts) - 0.45)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    # Legend
    legend_handles = [
        Patch(facecolor=COLORS[i], alpha=0.82, label=AGENT_LABELS.get(agent, agent))
        for i, agent in enumerate(AGENTS)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=4,
        frameon=False,
        fontsize=10
    )

    raw_pvalues, adjusted_pvalues = compute_significance(data, layouts)
    print_significance(raw_pvalues, adjusted_pvalues, layouts)
    significance_top = add_significance_labels(
        ax,
        data,
        layouts,
        position_lookup,
        adjusted_pvalues,
    )

    y_max = max(np.max(data[layout][agent]) for layout in layouts for agent in AGENTS)
    ax.set_ylim(bottom=-10, top=max(y_max * 1.08, significance_top * 1.04))

    fig.tight_layout()

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved boxplot to {SAVE_PATH}")


if __name__ == "__main__":
    main()
