import csv
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

AGENTS = ["Co-π-Tree", "Co-π-Tree-TI", "SP", "PBT", "FCP", "MEP", "COLE"]
AGENT_LABELS = {
    "Co-π-Tree": r"Co-$\pi$-Tree",
    "Co-π-Tree-TI": r"Co-$\pi$-Tree-TI",
}

# role_2 uses slashes to indicate switch
ROLES = [
    ("role_1", "Role 1", ""),
    ("role_2", "Switch", "////"),
]

# Readjust colors:
# Increase contrast within the red, green, and purple families for easier distinction
COLORS = [
    "#8B0000",  # Co-π-Tree - dark red
    "#FF6B6B",  # Co-π-Tree-TI - light red
    "#006400",  # SP - dark green
    "#7CCD7C",  # PBT - light green
    "#4B0082",  # FCP - dark purple
    "#C084FC",  # MEP - light purple
    "#FF8C00",  # COLE - orange
]


def load_data(csv_path):
    """Load raw rewards from the long-format CSV."""
    layout_name = {raw: display for raw, display in LAYOUTS}
    data = {
        display: {
            agent: {role: [] for role, _, _ in ROLES}
            for agent in AGENTS
        }
        for _, display in LAYOUTS
    }

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"layout", "role_assignment", "method", "reward"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        for row in reader:
            raw_layout = row["layout"]
            role = row["role_assignment"]
            method = row["method"]
            if (
                raw_layout not in layout_name
                or method not in AGENTS
                or role not in data[layout_name[raw_layout]][method]
            ):
                continue
            data[layout_name[raw_layout]][method][role].append(int(row["reward"]))

    for layout in data:
        for agent in AGENTS:
            for role, _, _ in ROLES:
                values = data[layout][agent][role]
                if len(values) != 5:
                    raise ValueError(
                        f"Expected 5 rewards for {layout}/{agent}/{role}, "
                        f"got {len(values)}."
                    )
                data[layout][agent][role] = np.array(values)

    return data


def main():
    data = load_data(CSV_PATH)
    layouts = [display for _, display in LAYOUTS]

    # Slightly reduce the overall height
    fig, ax = plt.subplots(figsize=(12, 4.6))

    agent_step = 0.105
    role_gap = 0.045
    box_width = 0.032
    center_offset = (len(AGENTS) - 1) / 2

    box_data = []
    positions = []
    box_colors = []
    box_hatches = []

    for i, layout in enumerate(layouts):
        for j, agent in enumerate(AGENTS):
            agent_center = i + (j - center_offset) * agent_step
            for k, (role, _, hatch) in enumerate(ROLES):
                role_offset = (k - (len(ROLES) - 1) / 2) * role_gap
                box_data.append(data[layout][agent][role])
                positions.append(agent_center + role_offset)
                box_colors.append(COLORS[j])
                box_hatches.append(hatch)

    bplot = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showmeans=False,     # Do not show mean markers
        showfliers=False,    # Do not show outliers
    )

    # Set boxplot colors
    for patch, color, hatch in zip(bplot["boxes"], box_colors, box_hatches):
        patch.set_facecolor(color)
        patch.set_edgecolor("#222222")
        patch.set_hatch(hatch)
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
    agent_legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=4,
        frameon=False,
        fontsize=10
    )
    ax.add_artist(agent_legend)

    role_handles = [
        Patch(
            facecolor="white",
            edgecolor="#222222",
            hatch=hatch,
            label=label,
        )
        for _, label, hatch in ROLES
    ]
    ax.legend(
        handles=role_handles,
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    # No extra space is needed for significance markers
    y_max = max(
        np.max(data[layout][agent][role])
        for layout in layouts
        for agent in AGENTS
        for role, _, _ in ROLES
    )
    ax.set_ylim(bottom=-10, top=y_max * 1.08)

    fig.tight_layout()

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved boxplot to {SAVE_PATH}")


if __name__ == "__main__":
    main()
