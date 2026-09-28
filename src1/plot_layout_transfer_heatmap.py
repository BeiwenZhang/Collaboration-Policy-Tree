import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = (
    PROJECT_DIR / "results/layout_transfer" / "layout_transfer_main" / "summary.json"
)
DEFAULT_SP_SUMMARY_PATH = (
    PROJECT_DIR / "results/cross_layout_sp" / "sp_probe_batch_zsc_both_with_diag.json"
)
DEFAULT_REFERENCE_SUMMARY_PATH = (
    Path(__file__).resolve().parent / "plot_data" / "layout_transfer_reference.json"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "layout_transfer_heatmap.pdf"
LAYOUT_LABELS_TWO_LINE = {
    "cramped_room": "Cramped\nRm.",
    "coordination_ring": "Coord.\nRing",
    "counter_circuit": "CT.\nCircuit",
    "asymmetric_advantages": "Asymm.\nAdv.",
    "forced_coordination": "Forced\nCoord.",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a paper-ready layout-transfer heatmap from summary.json."
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to the transfer summary.json produced by layout_transfer_experiment.py.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for the heatmap PDF. When --role both, suffixes are added automatically.",
    )
    parser.add_argument(
        "--sp-summary",
        default=str(DEFAULT_SP_SUMMARY_PATH),
        help="Path to the SP cross-layout batch summary JSON.",
    )
    parser.add_argument(
        "--reference-summary",
        default=str(DEFAULT_REFERENCE_SUMMARY_PATH),
        help="Path to the reference-method layout-transfer JSON.",
    )
    parser.add_argument(
        "--metric",
        choices=["retention", "raw"],
        default="retention",
        help="Plot normalized retention or raw reward.",
    )
    parser.add_argument(
        "--role",
        choices=["combined", "p0", "p1", "both"],
        default="both",
        help="Which aggregation to visualize. 'both' saves separate p0 and p1 figures.",
    )
    parser.add_argument(
        "--annot",
        choices=["auto", "percent", "raw", "both", "none"],
        default="percent",
        help="Cell annotation style.",
    )
    return parser.parse_args()


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_matrix_keys(metric: str, role: str):
    if role == "combined":
        raw_key = "combined_mean_reward"
        retention_key = "combined_retention"
        title_role = "combined"
    elif role == "p0":
        raw_key = "p0_mean_reward"
        retention_key = "p0_retention"
        title_role = "p0"
    else:
        raw_key = "p1_mean_reward"
        retention_key = "p1_retention"
        title_role = "p1"

    if metric == "retention":
        return retention_key, raw_key, title_role
    return raw_key, raw_key, title_role


def matrix_to_array(matrix_dict, layout_order):
    arr = np.full((len(layout_order), len(layout_order)), np.nan, dtype=float)
    for i, src in enumerate(layout_order):
        for j, tgt in enumerate(layout_order):
            value = matrix_dict.get(src, {}).get(tgt)
            if value is not None:
                arr[i, j] = float(value)
    return arr


def build_annotations(metric_arr, raw_arr, annot_mode, metric):
    rows, cols = metric_arr.shape
    text = [["" for _ in range(cols)] for _ in range(rows)]

    if annot_mode == "auto":
        annot_mode = "both" if metric == "retention" else "raw"

    for i in range(rows):
        for j in range(cols):
            metric_value = metric_arr[i, j]
            raw_value = raw_arr[i, j]
            if math.isnan(metric_value) and math.isnan(raw_value):
                text[i][j] = "n/a"
                continue

            if annot_mode == "none":
                text[i][j] = ""
            elif annot_mode == "percent":
                text[i][j] = "" if math.isnan(metric_value) else f"{metric_value * 100:.0f}%"
            elif annot_mode == "raw":
                text[i][j] = "" if math.isnan(raw_value) else f"{raw_value:.1f}"
            else:
                top = "" if math.isnan(metric_value) else f"{metric_value * 100:.0f}%"
                bottom = "" if math.isnan(raw_value) else f"({raw_value:.1f})"
                if top and bottom:
                    text[i][j] = f"{top}\n{bottom}"
                else:
                    text[i][j] = top or bottom
    return text


def get_two_line_labels(layout_order):
    return [LAYOUT_LABELS_TWO_LINE.get(layout, layout) for layout in layout_order]


def make_colormap(metric: str):
    if metric == "retention":
        return LinearSegmentedColormap.from_list(
            "transfer_retention",
            ["#f5f3ef", "#f6f2eb", "#dbe8e8", "#73b8bf", "#2f7f8d", "#103743"],
        )
    return LinearSegmentedColormap.from_list(
        "transfer_reward",
        ["#f7f2e8", "#cfe4df", "#78b0b1", "#245c73"],
    )


class MidpointGammaNorm(Normalize):
    """
    Smooth normalization around a semantic midpoint. Values below the midpoint
    map to the warm half, and values above the midpoint map to the cool half.
    The upper half gets a mild gamma boost so differences in strong cells are
    visible without looking visually harsh.
    """

    def __init__(
        self,
        vmin=None,
        vmax=None,
        clip=False,
        midpoint=1.0,
        lower_gamma=1.0,
        upper_gamma=0.78,
    ):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint
        self.lower_gamma = lower_gamma
        self.upper_gamma = upper_gamma

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)
        data = np.array(result.data, dtype=float, copy=True)
        mask = np.ma.getmask(result)

        self.autoscale_None(data)
        vmin = float(self.vmin)
        vmax = float(self.vmax)

        if clip:
            data = np.clip(data, vmin, vmax)

        if vmax <= vmin:
            mapped = np.zeros_like(data)
        else:
            midpoint = min(max(self.midpoint, vmin), vmax)
            low_span = max(midpoint - vmin, 1e-12)
            high_span = max(vmax - midpoint, 1e-12)

            mapped = np.empty_like(data, dtype=float)
            low_mask = data <= midpoint
            high_mask = data > midpoint

            low_scaled = (data[low_mask] - vmin) / low_span
            high_scaled = (data[high_mask] - midpoint) / high_span

            mapped[low_mask] = 0.65 * np.power(low_scaled, self.lower_gamma)
            mapped[high_mask] = 0.65 + 0.35 * np.power(high_scaled, self.upper_gamma)

        mapped = np.ma.array(mapped, mask=mask)
        if is_scalar:
            return mapped[0]
        return mapped


def build_shared_norm(metric: str, *arrays):
    finite_chunks = []
    for arr in arrays:
        if arr is None:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_chunks.append(finite)

    if finite_chunks:
        values = np.concatenate(finite_chunks)
    else:
        values = np.array([0.0, 1.0], dtype=float)

    if metric == "retention":
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        vmin = min(vmin, 1.0)
        vmax = max(vmax, 1.0)
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        return MidpointGammaNorm(
            vmin=vmin,
            vmax=vmax,
            midpoint=1.0,
            lower_gamma=1.0,
            upper_gamma=0.78,
        ), "Retention"

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6
    return Normalize(vmin=vmin, vmax=vmax), "Mean Reward"


def build_output_path(base_output: Path, role: str) -> Path:
    if role == "combined":
        return base_output
    return base_output.with_name(f"{base_output.stem}_{role}{base_output.suffix}")


def load_reference_matrix(path: Path, layout_order):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    matrix = payload.get("retention")
    if matrix is None:
        raise ValueError(f"Missing 'retention' matrix in {path}")
    arr = np.array(matrix, dtype=float)
    if arr.shape != (len(layout_order), len(layout_order)):
        raise ValueError(
            f"Reference matrix shape {arr.shape} does not match layout count {len(layout_order)}."
        )
    return arr


def prepare_heatmap_data(summary, metric: str, role: str, annot: str):
    layout_order = summary["layout_order"]
    matrix_key, raw_key, _ = choose_matrix_keys(metric, role)

    metric_arr = matrix_to_array(summary["matrices"][matrix_key], layout_order)
    raw_arr = matrix_to_array(summary["matrices"][raw_key], layout_order)
    annotations = build_annotations(metric_arr, raw_arr, annot, metric)
    return layout_order, metric_arr, annotations


def prepare_reference_heatmap_data(summary, annot: str, reference_summary_path: Path):
    layout_order = summary["layout_order"]
    metric_arr = load_reference_matrix(reference_summary_path, layout_order)
    raw_arr = metric_arr.copy()
    annotations = build_annotations(metric_arr, raw_arr, annot, "retention")
    return layout_order, metric_arr, annotations


def prepare_sp_heatmap_data(summary, sp_bundle, annot: str):
    layout_order = summary["layout_order"]
    reward_matrix = sp_bundle.get("matrix_avg_reward", {})
    feasible_pairs = sp_bundle.get("feasible_pairs", [])
    feasible_set = {
        (row["source_layout"], row["target_layout"])
        for row in feasible_pairs
    }

    reward_arr = np.full((len(layout_order), len(layout_order)), np.nan, dtype=float)
    missing_mask = np.ones((len(layout_order), len(layout_order)), dtype=bool)

    for i, source in enumerate(layout_order):
        for j, target in enumerate(layout_order):
            if (source, target) in feasible_set:
                value = reward_matrix.get(source, {}).get(target)
                if value is not None:
                    reward_arr[i, j] = float(value)
                    missing_mask[i, j] = False

    retention_arr = np.full_like(reward_arr, np.nan)
    for j, target in enumerate(layout_order):
        target_diag = reward_matrix.get(target, {}).get(target)
        if target_diag is None or target_diag == 0:
            continue
        for i, source in enumerate(layout_order):
            if not missing_mask[i, j] and not np.isnan(reward_arr[i, j]):
                retention_arr[i, j] = float(reward_arr[i, j] / target_diag)

    annotations = [["" for _ in layout_order] for _ in layout_order]
    for i in range(len(layout_order)):
        for j in range(len(layout_order)):
            if missing_mask[i, j]:
                annotations[i][j] = "/"
            elif np.isnan(retention_arr[i, j]):
                annotations[i][j] = "/"
                missing_mask[i, j] = True
            else:
                annotations[i][j] = f"{retention_arr[i, j] * 100:.0f}%"

    return layout_order, retention_arr, annotations, missing_mask


def draw_heatmap_on_axis(
    ax,
    summary,
    metric: str,
    role: str,
    annot: str,
    show_ylabel: bool,
    show_colorbar: bool,
    norm=None,
):
    layout_order, metric_arr, annotations = prepare_heatmap_data(summary, metric, role, annot)

    cmap = make_colormap(metric)
    cmap.set_bad(color="#eeeeee")

    if norm is None:
        norm, colorbar_label = build_shared_norm(metric, metric_arr)
    else:
        colorbar_label = "Retention" if metric == "retention" else "Mean Reward"

    im = ax.imshow(metric_arr, cmap=cmap, norm=norm)

    ax.set_xticks(range(len(layout_order)))
    ax.set_yticks(range(len(layout_order)))
    ax.set_xticklabels(get_two_line_labels(layout_order), rotation=0, ha="center")
    ax.set_yticklabels(get_two_line_labels(layout_order))
    ax.set_xlabel("Target Layout", labelpad=8, fontsize=13, fontweight="normal")
    if show_ylabel:
        ax.set_ylabel("Source Layout", labelpad=10, fontsize=13, fontweight="normal")
    else:
        ax.set_ylabel("")

    ax.set_xticks(np.arange(-0.5, len(layout_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layout_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=8)

    finite_values = metric_arr[np.isfinite(metric_arr)]
    for i in range(metric_arr.shape[0]):
        for j in range(metric_arr.shape[1]):
            label = annotations[i][j]
            if not label:
                continue
            value = metric_arr[i, j]
            color = "#1f1f1f"
            if np.isfinite(value) and norm(value) > 0.62:
                color = "white"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="semibold",
                color=color,
                linespacing=1.0,
            )

    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        if metric == "retention":
            ticks = cbar.get_ticks()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tick * 100:.0f}%" for tick in ticks])
        cbar.set_label(colorbar_label, fontsize=13, fontweight="normal", rotation=90, labelpad=10)
        cbar.ax.tick_params(labelsize=12)


def draw_custom_heatmap_on_axis(
    ax,
    layout_order,
    metric_arr,
    annotations,
    metric: str,
    show_ylabel: bool,
    norm=None,
    missing_mask=None,
):
    cmap = make_colormap(metric)

    if norm is None:
        norm, _ = build_shared_norm(metric, metric_arr)

    display_arr = np.array(metric_arr, copy=True)
    if missing_mask is not None:
        display_arr[missing_mask] = float(norm.vmin)

    im = ax.imshow(display_arr, cmap=cmap, norm=norm)

    ax.set_xticks(range(len(layout_order)))
    ax.set_yticks(range(len(layout_order)))
    ax.set_xticklabels(get_two_line_labels(layout_order), rotation=0, ha="center")
    ax.set_yticklabels(get_two_line_labels(layout_order))
    ax.set_xlabel("Target Layout", labelpad=8, fontsize=13, fontweight="normal")
    if show_ylabel:
        ax.set_ylabel("Source Layout", labelpad=10, fontsize=13, fontweight="normal")
    else:
        ax.set_ylabel("")

    ax.set_xticks(np.arange(-0.5, len(layout_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layout_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=8)

    for i in range(metric_arr.shape[0]):
        for j in range(metric_arr.shape[1]):
            label = annotations[i][j]
            if not label:
                continue
            if missing_mask is not None and missing_mask[i, j]:
                color = "#1f1f1f"
            else:
                value = metric_arr[i, j]
                color = "#1f1f1f"
                if np.isfinite(value) and norm(value) > 0.62:
                    color = "white"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="semibold",
                color=color,
                linespacing=1.0,
            )
    return im


def draw_single_heatmap(summary, output_path: Path, metric: str, role: str, annot: str):
    fig, ax = plt.subplots(figsize=(8.8, 6.9))
    _, metric_arr, _ = prepare_heatmap_data(summary, metric, role, annot)
    norm, _ = build_shared_norm(metric, metric_arr)
    draw_heatmap_on_axis(
        ax=ax,
        summary=summary,
        metric=metric,
        role=role,
        annot=annot,
        show_ylabel=True,
        show_colorbar=True,
        norm=norm,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def draw_three_panel_heatmaps(summary, sp_bundle, reference_summary_path: Path, output_path: Path, metric: str, annot: str):
    fig = plt.figure(figsize=(17.0, 5.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.48)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])
    _, combined_metric_arr, _ = prepare_heatmap_data(summary, metric, "combined", annot)
    layout_order, reference_metric_arr, reference_annotations = prepare_reference_heatmap_data(
        summary, annot, reference_summary_path
    )
    (
        sp_layout_order,
        sp_metric_arr,
        sp_annotations,
        sp_missing_mask,
    ) = prepare_sp_heatmap_data(summary, sp_bundle, annot)
    if sp_layout_order != layout_order:
        raise ValueError("SP summary layout order does not match main summary layout order.")
    shared_norm, colorbar_label = build_shared_norm(
        metric, combined_metric_arr, reference_metric_arr, sp_metric_arr
    )

    draw_custom_heatmap_on_axis(
        ax=ax_left,
        layout_order=layout_order,
        metric_arr=sp_metric_arr,
        annotations=sp_annotations,
        metric=metric,
        show_ylabel=True,
        norm=shared_norm,
        missing_mask=sp_missing_mask,
    )
    draw_custom_heatmap_on_axis(
        ax=ax_mid,
        layout_order=layout_order,
        metric_arr=reference_metric_arr,
        annotations=reference_annotations,
        metric=metric,
        show_ylabel=False,
        norm=shared_norm,
    )
    draw_heatmap_on_axis(
        ax=ax_right,
        summary=summary,
        metric=metric,
        role="combined",
        annot=annot,
        show_ylabel=False,
        norm=shared_norm,
        show_colorbar=False,
    )
    cmap = make_colormap(metric)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=shared_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    if metric == "retention":
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{tick * 100:.0f}%" for tick in ticks])
    cbar.set_label(colorbar_label, fontsize=13, fontweight="normal", rotation=90, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    ax_left.text(
        0.5,
        -0.3,
        r"(a) SP",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        fontfamily="serif",
        fontname="Times New Roman",
    )
    ax_mid.text(
        0.5,
        -0.3,
        r"(b) ProAgent",
        transform=ax_mid.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        fontfamily="serif",
        fontname="Times New Roman",
    )
    ax_right.text(
        0.5,
        -0.3,
        r"(c) Co-$\pi$-tree",
        transform=ax_right.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        fontfamily="serif",
        fontname="Times New Roman",
    )

    fig.subplots_adjust(left=0.06, right=0.965, bottom=0.14, top=0.97, wspace=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def main():
    args = parse_args()
    setup_style()

    summary_path = Path(args.summary).expanduser().resolve()
    sp_summary_path = Path(args.sp_summary).expanduser().resolve()
    reference_summary_path = Path(args.reference_summary).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    bundle = load_summary(summary_path)
    summary = bundle["summary"]
    if args.role == "both":
        sp_bundle = load_summary(sp_summary_path)
        draw_three_panel_heatmaps(
            summary=summary,
            sp_bundle=sp_bundle,
            reference_summary_path=reference_summary_path,
            output_path=output_path,
            metric=args.metric,
            annot=args.annot,
        )
    else:
        draw_single_heatmap(
            summary=summary,
            output_path=build_output_path(output_path, args.role),
            metric=args.metric,
            role=args.role,
            annot=args.annot,
        )


if __name__ == "__main__":
    main()
