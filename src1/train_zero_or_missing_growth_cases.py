import argparse
import json
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import List, Optional
from project_paths import MAIN_PY, results_dir


BASE_RESULTS_DIR = str(results_dir("results/training_runs"))
DEFAULT_MAIN_PY = str(MAIN_PY)
DEFAULT_PYTHON_BIN = "python"
DEFAULT_SEED = 1

LAYOUT_ORDER = [
    "cramped_room",
    "coordination_ring",
    "counter_circuit",
    "asymmetric_advantages",
    "forced_coordination",
]
# "BC",
TEAMMATES = [ "COLE", "FCP", "MEP", "PBT", "SP"]

LLMTREE_ALIASES = {"llmtree", "cotree", "co-tree", "co_tree"}
EXP_PATTERN = re.compile(r"^exp_(\d{8}_\d{6})_layout_(.+?)_p0-(.+?)_p1-(.+)$")
PLAN_INDEX_PATTERN = re.compile(r"plan_(\d+)")


@dataclass
class TrialPoint:
    plan_idx: int
    avg: float
    std: Optional[float]
    source_file: str


@dataclass
class RunRecord:
    exp_name: str
    exp_path: str
    timestamp: str
    layout: str
    teammate: str
    llmtree_side: str
    points: List[TrialPoint]


@dataclass
class TrainTarget:
    layout: str
    teammate: str
    p0: str
    p1: str
    reason: str
    llmtree_side: str
    source_exp_name: Optional[str] = None
    delta: Optional[float] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-click train runs for teammate/layout cases whose llmtree-as-p0 and "
            "llmtree-as-p1 runs are both still zero-improvement. Optionally, teammate/"
            "layout pairs with no curve at all can be added too. Only train mode is used, "
            "and forced_coordination is excluded."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=BASE_RESULTS_DIR,
        help="Existing training-results directory used to inspect current curves.",
    )
    parser.add_argument(
        "--main-py",
        default=DEFAULT_MAIN_PY,
        help="Path to src1/main.py used for launching train mode.",
    )
    parser.add_argument(
        "--python-bin",
        default=DEFAULT_PYTHON_BIN,
        help="Python command used to launch training (default: python).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Seed used to mirror the current PDF's run selection when multiple runs "
            f"exist for one teammate (default: {DEFAULT_SEED})."
        ),
    )
    parser.add_argument(
        "--include-no-curve",
        action="store_true",
        help=(
            "Also include teammate/layout pairs with no curve at all. By default, "
            "the script only reruns teammate/layout pairs whose p0 and p1 sides are "
            "both still zero-improvement."
        ),
    )
    parser.add_argument(
        "--missing-side",
        choices=["both", "p0", "p1"],
        default="both",
        help=(
            "When --include-no-curve is enabled, choose whether to train both "
            "llmtree sides, only p0=llmtree, or only p1=llmtree."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many targets to run, in discovery order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered targets without launching any training command.",
    )
    parser.add_argument(
        "--print-summary-only",
        action="store_true",
        help=(
            "Only print the discovered targets and summary counts, then exit. "
            "This is like --dry-run but intended for quick inspection."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if one training command fails.",
    )
    parser.add_argument(
        "--plan-json",
        default=None,
        help="Optional path to save the discovered training targets as JSON.",
    )
    return parser.parse_args()


def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_agent_name(name):
    cleaned = name.strip()
    if cleaned.lower() in LLMTREE_ALIASES:
        return "llmtree"
    return cleaned


def extract_exp_info(folder_name):
    match = EXP_PATTERN.match(folder_name)
    if not match:
        return None
    timestamp, layout, p0, p1 = match.groups()
    return {
        "timestamp": timestamp,
        "layout": layout,
        "p0": normalize_agent_name(p0),
        "p1": normalize_agent_name(p1),
    }


def plan_file_priority(file_name):
    if file_name.endswith("_best_final.json"):
        return 2
    if file_name.endswith("_best.json"):
        return 1
    return 0


def extract_plan_index(file_name, data):
    if isinstance(data, dict) and "plan_idx" in data:
        return int(data["plan_idx"])
    match = PLAN_INDEX_PATTERN.search(file_name)
    if match:
        return int(match.group(1))
    return None


def collect_trial_points(exp_path):
    best_by_idx = {}

    for file_name in sorted(os.listdir(exp_path)):
        if not file_name.startswith("plan_") or not file_name.endswith(".json"):
            continue

        full_path = os.path.join(exp_path, file_name)
        data = safe_load_json(full_path)
        if not data:
            continue

        plan_idx = extract_plan_index(file_name, data)
        if plan_idx is None:
            continue

        result = data.get("final_result", {}).get("result", {})
        avg = result.get("avg")
        std = result.get("std")
        if avg is None:
            continue

        point = TrialPoint(
            plan_idx=plan_idx,
            avg=float(avg),
            std=float(std) if std is not None else None,
            source_file=file_name,
        )

        priority = plan_file_priority(file_name)
        current = best_by_idx.get(plan_idx)
        if current is None or priority > current["priority"]:
            best_by_idx[plan_idx] = {"priority": priority, "point": point}

    return [best_by_idx[idx]["point"] for idx in sorted(best_by_idx)]


def collect_runs(results_dir):
    runs_by_layout = defaultdict(lambda: defaultdict(list))

    for folder_name in sorted(os.listdir(results_dir)):
        exp_info = extract_exp_info(folder_name)
        if exp_info is None:
            continue
        if exp_info["layout"] not in LAYOUT_ORDER:
            continue

        p0 = exp_info["p0"]
        p1 = exp_info["p1"]
        if (p0 == "llmtree") == (p1 == "llmtree"):
            continue

        llmtree_side = "p0" if p0 == "llmtree" else "p1"
        teammate = p1 if p0 == "llmtree" else p0
        exp_path = os.path.join(results_dir, folder_name)
        points = collect_trial_points(exp_path)
        if len(points) < 2:
            continue

        runs_by_layout[exp_info["layout"]][teammate].append(
            RunRecord(
                exp_name=folder_name,
                exp_path=exp_path,
                timestamp=exp_info["timestamp"],
                layout=exp_info["layout"],
                teammate=teammate,
                llmtree_side=llmtree_side,
                points=points,
            )
        )

    return runs_by_layout


def make_non_decreasing_curve(points):
    accepted_avg = []
    accepted_std = []
    best_avg = None
    best_std = None

    for point in points:
        if best_avg is None or point.avg >= best_avg:
            best_avg = point.avg
            best_std = point.std
        accepted_avg.append(best_avg)
        accepted_std.append(best_std)

    return accepted_avg, accepted_std


def compute_curve_delta(record):
    accepted_avg, _ = make_non_decreasing_curve(record.points)
    return float(accepted_avg[-1] - accepted_avg[0])


def group_runs_by_side(runs_by_layout):
    grouped = defaultdict(lambda: defaultdict(lambda: {"p0": [], "p1": []}))

    for layout, teammate_map in runs_by_layout.items():
        for teammate, records in teammate_map.items():
            for record in records:
                grouped[layout][teammate][record.llmtree_side].append(record)

    return grouped


def records_all_zero(records):
    return bool(records) and all(abs(compute_curve_delta(record)) < 1e-9 for record in records)


def pick_reference_record(records):
    if not records:
        return None
    return max(records, key=lambda record: record.timestamp)


def build_both_sides_zero_targets(runs_by_layout):
    targets = []
    grouped = group_runs_by_side(runs_by_layout)

    for layout in LAYOUT_ORDER:
        for teammate, side_map in grouped.get(layout, {}).items():
            p0_records = side_map["p0"]
            p1_records = side_map["p1"]

            if not (records_all_zero(p0_records) and records_all_zero(p1_records)):
                continue

            for llmtree_side, p0, p1, records in [
                ("p0", "llmtree", teammate, p0_records),
                ("p1", teammate, "llmtree", p1_records),
            ]:
                reference_record = pick_reference_record(records)
                targets.append(
                    TrainTarget(
                        layout=layout,
                        teammate=teammate,
                        p0=p0,
                        p1=p1,
                        reason="both_sides_zero_for_teammate",
                        llmtree_side=llmtree_side,
                        source_exp_name=reference_record.exp_name if reference_record else None,
                        delta=compute_curve_delta(reference_record) if reference_record else None,
                    )
                )

    return targets


def build_missing_curve_targets(runs_by_layout, missing_side):
    targets = []

    for layout in LAYOUT_ORDER:
        teammate_map = runs_by_layout.get(layout, {})
        for teammate in TEAMMATES:
            if teammate in teammate_map:
                continue

            orientations = []
            if missing_side in {"both", "p0"}:
                orientations.append(("llmtree", teammate, "p0"))
            if missing_side in {"both", "p1"}:
                orientations.append((teammate, "llmtree", "p1"))

            for p0, p1, llmtree_side in orientations:
                targets.append(
                    TrainTarget(
                        layout=layout,
                        teammate=teammate,
                        p0=p0,
                        p1=p1,
                        reason="no_curve_in_current_pdf",
                        llmtree_side=llmtree_side,
                    )
                )

    return targets


def dedupe_targets(targets):
    deduped = []
    seen = set()

    for target in targets:
        key = (target.layout, target.p0, target.p1)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)

    return deduped


def maybe_save_plan_json(plan_json_path, targets):
    if not plan_json_path:
        return

    payload = [asdict(target) for target in targets]
    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved training plan to: {plan_json_path}")


def print_targets(targets):
    if not targets:
        print("[INFO] No targets discovered.")
        return

    print("[INFO] Training targets:")
    for idx, target in enumerate(targets, start=1):
        delta_text = ""
        if target.delta is not None:
            delta_text = f" | delta={target.delta:.1f}"
        source_text = ""
        if target.source_exp_name:
            source_text = f" | source={target.source_exp_name}"
        print(
            f"{idx:02d}. layout={target.layout} | p0={target.p0} | p1={target.p1} "
            f"| reason={target.reason}{delta_text}{source_text}"
        )

    both_sides_zero_count = sum(
        1 for target in targets if target.reason == "both_sides_zero_for_teammate"
    )
    missing_count = sum(1 for target in targets if target.reason == "no_curve_in_current_pdf")
    print(
        f"[INFO] Summary: total={len(targets)} "
        f"both_sides_zero={both_sides_zero_count} no_curve={missing_count}"
    )


def build_train_command(target, args):
    return [
        args.python_bin,
        args.main_py,
        "--layout",
        target.layout,
        "--p0",
        target.p0,
        "--p1",
        target.p1,
        "--mode",
        "train",
    ]


def run_targets(targets, args):
    results = []

    for idx, target in enumerate(targets, start=1):
        cmd = build_train_command(target, args)
        print("")
        print(f"[RUN {idx}/{len(targets)}] {' '.join(cmd)}")
        start = time.perf_counter()
        status = "ok"
        error_text = None

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error_text = str(exc)
            elapsed = time.perf_counter() - start
            results.append(
                {
                    **asdict(target),
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "error": error_text,
                }
            )
            print(f"[ERROR] status=failed elapsed_seconds={elapsed:.3f} error={error_text}")
            if not args.continue_on_error:
                break
            continue

        elapsed = time.perf_counter() - start
        print(f"[DONE] status=ok elapsed_seconds={elapsed:.3f}")
        results.append(
            {
                **asdict(target),
                "status": status,
                "elapsed_seconds": elapsed,
                "error": error_text,
            }
        )

    return results


def main():
    args = parse_args()

    runs_by_layout = collect_runs(args.results_dir)
    targets = build_both_sides_zero_targets(runs_by_layout)
    if args.include_no_curve:
        targets.extend(build_missing_curve_targets(runs_by_layout, args.missing_side))
    targets = dedupe_targets(targets)

    if args.limit is not None:
        targets = targets[: args.limit]

    print_targets(targets)
    maybe_save_plan_json(args.plan_json, targets)

    if args.dry_run or args.print_summary_only:
        print("[INFO] Dry run only, no training command launched.")
        return

    results = run_targets(targets, args)
    ok_count = sum(1 for item in results if item["status"] == "ok")
    fail_count = sum(1 for item in results if item["status"] == "failed")
    print("")
    print(f"[SUMMARY] finished={len(results)} ok={ok_count} failed={fail_count}")


if __name__ == "__main__":
    main()
