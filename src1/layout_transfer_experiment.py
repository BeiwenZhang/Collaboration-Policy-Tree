import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from queue import Empty
from multiprocessing import Process, Queue
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "results/layout_transfer"
DEFAULT_PARTNER_POOL = ["FCP", "SP", "PBT", "COLE", "MEP", "BC"]
POLICY_DIR = PROJECT_DIR / "src1" / "policy"
DEFAULT_PARTNER_TIMEOUT_SECONDS = 900
LAYOUT_ORDER = [
    "cramped_room",
    "coordination_ring",
    "counter_circuit",
    "asymmetric_advantages",
    "forced_coordination",
]
LAYOUT_LABELS = {
    "cramped_room": "Cramped Rm.",
    "coordination_ring": "Coord. Ring",
    "counter_circuit": "CT. Circuit",
    "asymmetric_advantages": "Asymm. Adv.",
    "forced_coordination": "Forced Coord.",
}
LLMTREE_ALIASES = {"llmtree", "cotree", "co-tree", "co_tree"}
EXP_PATTERN = re.compile(r"^exp_(\d{8}_\d{6})_layout_(.+?)_p0-(.+?)_p1-(.+)$")


@dataclass
class SourcePlan:
    plan_path: str
    source_layout: str
    source_partner: str
    llmtree_side: str
    tag: str
    source_exp_name: Optional[str]
    source_timestamp: Optional[str]
    plan_stem: str
    plan_hash: str


def get_llmtree_policy_file(layout: str, agent_index: int) -> Path:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    return POLICY_DIR / f"{layout}_{agent_index}.txt"


def load_policy_code(policy_source_path: str) -> str:
    path = Path(policy_source_path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".json":
            plan_record = json.load(f)
            final_code = plan_record.get("final_code")
            if final_code is None:
                raise ValueError(f"No final_code found in {policy_source_path}")
            return final_code
        return f.read()


def stage_llmtree_policy(layout: str, agent_index: int, policy_source_path: str):
    policy_file = get_llmtree_policy_file(layout, agent_index)
    existed = policy_file.exists()
    old_code = None
    if existed:
        old_code = policy_file.read_text(encoding="utf-8")

    new_code = load_policy_code(policy_source_path)
    policy_file.write_text(new_code, encoding="utf-8")
    return policy_file, existed, old_code


def restore_llmtree_policy(staged_policy):
    if staged_policy is None:
        return

    policy_file, existed, old_code = staged_policy
    if existed:
        policy_file.write_text(old_code, encoding="utf-8")
    elif policy_file.exists():
        policy_file.unlink()


def _test_wrapper(q: Queue, args):
    try:
        os.chdir(PROJECT_DIR)
        from test_Overcooked.test_code_multi import test

        out = test(args)
        q.put(("result", out))
    except Exception:
        q.put(("error", traceback.format_exc()))


def run_test_in_subprocess(args):
    start_time = time.time()
    start_perf = time.perf_counter()
    q = Queue()
    p = Process(target=_test_wrapper, args=(q, args))
    p.start()
    timeout_seconds = float(getattr(args, "partner_timeout_seconds", DEFAULT_PARTNER_TIMEOUT_SECONDS))
    try:
        tag, data = q.get(timeout=timeout_seconds)
    except Empty:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        elapsed = time.perf_counter() - start_perf
        return "timeout", {
            "message": (
                f"Subprocess exceeded timeout after {timeout_seconds:.0f} seconds "
                f"for layout={getattr(args, 'layout', '?')}, "
                f"p0={getattr(args, 'p0', '?')}, p1={getattr(args, 'p1', '?')}"
            ),
            "test_elapsed_seconds": elapsed,
            "test_started_at": datetime.fromtimestamp(start_time).isoformat(
                timespec="seconds"
            ),
            "test_finished_at": datetime.now().isoformat(timespec="seconds"),
        }

    p.join(timeout=5)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)

    elapsed = time.perf_counter() - start_perf
    if isinstance(data, dict):
        data["test_elapsed_seconds"] = elapsed
        data["test_started_at"] = datetime.fromtimestamp(start_time).isoformat(
            timespec="seconds"
        )
        data["test_finished_at"] = datetime.now().isoformat(timespec="seconds")
    return tag, data


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-layout transfer evaluation for saved llmtree plans and "
            "save raw plus aggregated results."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file. See layout_transfer_config_template.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve plans and print the evaluation schedule without running tests.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run evaluations even if the per-run JSON already exists.",
    )
    parser.add_argument(
        "--partner-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Optional override for per-partner evaluation timeout. "
            "Defaults to the config value or 900 seconds."
        ),
    )
    return parser.parse_args()


def normalize_agent_name(name: str) -> str:
    cleaned = str(name).strip()
    if cleaned.lower() in LLMTREE_ALIASES:
        return "llmtree"
    return cleaned


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(mean(values))


def safe_std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    return float(pstdev(values))


def pretty_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_plan_metadata_from_path(plan_path: Path) -> Dict[str, Optional[str]]:
    exp_name = plan_path.parent.name
    match = EXP_PATTERN.match(exp_name)
    if not match:
        return {
            "source_exp_name": exp_name,
            "source_timestamp": None,
            "source_layout": None,
            "source_partner": None,
            "llmtree_side": None,
        }

    timestamp, layout, p0, p1 = match.groups()
    p0 = normalize_agent_name(p0)
    p1 = normalize_agent_name(p1)
    if (p0 == "llmtree") == (p1 == "llmtree"):
        raise ValueError(
            f"Expected exactly one llmtree side in experiment folder {exp_name!r}, "
            f"got p0={p0!r}, p1={p1!r}."
        )

    llmtree_side = "p0" if p0 == "llmtree" else "p1"
    source_partner = p1 if llmtree_side == "p0" else p0
    return {
        "source_exp_name": exp_name,
        "source_timestamp": timestamp,
        "source_layout": layout,
        "source_partner": source_partner,
        "llmtree_side": llmtree_side,
    }


def resolve_source_plan(entry: Dict[str, str]) -> SourcePlan:
    plan_path = Path(entry["plan_path"]).expanduser().resolve()
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")

    parsed = parse_plan_metadata_from_path(plan_path)
    source_layout = entry.get("source_layout") or parsed["source_layout"]
    source_partner = entry.get("source_partner") or parsed["source_partner"]
    llmtree_side = entry.get("llmtree_side") or parsed["llmtree_side"]

    if source_layout is None or source_partner is None or llmtree_side is None:
        raise ValueError(
            "Could not infer source_layout/source_partner/llmtree_side from the "
            f"path {plan_path}. Please specify them explicitly in the config."
        )

    if source_layout not in LAYOUT_ORDER:
        raise ValueError(f"Unsupported source layout: {source_layout}")
    if llmtree_side not in {"p0", "p1"}:
        raise ValueError(f"llmtree_side must be 'p0' or 'p1', got {llmtree_side!r}")

    plan_hash = hashlib.sha1(str(plan_path).encode("utf-8")).hexdigest()[:8]
    default_tag = (
        f"{source_layout}__{llmtree_side}__train_{source_partner}__"
        f"{plan_path.stem}__{plan_hash}"
    )
    return SourcePlan(
        plan_path=str(plan_path),
        source_layout=source_layout,
        source_partner=source_partner,
        llmtree_side=llmtree_side,
        tag=entry.get("tag") or default_tag,
        source_exp_name=entry.get("source_exp_name") or parsed["source_exp_name"],
        source_timestamp=entry.get("source_timestamp") or parsed["source_timestamp"],
        plan_stem=plan_path.stem,
        plan_hash=plan_hash,
    )


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if "plan_entries" not in config or not config["plan_entries"]:
        raise ValueError("Config must contain a non-empty 'plan_entries' list.")

    config = deepcopy(config)
    config.setdefault("experiment_name", config_path.stem)
    config.setdefault("target_layouts", LAYOUT_ORDER)
    config.setdefault("partner_pool", DEFAULT_PARTNER_POOL)
    config.setdefault("exclude_source_partner_from_eval", True)
    config.setdefault("test_times", 3)
    config.setdefault("horizon", 400)
    config.setdefault("save_replay", False)
    config.setdefault("replay_episode", "worst")
    config.setdefault("replay_fps", 6)
    config.setdefault("replay_tile_size", 72)
    config.setdefault("replay_frames", False)
    config.setdefault("replay_gif", False)
    config.setdefault("partner_timeout_seconds", DEFAULT_PARTNER_TIMEOUT_SECONDS)
    return config


def build_eval_partners(config: Dict, source_partner: str) -> List[str]:
    partners = [normalize_agent_name(p) for p in config["partner_pool"]]
    partners = [p for p in partners if p != "llmtree"]
    if config.get("exclude_source_partner_from_eval", True):
        partners = [p for p in partners if p != source_partner]
    return partners


def make_test_args(
    target_layout: str,
    llmtree_side: str,
    partner: str,
    config: Dict,
    replay_dir: Optional[str] = None,
):
    if llmtree_side == "p0":
        p0, p1 = "llmtree", partner
    else:
        p0, p1 = partner, "llmtree"

    return SimpleNamespace(
        layout=target_layout,
        p0=p0,
        p1=p1,
        horizon=int(config["horizon"]),
        test_times=int(config["test_times"]),
        partner_timeout_seconds=float(config["partner_timeout_seconds"]),
        save_replay=bool(config.get("save_replay", False)),
        replay_dir=replay_dir,
        replay_episode=config.get("replay_episode", "worst"),
        replay_fps=int(config.get("replay_fps", 6)),
        replay_tile_size=int(config.get("replay_tile_size", 72)),
        replay_frames=bool(config.get("replay_frames", False)),
        replay_gif=bool(config.get("replay_gif", False)),
    )


def parse_partner_result(tag: str, data) -> Dict:
    if tag == "error":
        raise RuntimeError(data)

    if tag == "timeout":
        return {
            "scores": [],
            "avg": None,
            "std": None,
            "worst_score": None,
            "test_elapsed_seconds": data.get("test_elapsed_seconds") if isinstance(data, dict) else None,
            "test_started_at": data.get("test_started_at") if isinstance(data, dict) else None,
            "test_finished_at": data.get("test_finished_at") if isinstance(data, dict) else None,
            "raw_result": data,
            "tag": tag,
        }

    result_dict = data.get("result", {}) if isinstance(data, dict) else {}
    scores = result_dict.get("scores", [])
    return {
        "scores": scores,
        "avg": float(result_dict.get("avg", 0.0)),
        "std": float(result_dict.get("std", 0.0)),
        "worst_score": result_dict.get("worst_score"),
        "test_elapsed_seconds": data.get("test_elapsed_seconds") if isinstance(data, dict) else None,
        "test_started_at": data.get("test_started_at") if isinstance(data, dict) else None,
        "test_finished_at": data.get("test_finished_at") if isinstance(data, dict) else None,
        "raw_result": data,
        "tag": tag,
    }


def evaluate_source_plan_on_target(
    source_plan: SourcePlan,
    target_layout: str,
    config: Dict,
    output_dir: Path,
    resume: bool,
):
    run_file = (
        output_dir
        / "evaluations"
        / source_plan.tag
        / f"to_{target_layout}.json"
    )
    if resume and run_file.exists():
        with run_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    eval_partners = build_eval_partners(config, source_plan.source_partner)
    if not eval_partners:
        raise ValueError(
            f"No evaluation partners remain for source partner {source_plan.source_partner!r}. "
            "Adjust partner_pool or exclude_source_partner_from_eval."
        )

    agent_index = 0 if source_plan.llmtree_side == "p0" else 1
    staged_policy = stage_llmtree_policy(
        target_layout,
        agent_index,
        source_plan.plan_path,
    )

    partner_results = {}
    try:
        for partner in eval_partners:
            print(
                f"    -> partner={partner} | target_layout={target_layout} | "
                f"source_tag={source_plan.tag}"
            )
            args = make_test_args(
                target_layout=target_layout,
                llmtree_side=source_plan.llmtree_side,
                partner=partner,
                config=config,
                replay_dir=None,
            )
            tag, data = run_test_in_subprocess(args)
            partner_results[partner] = parse_partner_result(tag, data)
            partner_record = partner_results[partner]
            print(
                f"       finished tag={tag} | avg={partner_record['avg']} | "
                f"elapsed={partner_record['test_elapsed_seconds']}"
            )
    finally:
        restore_llmtree_policy(staged_policy)

    partner_avgs = [
        entry["avg"] for entry in partner_results.values() if entry["avg"] is not None
    ]
    partner_stds = [
        entry["std"] for entry in partner_results.values() if entry["std"] is not None
    ]
    total_test_elapsed = sum(
        entry.get("test_elapsed_seconds") or 0.0 for entry in partner_results.values()
    )

    record = {
        "source_plan": {
            "tag": source_plan.tag,
            "plan_path": source_plan.plan_path,
            "plan_stem": source_plan.plan_stem,
            "plan_hash": source_plan.plan_hash,
            "source_layout": source_plan.source_layout,
            "source_partner": source_plan.source_partner,
            "llmtree_side": source_plan.llmtree_side,
            "source_exp_name": source_plan.source_exp_name,
            "source_timestamp": source_plan.source_timestamp,
        },
        "target_layout": target_layout,
        "evaluation_partners": eval_partners,
        "partner_results": partner_results,
        "overall": {
            "mean_partner_avg": safe_mean(partner_avgs),
            "std_partner_avg": safe_std(partner_avgs),
            "mean_partner_std": safe_mean(partner_stds),
            "n_partners": len(eval_partners),
            "total_test_elapsed_seconds": float(total_test_elapsed),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    pretty_json(record, run_file)
    return record


def init_matrix(layouts: List[str]):
    return {src: {tgt: None for tgt in layouts} for src in layouts}


def build_summary(config: Dict, resolved_plans: List[SourcePlan], run_records: List[Dict]) -> Dict:
    layouts = list(config["target_layouts"])
    grouped = defaultdict(list)
    for record in run_records:
        key = (record["source_plan"]["source_layout"], record["target_layout"])
        grouped[key].append(record)

    cells = []
    combined_reward = init_matrix(layouts)
    combined_retention = init_matrix(layouts)
    p0_reward = init_matrix(layouts)
    p1_reward = init_matrix(layouts)
    p0_retention = init_matrix(layouts)
    p1_retention = init_matrix(layouts)

    for source_layout in layouts:
        for target_layout in layouts:
            runs = grouped.get((source_layout, target_layout), [])
            combined_values = [
                run["overall"]["mean_partner_avg"]
                for run in runs
                if run["overall"]["mean_partner_avg"] is not None
            ]
            role_values = {"p0": [], "p1": []}
            partner_values = defaultdict(list)

            for run in runs:
                llmtree_side = run["source_plan"]["llmtree_side"]
                overall_mean = run["overall"]["mean_partner_avg"]
                if overall_mean is not None:
                    role_values[llmtree_side].append(overall_mean)
                for partner, partner_record in run["partner_results"].items():
                    partner_values[partner].append(partner_record["avg"])

            cell = {
                "source_layout": source_layout,
                "target_layout": target_layout,
                "n_source_plans": len(runs),
                "combined": {
                    "mean_reward": safe_mean(combined_values),
                    "std_over_source_plans": safe_std(combined_values),
                },
                "by_role": {
                    "p0": {
                        "mean_reward": safe_mean(role_values["p0"]),
                        "std_over_source_plans": safe_std(role_values["p0"]),
                        "n_source_plans": len(role_values["p0"]),
                    },
                    "p1": {
                        "mean_reward": safe_mean(role_values["p1"]),
                        "std_over_source_plans": safe_std(role_values["p1"]),
                        "n_source_plans": len(role_values["p1"]),
                    },
                },
                "per_partner": {
                    partner: {
                        "mean_reward": safe_mean(values),
                        "std_over_source_plans": safe_std(values),
                        "n_source_plans": len(values),
                    }
                    for partner, values in sorted(partner_values.items())
                },
            }
            cells.append(cell)

            combined_reward[source_layout][target_layout] = cell["combined"]["mean_reward"]
            p0_reward[source_layout][target_layout] = cell["by_role"]["p0"]["mean_reward"]
            p1_reward[source_layout][target_layout] = cell["by_role"]["p1"]["mean_reward"]

    diag_combined = {
        layout: combined_reward[layout][layout] for layout in layouts
    }
    diag_p0 = {
        layout: p0_reward[layout][layout] for layout in layouts
    }
    diag_p1 = {
        layout: p1_reward[layout][layout] for layout in layouts
    }

    for cell in cells:
        source_layout = cell["source_layout"]
        target_layout = cell["target_layout"]

        target_diag = diag_combined[target_layout]
        value = cell["combined"]["mean_reward"]
        retention = None
        if value is not None and target_diag not in (None, 0):
            retention = float(value / target_diag)
        cell["combined"]["retention_vs_target_diag"] = retention
        combined_retention[source_layout][target_layout] = retention

        for role, diag_map, matrix in [
            ("p0", diag_p0, p0_retention),
            ("p1", diag_p1, p1_retention),
        ]:
            role_value = cell["by_role"][role]["mean_reward"]
            role_diag = diag_map[target_layout]
            role_retention = None
            if role_value is not None and role_diag not in (None, 0):
                role_retention = float(role_value / role_diag)
            cell["by_role"][role]["retention_vs_target_diag"] = role_retention
            matrix[source_layout][target_layout] = role_retention

    return {
        "layout_order": layouts,
        "layout_labels": {layout: LAYOUT_LABELS[layout] for layout in layouts},
        "partner_pool": build_eval_partners(
            config,
            source_partner="__never_used__",
        ),
        "n_source_plans": len(resolved_plans),
        "diagonal_reference": {
            "combined_mean_reward": diag_combined,
            "p0_mean_reward": diag_p0,
            "p1_mean_reward": diag_p1,
        },
        "cells": cells,
        "matrices": {
            "combined_mean_reward": combined_reward,
            "combined_retention": combined_retention,
            "p0_mean_reward": p0_reward,
            "p0_retention": p0_retention,
            "p1_mean_reward": p1_reward,
            "p1_retention": p1_retention,
        },
    }


def write_csvs(output_dir: Path, run_records: List[Dict], summary: Dict):
    plan_rows = []
    for record in run_records:
        source = record["source_plan"]
        for partner, partner_result in record["partner_results"].items():
            plan_rows.append(
                {
                    "source_tag": source["tag"],
                    "source_layout": source["source_layout"],
                    "target_layout": record["target_layout"],
                    "llmtree_side": source["llmtree_side"],
                    "source_partner": source["source_partner"],
                    "eval_partner": partner,
                    "avg_reward": partner_result["avg"],
                    "std_reward": partner_result["std"],
                    "scores_json": json.dumps(partner_result["scores"]),
                    "test_elapsed_seconds": partner_result["test_elapsed_seconds"],
                }
            )

    plan_csv = output_dir / "plan_partner_results.csv"
    with plan_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_tag",
                "source_layout",
                "target_layout",
                "llmtree_side",
                "source_partner",
                "eval_partner",
                "avg_reward",
                "std_reward",
                "scores_json",
                "test_elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(plan_rows)

    cell_rows = []
    for cell in summary["cells"]:
        cell_rows.append(
            {
                "source_layout": cell["source_layout"],
                "target_layout": cell["target_layout"],
                "n_source_plans": cell["n_source_plans"],
                "combined_mean_reward": cell["combined"]["mean_reward"],
                "combined_std_over_source_plans": cell["combined"]["std_over_source_plans"],
                "combined_retention_vs_target_diag": cell["combined"]["retention_vs_target_diag"],
                "p0_mean_reward": cell["by_role"]["p0"]["mean_reward"],
                "p0_retention_vs_target_diag": cell["by_role"]["p0"]["retention_vs_target_diag"],
                "p1_mean_reward": cell["by_role"]["p1"]["mean_reward"],
                "p1_retention_vs_target_diag": cell["by_role"]["p1"]["retention_vs_target_diag"],
            }
        )

    cell_csv = output_dir / "cell_summary.csv"
    with cell_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_layout",
                "target_layout",
                "n_source_plans",
                "combined_mean_reward",
                "combined_std_over_source_plans",
                "combined_retention_vs_target_diag",
                "p0_mean_reward",
                "p0_retention_vs_target_diag",
                "p1_mean_reward",
                "p1_retention_vs_target_diag",
            ],
        )
        writer.writeheader()
        writer.writerows(cell_rows)


def print_schedule(resolved_plans: List[SourcePlan], config: Dict):
    print("\nResolved source plans:")
    for plan in resolved_plans:
        partners = build_eval_partners(config, plan.source_partner)
        print(
            f"  - {plan.tag}\n"
            f"    plan={plan.plan_path}\n"
            f"    source_layout={plan.source_layout}, llmtree_side={plan.llmtree_side}, "
            f"source_partner={plan.source_partner}, eval_partners={partners}"
        )

    print("\nTransfer schedule:")
    for plan in resolved_plans:
        for target_layout in config["target_layouts"]:
            print(
                f"  {plan.source_layout} -> {target_layout} | "
                f"side={plan.llmtree_side} | source_partner={plan.source_partner}"
            )


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if args.partner_timeout_seconds is not None:
        config["partner_timeout_seconds"] = float(args.partner_timeout_seconds)
    for layout in config["target_layouts"]:
        if layout not in LAYOUT_ORDER:
            raise ValueError(f"Unsupported target layout: {layout}")

    resolved_plans = [resolve_source_plan(entry) for entry in config["plan_entries"]]
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / config["experiment_name"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        "config_path": str(config_path),
        "experiment_name": config["experiment_name"],
        "target_layouts": config["target_layouts"],
        "partner_pool": config["partner_pool"],
        "exclude_source_partner_from_eval": config["exclude_source_partner_from_eval"],
        "test_times": config["test_times"],
        "horizon": config["horizon"],
        "partner_timeout_seconds": config["partner_timeout_seconds"],
        "save_replay": config["save_replay"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "resolved_plan_entries": [plan.__dict__ for plan in resolved_plans],
    }
    pretty_json(resolved_config, output_dir / "resolved_config.json")

    print_schedule(resolved_plans, config)
    if args.dry_run:
        print("\nDry run only. No evaluations were executed.")
        return

    run_records = []
    resume = not args.no_resume
    total_jobs = len(resolved_plans) * len(config["target_layouts"])
    job_index = 0

    for source_plan in resolved_plans:
        for target_layout in config["target_layouts"]:
            job_index += 1
            print(
                f"\n[{job_index}/{total_jobs}] Evaluating {source_plan.tag} "
                f"on target layout {target_layout}"
            )
            record = evaluate_source_plan_on_target(
                source_plan=source_plan,
                target_layout=target_layout,
                config=config,
                output_dir=output_dir,
                resume=resume,
            )
            run_records.append(record)
            overall = record["overall"]
            print(
                f"  mean_partner_avg={overall['mean_partner_avg']:.3f} "
                f"| n_partners={overall['n_partners']}"
            )

    summary = build_summary(config, resolved_plans, run_records)
    result_bundle = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": resolved_config,
        "run_count": len(run_records),
        "summary": summary,
    }
    pretty_json(result_bundle, output_dir / "summary.json")
    write_csvs(output_dir, run_records, summary)

    print(f"\nSaved transfer experiment summary to {output_dir / 'summary.json'}")
    print(f"Saved per-plan CSV to {output_dir / 'plan_partner_results.csv'}")
    print(f"Saved cell summary CSV to {output_dir / 'cell_summary.csv'}")


if __name__ == "__main__":
    main()
