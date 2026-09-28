import os
import subprocess
import json
import time
import sys
from glob import glob
from project_paths import MAIN_PY, results_dir
#Original script for running all experiments
# ========= Configuration =========
layouts = [
    
    # "forced_coordination",
    "cramped_room",
    "coordination_ring",
    "counter_circuit",
    "asymmetric_advantages"
]

algs = ["SP","MEP","BC", "FCP", "PBT","COLE"  ]
# 
LLM = "llmtree"

BASE_DIR = str(results_dir("results/main_runs"))

PYTHON = sys.executable
MAIN = str(MAIN_PY)


# ========= Run command =========
def run_cmd(cmd):
    print("\n[RUN]", " ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - start
    print(f"[TIME] elapsed_seconds={elapsed:.3f}")
    return elapsed


def load_train_metrics(exp_dir):
    metrics_file = os.path.join(exp_dir, "llm_usage_summary.json")
    if not os.path.exists(metrics_file):
        return None
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read train metrics {metrics_file}: {e}")
        return None


# ========= Find latest experiment directory =========
def find_latest_exp(layout, p0, p1):
    # Update filename pattern to ensure files start with "exp"
    pattern = f"exp*layout_{layout}_p0-{p0}_p1-{p1}"
    candidates = glob(os.path.join(BASE_DIR, pattern))

    if not candidates:
        return None

    return sorted(candidates)[-1]


# ========= Find the first plan that reaches the highest score =========
def find_best_plan(exp_dir):
    plan_files = sorted(glob(os.path.join(exp_dir, "plan_*.json")))

    best_score = float("-inf")
    best_file = None

    for f in plan_files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)

            # Get the score position
            score = data["final_result"]["result"]["avg"]
            print(f"[INFO] {f} | score={score}")

            # First occurrence of the highest score
            if score > best_score:
                best_score = score
                best_file = f

        except Exception as e:
            print(f"[ERROR] reading {f}: {e}")

    if best_file:
        # Check whether the filename already ends with "_best.json"; if so, do not modify it
        if not best_file.endswith("_best.json"):
            # Add the "_best" suffix to the filename
            best_file_path = best_file.replace(".json", "_best.json")
            os.rename(best_file, best_file_path)
            print(f"[INFO] Renamed best plan file to: {best_file_path}")
        else:
            best_file_path = best_file
            print(f"[INFO] Best plan file already named {best_file_path}, no renaming needed.")

    return best_file_path, best_score


# ========= Find the last plan with the highest score =========
def find_final_best_plan(exp_dir):
    plan_files = sorted(glob(os.path.join(exp_dir, "plan_*.json")))

    final_best_score = float("-inf")
    final_best_file = None

    for f in plan_files:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)

            # Get the score position
            score = data["final_result"]["result"]["avg"]
            print(f"[INFO] {f} | score={score}")

            # Always update to the last occurrence of the highest score
            if score >= final_best_score:
                final_best_score = score
                final_best_file = f

        except Exception as e:
            print(f"[ERROR] reading {f}: {e}")

    if final_best_file:
        # Check whether filename ends with "_best_final.json"; if so, do not modify it
        if final_best_file.endswith("_best.json"):
            print(f"[INFO] final_best_plan is already best, skipping final test: {final_best_file}")
            return None, final_best_score

        if not final_best_file.endswith("_best_final.json"):
            # Add the "_best_final" suffix to the filename
            final_best_file_path = final_best_file.replace(".json", "_best_final.json")
            os.rename(final_best_file, final_best_file_path)
            print(f"[INFO] Renamed final best plan file to: {final_best_file_path}")
        else:
            final_best_file_path = final_best_file
            print(f"[INFO] Final best plan file already named {final_best_file_path}, no renaming needed.")

    return final_best_file_path, final_best_score


# ========= Main workflow =========
summary = []

for layout in layouts:
    for alg in algs:

        # Two training modes
        settings = [
            (LLM, alg),   # llmtree vs alg
            (alg, LLM)    # alg vs llmtree
        ]

        for p0, p1 in settings:
            train_elapsed_seconds = None
            test_elapsed_seconds = None
            test_final_elapsed_seconds = None

            print("\n==============================")
            print(f"Layout: {layout} | p0={p0}, p1={p1}")
            print("==============================")

            # ========= 1. Check whether an experiment directory already exists =========
            exp_dir = find_latest_exp(layout, p0, p1)
            if exp_dir:
                print(f"[INFO] Experiment already exists at: {exp_dir}, skipping training.")
            else:
                # ========= 1. Train =========
                cmd_train = [
                    PYTHON, MAIN,
                    "--layout", layout,
                    "--p0", p0,
                    "--p1", p1,
                    "--mode", "train"
                ]
                train_elapsed_seconds = run_cmd(cmd_train)

                # ========= Find experiment directory =========
                exp_dir = find_latest_exp(layout, p0, p1)

                if exp_dir is None:
                    print("[WARN] No experiment dir found!")
                    continue

                print(f"[INFO] exp_dir = {exp_dir}")
            train_metrics = load_train_metrics(exp_dir)
            train_usage = (train_metrics or {}).get("llm_usage", {})

            # ========= 2. Find best plan and final best plan =========
            best_plan, best_score = find_best_plan(exp_dir)
            final_best_plan, final_best_score = find_final_best_plan(exp_dir)

            if best_plan is None:
                print("[WARN] No valid plan found!")
                continue

            print(f"[BEST] {best_plan} | score={best_score}")
            print(f"[FINAL BEST] {final_best_plan} | score={final_best_score}")

            # ========= 3. Test =========
            train_alg = alg  # Current training partner

            cmd_test = [
                PYTHON, MAIN,
                "--layout", layout,
                "--p0", p0,
                "--p1", p1,
                "--mode", "test",
                "--plan_path", best_plan,
                "--test_alg", train_alg
            ]

            # Test if final_best_plan exists and is not best
            if final_best_plan:
                cmd_test_final = [
                    PYTHON, MAIN,
                    "--layout", layout,
                    "--p0", p0,
                    "--p1", p1,
                    "--mode", "test",
                    "--plan_path", final_best_plan,
                    "--test_alg", train_alg,
                    "--final"
                ]
                test_final_elapsed_seconds = run_cmd(cmd_test_final)

            test_elapsed_seconds = run_cmd(cmd_test)

            # ========= Record =========
            summary.append({
                "layout": layout,
                "p0": p0,
                "p1": p1,
                "train_alg": train_alg,
                "best_plan": best_plan,
                "best_score": best_score,
                "final_best_plan": final_best_plan,
                "final_best_score": final_best_score,
                "train_elapsed_seconds": train_elapsed_seconds,
                "train_metrics_file": os.path.join(exp_dir, "llm_usage_summary.json"),
                "train_total_tokens": train_usage.get("total_tokens"),
                "train_query_count": train_usage.get("query_count"),
                "train_successful_query_count": train_usage.get("successful_query_count"),
                "train_failed_query_count": train_usage.get("failed_query_count"),
                "test_elapsed_seconds": test_elapsed_seconds,
                "test_final_elapsed_seconds": test_final_elapsed_seconds
            })


# ========= Save =========
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ ALL DONE!")
