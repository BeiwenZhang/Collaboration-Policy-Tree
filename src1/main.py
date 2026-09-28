import os
from argparse import ArgumentParser
import traceback
from multiprocessing import Process, Queue
from queue import Empty
from api.LLMCoder import LLMCoder
from api.LLMPlanner import LLMPlanner
from api.LLMSummarizer import LLMSummarizer
from LLM.call_llm_api.call_llm import main_logger
from test_Overcooked.test_code_multi import test
import random
from datetime import datetime
import json
import time
from rich import print
from metrics import get_llm_metrics, reset_llm_metrics, save_llm_metrics
from project_paths import POLICY_DIR, REPLAY_DIR, results_dir


#Version with low-score rejection mechanism
TEST_RESULTS_DIR = str(results_dir("results/main_runs"))
EXISTING_PLAN_RESULTS_DIR = str(results_dir("results/existing_plan_runs"))
AI_PARTNER_ALGS = ("FCP", "SP", "PBT", "COLE", "MEP", "BC")


def get_llmtree_policy_file(layout, agent_index):
    policy_dir = str(POLICY_DIR)
    os.makedirs(policy_dir, exist_ok=True)
    return os.path.join(policy_dir, f"{layout}_{agent_index}.txt")


def load_policy_code(policy_source_path):
    if policy_source_path is None:
        return None

    with open(policy_source_path, "r", encoding="utf-8") as f:
        if policy_source_path.endswith(".json"):
            plan_record = json.load(f)
            final_code = plan_record.get("final_code")
            if final_code is None:
                raise ValueError(f"No final_code found in {policy_source_path}")
            return final_code
        return f.read()


def stage_llmtree_policy(layout, agent_index, policy_source_path):
    if policy_source_path is None:
        return None

    policy_file = get_llmtree_policy_file(layout, agent_index)
    existed = os.path.exists(policy_file)
    old_code = None
    if existed:
        with open(policy_file, "r", encoding="utf-8") as f:
            old_code = f.read()

    new_code = load_policy_code(policy_source_path)
    with open(policy_file, "w", encoding="utf-8") as f:
        f.write(new_code)

    return policy_file, existed, old_code


def restore_llmtree_policy(staged_policy):
    if staged_policy is None:
        return

    policy_file, existed, old_code = staged_policy
    if existed:
        with open(policy_file, "w", encoding="utf-8") as f:
            f.write(old_code)
    elif os.path.exists(policy_file):
        os.remove(policy_file)


def test_llmtree_duel(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(
        TEST_RESULTS_DIR,
        f"test_{timestamp}_layout_{args.layout}_p0-llmtree_p1-llmtree_duel"
    )
    os.makedirs(result_dir, exist_ok=True)

    staged_policies = []
    original_replay_dir = args.replay_dir

    try:
        staged_policies.append(stage_llmtree_policy(args.layout, 0, args.p0_policy_path))
        staged_policies.append(stage_llmtree_policy(args.layout, 1, args.p1_policy_path))

        if args.save_replay and args.replay_dir is None:
            args.replay_dir = os.path.join(result_dir, "replay")

        tag, data = run_test_in_subprocess(args)
    finally:
        args.replay_dir = original_replay_dir
        for staged_policy in reversed(staged_policies):
            restore_llmtree_policy(staged_policy)

    result_record = {
        "layout": args.layout,
        "p0": args.p0,
        "p1": args.p1,
        "p0_policy_path": args.p0_policy_path,
        "p1_policy_path": args.p1_policy_path,
        "policy_file_p0": get_llmtree_policy_file(args.layout, 0),
        "policy_file_p1": get_llmtree_policy_file(args.layout, 1),
        "tag": tag,
        "raw_result": data,
    }

    summary_file = os.path.join(result_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(result_record, f, ensure_ascii=False, indent=2)

    if tag != "result":
        raise RuntimeError(f"llmtree duel failed; see {summary_file}")

    print(f"✅ llmtree duel result saved to: {summary_file}")
    return result_record


def _test_wrapper(q, args):
    try:
        # print("print:",args.asyn_episode)
        # print("run.asyn_0:",args.asyn_0)
        # print("run.asyn_epsilon:",args.asyn_epsilon)
        # print("run.asyn_1:",args.asyn_1)
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
    while True:
        try:
            tag, data = q.get(timeout=1)
            break
        except Empty:
            if not p.is_alive():
                tag = "error"
                data = f"evaluation subprocess exited with code {p.exitcode}"
                break
    p.join(timeout=5)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
    elapsed = time.perf_counter() - start_perf
    if isinstance(data, dict):
        data["test_elapsed_seconds"] = elapsed
        data["test_started_at"] = datetime.fromtimestamp(start_time).isoformat(timespec="seconds")
        data["test_finished_at"] = datetime.now().isoformat(timespec="seconds")
    return tag, data

def run(args):
    from datetime import datetime
    import os, json
    import numpy as np
    reset_llm_metrics()
    train_start_time = time.time()
    train_start_perf = time.perf_counter()

    # ==== 1. Create experiment directory ====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(
        TEST_RESULTS_DIR,
        f"exp_{timestamp}_layout_{args.layout}_p0-{args.p0}_p1-{args.p1}"
    )
    os.makedirs(exp_dir, exist_ok=True)
    metrics_file = os.path.join(exp_dir, "llm_usage_summary.json")

    def save_train_metrics(status, plan_idx=None):
        metadata = {
            "status": status,
            "layout": args.layout,
            "p0": args.p0,
            "p1": args.p1,
            "mode": args.mode,
            "plan_times": plan_times,
            "generate_times": generate_times,
            "started_at": datetime.fromtimestamp(train_start_time).isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - train_start_perf,
        }
        if plan_idx is not None:
            metadata["last_plan_idx"] = plan_idx
        save_llm_metrics(metrics_file, metadata)

    planner = LLMPlanner(args.layout, 0 if args.p0 == "llmtree" else 1)
    plan_times = 10
    generate_times = 10
    save_train_metrics("started")

    # ==== 2. Global state ====
    last_tactics = None
    last_score = None
    last_data = None
    promotion = None
    stuck_steps = 0

    def save_progress(path, record):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    # ==== 3. Main loop ====
    for plan_idx in range(plan_times):

        print(f"\n===== Plan Iteration {plan_idx} =====")

        plan_file = os.path.join(exp_dir, f"plan_{plan_idx:02d}.json")
        plan_record = {
            "plan_idx": plan_idx,
            "selected_teammate": args.p1 if args.p0 == "llmtree" else args.p0,
            "tactics": None,
            "final_code": None,
            "final_result": None,
            "scores": None,
            "avg_score": None,
            "std": None,
            "promotion": None,
            "stuck_steps": stuck_steps,
            "llm_usage_after_plan": None,
        }

        # ==== 3.1 Planner ====
        if last_tactics is None:
            print("▶ First iteration: generate initial tactics!planner!")
            tactics = planner.plan(
                None,
                None,
                args.planner_merge,
                use_last=False,
                stuck_steps=stuck_steps
            )
        else:
            print("▶ Generate new tactics based on last accepted tactics!planner!")
            tactics = planner.plan(
                last_tactics,
                promotion,
                args.planner_merge,
                use_last=True,
                stuck_steps=stuck_steps
            )

        plan_record["tactics"] = tactics
        save_progress(plan_file, plan_record)

        # ==== 3.2 Generating + Testing ====
        print("▶ Generating code!coder!")
        coder = LLMCoder(args.layout, 0 if args.p0 == "llmtree" else 1)
        tag, data, code = '', {}, ''

        for _ in range(generate_times):
            code = coder.generate_code(tactics, code, data)
            tag, data = run_test_in_subprocess(args)

            if tag == "result":
                plan_record["final_code"] = code
                plan_record["final_result"] = data
                # print("data!:",data)
                break
            if _ == generate_times - 1:
                print("❌ Failed to generate code, resample promotion")
                promotion = None

        if tag != "result" or not isinstance(data, dict):
            plan_record["generation_error"] = data
            plan_record["llm_usage_after_plan"] = get_llm_metrics()
            save_progress(plan_file, plan_record)
            save_train_metrics("failed", plan_idx)
            raise RuntimeError(
                f"Failed to generate an executable policy after {generate_times} attempts; "
                f"see {plan_file}"
            )

        save_progress(plan_file, plan_record)
        print("data!:",data)
        # ==== 3.3 Score Check ====
        result_dict = data.get("result", {})

        scores = result_dict.get("scores", [])
        cur_score = float(result_dict.get("avg", 0))   # Use avg as the optimization objective
        cur_std = float(result_dict.get("std", 0))

        print(f"▶ Current score: {cur_score}, std: {cur_std}, scores: {scores}")

        # 👉 Store explicitly for later analysis


        summarizer = LLMSummarizer(
            args.layout,
            0 if args.p0 == "llmtree" else 1
        )
        print("▶ Summarizing!summarizer!")

        # ==== 3.4 Reject (rollback)====
        if last_score is not None and cur_score < last_score:
            print(
                f"❌ Score dropped ({cur_score} < {last_score}), "
                "rollback tactics but resample promotion"
            )

            stuck_steps += 1
            plan_record["stuck_steps"] = stuck_steps

            promotion = summarizer.summarize(
                last_tactics,
                last_data,
            )

            plan_record["promotion"] = promotion
            plan_record["promotion_source"] = "resample_from_success"
            plan_record["llm_usage_after_plan"] = get_llm_metrics()

            save_progress(plan_file, plan_record)
            save_train_metrics("running", plan_idx)
            continue

        # ==== 3.5 Accept (success path)====
        if last_score is not None and cur_score == last_score:
            stuck_steps += 1
        else:
            stuck_steps = 0

        promotion = summarizer.summarize(tactics, data)

        last_tactics = tactics
        last_data = data
        last_score = cur_score

        plan_record["promotion"] = promotion
        plan_record["promotion_source"] = "accepted"
        plan_record["stuck_steps"] = stuck_steps
        plan_record["llm_usage_after_plan"] = get_llm_metrics()

        save_progress(plan_file, plan_record)
        save_train_metrics("running", plan_idx)

        print(f"✅ Accepted new tactics, last_score updated to {last_score}")

    save_train_metrics("completed", plan_times - 1)
    print(f"\n✅ All plans saved under: {exp_dir}")

def test_existing_code_multiple(args, policy_source_path, partner_list, final=False):
    """Evaluate one Co-π-tree policy with each partner in ``partner_list``."""
    import numpy as np

    policy_source_path = os.path.abspath(policy_source_path)
    source_dir = os.path.basename(os.path.dirname(policy_source_path)) or "policy"
    source_suffix = source_dir[4:] if source_dir.startswith("exp_") else source_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_suffix = "_final" if final else ""
    exp_dir = os.path.join(
        TEST_RESULTS_DIR,
        f"test_{source_suffix}_{timestamp}{final_suffix}",
    )
    os.makedirs(exp_dir, exist_ok=False)

    llmtree_index = 0 if args.p0 == "llmtree" else 1
    final_code = load_policy_code(policy_source_path)
    staged_policy = stage_llmtree_policy(args.layout, llmtree_index, policy_source_path)
    original_p0, original_p1 = args.p0, args.p1
    original_replay_dir = args.replay_dir
    summary = {}

    try:
        for idx, partner in enumerate(partner_list):
            print(f"\n=== Testing partner: {partner} ===")
            if llmtree_index == 0:
                args.p0, args.p1 = "llmtree", partner
                partner_player = 1
            else:
                args.p0, args.p1 = partner, "llmtree"
                partner_player = 0

            if args.save_replay:
                args.replay_dir = os.path.join(
                    exp_dir, f"replay_{idx:02d}_{partner}"
                )

            try:
                tag, data = run_test_in_subprocess(args)
            finally:
                args.replay_dir = original_replay_dir

            plan_file = os.path.join(
                exp_dir, f"test_plan_{idx:02d}_{partner}.json"
            )
            if tag != "result" or not isinstance(data, dict):
                error_record = {
                    "partner": partner,
                    "partner_player": partner_player,
                    "llmtree_player": llmtree_index,
                    "tag": tag,
                    "error": data,
                }
                with open(plan_file, "w", encoding="utf-8") as f:
                    json.dump(error_record, f, ensure_ascii=False, indent=2)
                raise RuntimeError(
                    f"Evaluation failed for partner {partner}; see {plan_file}"
                )

            result_dict = data.get("result", {})
            scores = result_dict.get("scores", [])
            avg = float(result_dict.get("avg", 0))
            std = float(result_dict.get("std", 0))
            test_elapsed_seconds = data.get("test_elapsed_seconds")
            result_record = {
                "partner": partner,
                "partner_player": partner_player,
                "llmtree_player": llmtree_index,
                "policy_source_path": policy_source_path,
                "final_code": final_code,
                "scores": scores,
                "avg": avg,
                "std": std,
                "test_elapsed_seconds": test_elapsed_seconds,
                "raw_result": data,
                "tag": tag,
            }
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(result_record, f, ensure_ascii=False, indent=2)

            summary[partner] = {
                "scores": scores,
                "avg": avg,
                "std": std,
                "test_elapsed_seconds": test_elapsed_seconds,
            }
            print(f"✅ Saved: {plan_file}")
            print(f"   avg={avg}, std={std}, scores={scores}")
    finally:
        args.p0, args.p1 = original_p0, original_p1
        args.replay_dir = original_replay_dir
        restore_llmtree_policy(staged_policy)

    all_avgs = [record["avg"] for record in summary.values()]
    summary["overall"] = {
        "avg_across_algorithms": float(np.mean(all_avgs)) if all_avgs else 0.0,
        "std_across_algorithms": float(np.std(all_avgs)) if all_avgs else 0.0,
        "var_across_algorithms": float(np.var(all_avgs)) if all_avgs else 0.0,
        "total_test_elapsed_seconds": float(sum(
            record.get("test_elapsed_seconds", 0) or 0
            for record in summary.values()
        )),
    }
    summary_file = os.path.join(exp_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n📊 Overall Summary:")
    print(summary["overall"])
    print(f"📁 Summary saved to: {summary_file}")
    return summary


if __name__ == '__main__':    
    parser = ArgumentParser(description='OvercookedAI Experiment')
    parser.add_argument('--layout', '-l', type=str, default='coordination_ring', choices=['cramped_room', 'asymmetric_advantages', 'coordination_ring', 'forced_coordination', 'counter_circuit'])
    main_algorithms = ('llmtree',) + AI_PARTNER_ALGS
    parser.add_argument('--p0', type=str, required=True, choices=main_algorithms,
                        help='algorithm for Player 0; llmtree means Co-pi-tree')
    parser.add_argument('--p1', type=str, required=True, choices=main_algorithms,
                        help='algorithm for Player 1; llmtree means Co-pi-tree')
    parser.add_argument('--horizon', type=int, default=400, help='Horizon steps in one game')
    parser.add_argument(
    '--planner_merge', 
    action='store_true',        # True if --merge_output is passed on the command line; default is False
    help='Whether to merge planner outputs into a single content (default: separate outputs)'
    )#Whether the planner should merge outputs
    parser.add_argument('--final', action='store_true',
                        help='append _final to the evaluation output directory name')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='train or test mode')

    parser.add_argument('--plan_path', type=str, default=None,
                        help='path to best plan json')
    parser.add_argument('--p0_policy_path', type=str, default=None,
                        help='test-only: policy source for p0 llmtree, supports plan json or raw .txt code')
    parser.add_argument('--p1_policy_path', type=str, default=None,
                        help='test-only: policy source for p1 llmtree, supports plan json or raw .txt code')

    parser.add_argument('--test_alg', type=str, choices=AI_PARTNER_ALGS, default=None,
                        help='source partner used while constructing the policy')
    parser.add_argument('--save_replay', action='store_true',
                        help='save rendered replay artifacts during test mode')
    parser.add_argument('--replay_dir', type=str, default=None,
                        help='directory for replay outputs; defaults to src1/rendered_replays/<timestamped-run>')
    parser.add_argument('--replay_episode', type=str, default='worst',
                        help='which evaluation episode to export: worst, best, first, last, all, or zero-based index')
    parser.add_argument('--replay_fps', type=int, default=6,
                        help='GIF playback speed for exported replays')
    parser.add_argument('--replay_tile_size', type=int, default=72,
                        help='tile size in pixels for exported replay frames')
    parser.add_argument('--replay_frames', action='store_true',
                        help='keep per-frame PNG files in addition to GIF output')
    parser.add_argument('--replay_gif', action='store_true',
                        help='export replay as GIF')
    parser.add_argument('--test_times', type=int, default=3,
                        help='number of evaluation episodes per partner in test mode')
    args = parser.parse_args()

    llmtree_count = (args.p0 == "llmtree") + (args.p1 == "llmtree")
    if args.mode == "train":
        if llmtree_count != 1:
            parser.error("train mode requires exactly one llmtree player")
    elif args.p0 == "llmtree" and args.p1 == "llmtree":
        if not args.p0_policy_path or not args.p1_policy_path:
            parser.error(
                "llmtree duel requires --p0_policy_path and --p1_policy_path"
            )
        for policy_path in (args.p0_policy_path, args.p1_policy_path):
            if not os.path.isfile(policy_path):
                parser.error(f"policy file does not exist: {policy_path}")
    else:
        if llmtree_count != 1:
            parser.error("test mode requires one llmtree player")
        if not args.plan_path or not args.test_alg:
            parser.error("holdout test requires --plan_path and --test_alg")
        if not os.path.isfile(args.plan_path):
            parser.error(f"policy source does not exist: {args.plan_path}")
        source_partner = args.p1 if args.p0 == "llmtree" else args.p0
        if source_partner != args.test_alg:
            parser.error("the non-llmtree player must match --test_alg")

    if args.save_replay and not args.replay_gif and not args.replay_frames:
        args.replay_gif = True

    if args.mode == "train":
        run(args)

    elif args.mode == "test":
        if args.p0 == "llmtree" and args.p1 == "llmtree":
            test_llmtree_duel(args)
        else:
            # Build the zero-shot test set
            test_list = [a for a in AI_PARTNER_ALGS if a != args.test_alg]

            print(f"[TEST] zero-shot partners: {test_list}")

            if args.p0 == "llmtree":
                # llmtree is p0 -> test p1_list
                test_existing_code_multiple(args, args.plan_path, test_list,args.final)

            elif args.p1 == "llmtree":
                # llmtree is p1 -> test p0_list
                test_existing_code_multiple(args, args.plan_path, test_list,args.final)

            else:
                raise ValueError("Test mode only supports llmtree involved")
