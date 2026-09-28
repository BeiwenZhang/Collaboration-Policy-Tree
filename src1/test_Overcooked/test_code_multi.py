import datetime
import os
import time

import numpy as np
from project_paths import REPLAY_DIR, STATE_LOG_PATH, ensure_local_libs

ensure_local_libs()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*")

VERSION = "0.0.1"
print(f"\n----This overcook version is {VERSION}----\n")

from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from test_Overcooked.replay_utils import save_episode_replay
from test_Overcooked.utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent


STATE_LOG_PATH = str(STATE_LOG_PATH)


def _variant_get(variant, key, default=None):
    if isinstance(variant, dict):
        return variant.get(key, default)
    return getattr(variant, key, default)


def _resolve_replay_targets(replay_episode, scores):
    replay_episode = str(replay_episode).strip().lower()
    if replay_episode == "all":
        return list(range(len(scores)))
    if replay_episode == "best":
        return [int(np.argmax(scores))]
    if replay_episode == "first":
        return [0]
    if replay_episode == "last":
        return [len(scores) - 1]
    if replay_episode == "worst":
        return [int(np.argmin(scores))]

    try:
        idx = int(replay_episode)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported replay_episode={replay_episode!r}. "
            "Use worst/best/first/last/all or a zero-based episode index."
        ) from exc

    if idx < 0 or idx >= len(scores):
        raise IndexError(f"replay_episode index {idx} is out of range for {len(scores)} test episodes")
    return [idx]


def _build_agent(alg, mdp, layout, episode_idx, agent_list_len, variant):
    if alg == "llmtree":
        idx = agent_list_len
        agent = make_agent(alg, mdp, layout, agent_index=idx)
        return agent, agent

    if alg == "ProAgent":
        agent = make_agent(
            alg,
            mdp,
            layout,
            model=_variant_get(variant, "gpt_model"),
            prompt_level=_variant_get(variant, "prompt_level"),
            belief_revision=_variant_get(variant, "belief_revision"),
            retrival_method=_variant_get(variant, "retrival_method"),
            K=_variant_get(variant, "K"),
        )
        return agent, None

    if alg == "BC":
        return make_agent(alg, mdp, layout, seed_id=episode_idx), None

    return make_agent(alg, mdp, layout), None


def test(variant):
    test_start_perf = time.perf_counter()
    test_started_at = datetime.datetime.now().isoformat(timespec="seconds")

    layout = _variant_get(variant, "layout")
    horizon = int(_variant_get(variant, "horizon", 400))
    test_times = int(_variant_get(variant, "test_times", 3))

    if VERSION == "1.1.0":
        mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout])
    elif VERSION == "0.0.1":
        mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout])
    else:
        raise ValueError(f"Unsupported overcooked version: {VERSION}")

    env = OvercookedEnv(mdp, horizon=horizon)
    p0_algo = _variant_get(variant, "p0")
    p1_algo = _variant_get(variant, "p1")
    target_llmtree_index = 0 if p0_algo == "llmtree" else 1

    save_replay = bool(_variant_get(variant, "save_replay", False))
    replay_dir = _variant_get(variant, "replay_dir")
    replay_episode = _variant_get(variant, "replay_episode", "worst")
    replay_fps = int(_variant_get(variant, "replay_fps", 6))
    replay_tile_size = int(_variant_get(variant, "replay_tile_size", 72))
    replay_frames = bool(_variant_get(variant, "replay_frames", True))
    replay_gif = bool(_variant_get(variant, "replay_gif", True))

    results = []
    all_feedbacks = []
    episode_rollouts = []

    for episode_idx in range(test_times):
        agents_list = []
        llmtree_agents = []

        for alg in [p0_algo, p1_algo]:
            agent, maybe_llmtree = _build_agent(alg, mdp, layout, episode_idx, len(agents_list), variant)
            if maybe_llmtree is not None:
                llmtree_agents.append(maybe_llmtree)
            agents_list.append(agent)

        team = AgentGroup(*agents_list)
        team.reset()
        env.reset()
        r_total = 0

        layout_prompt = ""
        if llmtree_agents:
            for llmtree_agent in llmtree_agents:
                current_layout_prompt = llmtree_agent.generate_layout_prompt()
                if not layout_prompt:
                    layout_prompt = current_layout_prompt

        feedback = {
            "layout": layout_prompt,
            "step": {},
            "result": "",
        }

        episode_states = [env.state.deepcopy()]
        episode_rewards = []

        for t in range(horizon):
            with open(STATE_LOG_PATH, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n")
                state_str = env.mdp.state_string(env.state).replace("ø", "o")
                log_file.write(state_str + "\n")

            joint_action = team.joint_action(env.state)

            if llmtree_agents:
                selected_feedback = None
                for llmtree_agent in llmtree_agents:
                    step_info = llmtree_agent.generate_step_feedback()
                    if step_info is None:
                        continue
                    if llmtree_agent.agent_index == target_llmtree_index:
                        selected_feedback = step_info
                        break
                    if selected_feedback is None:
                        selected_feedback = step_info
                if selected_feedback:
                    feedback["step"][t] = selected_feedback

            obs, reward, done, env_info = env.step(joint_action)
            del env_info

            r_total += reward
            episode_rewards.append(reward)
            episode_states.append(obs.deepcopy())

            if done:
                break

        feedback["result"] = r_total
        results.append(r_total)
        all_feedbacks.append(feedback)
        episode_rollouts.append(
            {
                "states": episode_states,
                "rewards": episode_rewards,
                "score": r_total,
            }
        )

        print(f"Test {episode_idx + 1}/{test_times}: {r_total}")

    worst_idx = int(np.argmin(results))
    worst_feedback = all_feedbacks[worst_idx]

    avg_r_total = float(np.mean(results))
    std_r_total = float(np.std(results))

    worst_feedback["result"] = {
        "scores": results,
        "avg": avg_r_total,
        "std": std_r_total,
        "worst_score": results[worst_idx],
    }

    if save_replay:
        if replay_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            replay_dir = os.path.join(
                str(REPLAY_DIR),
                f"{layout}_{p0_algo}_{p1_algo}_{timestamp}",
            )

        target_indices = _resolve_replay_targets(replay_episode, results)
        replay_outputs = []
        for target_idx in target_indices:
            rollout = episode_rollouts[target_idx]
            episode_output_dir = os.path.join(replay_dir, f"episode_{target_idx:02d}")
            replay_outputs.append(
                save_episode_replay(
                    mdp=mdp,
                    states=rollout["states"],
                    rewards=rollout["rewards"],
                    output_dir=episode_output_dir,
                    layout=layout,
                    p0_algo=p0_algo,
                    p1_algo=p1_algo,
                    selected_episode=target_idx,
                    fps=replay_fps,
                    tile_size=replay_tile_size,
                    save_frames=replay_frames,
                    save_gif=replay_gif,
                )
            )
        worst_feedback["replay_outputs"] = replay_outputs

    worst_feedback["test_elapsed_seconds"] = time.perf_counter() - test_start_perf
    worst_feedback["test_started_at"] = test_started_at
    worst_feedback["test_finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    return worst_feedback
