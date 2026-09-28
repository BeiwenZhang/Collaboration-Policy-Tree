import datetime
import os
import sys
import json
import numpy as np
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*") # ignore "Unable to register cuBLAS factory" due to use tf-CPU

from project_paths import ensure_local_libs

ensure_local_libs()

from distutils.util import strtobool
def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

# import pkg_resources
# VERSION = pkg_resources.get_distribution("overcooked_ai").version
#import importlib_metadata
#VERSION = importlib_metadata.version("overcooked_ai")
VERSION = '0.0.1'
print(f'\n----This overcook version is {VERSION}----\n')

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action


from test_Overcooked.utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent


def main(variant):
    test_start_perf = time.perf_counter()
    test_started_at = datetime.datetime.now().isoformat(timespec="seconds")

    layout = variant.layout
    horizon = variant.horizon
    episode = variant.episode
    test_times = 10

    if VERSION == '1.1.0':
        mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout])
    elif VERSION == '0.0.1':
        # This overcook version is 0.0.1
        # counter_circuit:       random3
        # forced_coordination:   random0
        # cramped_room:          simple
        # coordination_ring:     random1
        # asymmetric_advantages: unident_s
        mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout])

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    p0_algo = variant.p0
    p1_algo = variant.p1
    #print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")

    results = []
    for i in range(test_times):

        agents_list = []
        for agent_index, alg in enumerate([p0_algo, p1_algo]):
            if alg == 'llmtree':
                agent = make_agent(alg, mdp, layout, agent_index=agent_index)
            elif alg == "ProAgent":
                gpt_model = variant['gpt_model']
                retrival_method = variant['retrival_method']
                K = variant['K']
                prompt_level = variant['prompt_level']
                belief_revision = variant['belief_revision']
                agent = make_agent(alg, mdp, layout, model=gpt_model, 
                                   prompt_level=prompt_level, 
                                   belief_revision=belief_revision, 
                                   retrival_method=retrival_method, K=K)
            elif alg == "BC":
                agent = make_agent(alg, mdp, layout, seed_id=i)
            else:
                agent = make_agent(alg, mdp, layout)
            agents_list.append(agent)

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0

        for t in range(horizon):
            a_t = team.joint_action(env.state)
            obs, reward, done, env_info = env.step(a_t)

            ml_actions = obs.ml_actions
            skills = f""
            for i, ml_action in enumerate(ml_actions):
                if ml_action == None:
                    continue
                skills += f"P{i} finished <{ml_action}>. "
            #print(skills)

            r_total += reward
            #rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')

        ## finish one episode
        if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
            print(f"\n================\n")
            try: # ProAgent id = 0
                print(f"P1's real behavior: {team.agents[0].teammate_ml_actions_dict}")
                print(f"The infered P1's intention: {team.agents[0].teammate_intentions_dict}")
            except: # ProAgent id = 1
                print(f"P0's real behavior: {team.agents[1].teammate_ml_actions_dict}")
                print(f"The infered P0's intention: {team.agents[1].teammate_intentions_dict}")
            print(f"\n================\n")

        print(f"Episode {i+1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)


    result_dict = {
        "input": variant,
        "raw_results": results,
        "mean_result": int(np.mean(results)),
        "test_elapsed_seconds": time.perf_counter() - test_start_perf,
        "test_started_at": test_started_at,
        "test_finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    for (k,v) in result_dict.items():
        print(f'{k}: {v}')

    if variant['save']:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        if variant['log_dir'] == None and variant['debug']:
            log_dir = f"experiments/{timestamp}_{layout}_{horizon}_{p0_algo}_{p1_algo}_{episode}numep"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            log_dir = variant['log_dir']

        print(f"This is {log_dir}")
        if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
            json_file = f"{log_dir}/results_{episode}_{horizon}_{gpt_model}_{prompt_level}_{retrival_method}_{K}.json"
        else:
            json_file = f"{log_dir}/results_{episode}_{horizon}.json"
        with open(json_file, "w") as f:
            json.dump(result_dict, f, indent=4)
