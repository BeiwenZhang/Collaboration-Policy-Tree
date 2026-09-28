import json
import os
import pickle
import random
import itertools
import sys
import types
from collections import defaultdict
from abc import ABC, abstractmethod
from queue import Empty, Full, LifoQueue, Queue
from threading import Lock, Thread
import time
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.python.saved_model import tag_constants
except ImportError:
    tf = None
    tag_constants = None

try:
    import ray
except ImportError:
    ray = None
from utils import DOCKER_VOLUME, create_dirs
# from utils_ import get_agent_from_saved_model
# from utils_ import load_dict_from_file
try:
    from human_aware_rl.rllib.rllib import load_agent
except ImportError:
    load_agent = None
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.agents.agent import AgentFromPolicy, AgentPair
from overcooked_ai_py.planning.planners import (
    MediumLevelPlanner,
    NO_COUNTERS_PARAMS,
    MotionPlanner,
)
from overcooked_ai_py.utils import load_dict_from_file, load_pickle
from test_Overcooked.llmtree.decision_tree_agent import DecisionTreeAgent

# Relative path to where all static pre-trained agents are stored on server
AGENT_DIR = None

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = 400

LEGACY_MARL_AGENTS = {"SP", "MEP", "FCP", "PBT", "COLE"}
LEGACY_LAYOUT_ALIASES = {
    "counter_circuit_o_1order": "random3",
    "counter_circuit": "random3",
    "forced_coordination": "random0",
    "cramped_room": "simple",
    "coordination_ring": "random1",
    "asymmetric_advantages": "unident_s",
}

DEFAULT_BC_SEED = 0
DEFAULT_BC_SPLIT = "test"
# By default we keep legacy stable-baselines BC loader disabled because it
# mutates global TF runtime (TF1 compat shims) and can destabilize other agents.
ENABLE_GAIL_BC = str(os.getenv("ENABLE_GAIL_BC", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if ENABLE_GAIL_BC:
    print(
        "WARNING: ENABLE_GAIL_BC=1. Legacy BC runtime mutates global TensorFlow behavior "
        "and can destabilize SP/MEP/FCP/PBT/COLE in the same process."
    )


def _configure(max_game_time, agent_dir):
    global AGENT_DIR, MAX_GAME_TIME
    MAX_GAME_TIME = max_game_time
    AGENT_DIR = agent_dir


def fix_bc_path(path):
    """
    Loading a PPO agent trained with a BC agent requires loading the BC model as well when restoring the trainer, even though the BC model is not used in game
    For now the solution is to include the saved BC model and fix the relative path to the model in the config.pkl file
    """

    import dill

    # the path is the agents/Rllib.*/agent directory
    agent_path = os.path.dirname(path)
    with open(os.path.join(agent_path, "config.pkl"), "rb") as f:
        data = dill.load(f)
    bc_model_dir = data["bc_params"]["bc_config"]["model_dir"]
    last_dir = os.path.basename(bc_model_dir)
    bc_model_dir = os.path.join(agent_path, "bc_params", last_dir)
    data["bc_params"]["bc_config"]["model_dir"] = bc_model_dir
    with open(os.path.join(agent_path, "config.pkl"), "wb") as f:
        dill.dump(data, f)


def is_legacy_marl_agent(agent_name):
    return any(agent_name.startswith(prefix) for prefix in LEGACY_MARL_AGENTS)


def is_custom_bc_agent(agent_name):
    return agent_name == "BC" or agent_name.startswith("BC_")


def get_legacy_layout_name(layout_name):
    return LEGACY_LAYOUT_ALIASES.get(layout_name, layout_name)


def get_default_bc_run_name(layout_name, seed=DEFAULT_BC_SEED, split=DEFAULT_BC_SPLIT):
    legacy_layout = get_legacy_layout_name(layout_name)
    return "{}_bc_{}_seed{}".format(legacy_layout, split, seed)


def resolve_bc_run_name(agent_name, layout_name):
    if agent_name == "BC":
        return get_default_bc_run_name(layout_name)
    return agent_name[len("BC_") :]


def get_custom_bc_model_root():
    candidates = [
        os.path.join(AGENT_DIR, "models", "bc_runs") if AGENT_DIR is not None else None,
        "/app/static/assets/agents/models/bc_runs",
        "/app/proagent_models/bc_runs",
        "/app/models/bc_runs",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if os.path.isdir(candidate):
            return candidate
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return "/app/static/assets/agents/models/bc_runs"


def _ensure_stable_baselines_importable():
    candidates = [
        "/app/proagent_lib/stable-baselines",
        "/app/proagent_lib_ori/stable-baselines",
        "/app/test_Overcooked/lib/stable-baselines",
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _patch_tensorflow_for_legacy_stable_baselines():
    """
    stable_baselines (TF1-era) expects top-level tf symbols such as
    tf.Session/tf.placeholder/tf.variable_scope and tf.contrib.layers.fully_connected.
    The demo runs on TF2, so we expose a minimal compat shim before importing GAIL.
    """
    # Ensure graph-mode behavior is available.
    try:
        tf.compat.v1.disable_v2_behavior()
    except Exception:
        pass

    # Broad TF1->TF2 compatibility: copy missing top-level symbols from compat.v1.
    # This avoids chasing one missing alias at a time (ConfigProto, variables_initializer, etc.).
    for name in dir(tf.compat.v1):
        if name.startswith("_"):
            continue
        if hasattr(tf, name):
            continue
        try:
            setattr(tf, name, getattr(tf.compat.v1, name))
        except Exception:
            pass

    compat_map = [
        "Session",
        "Graph",
        "placeholder",
        "global_variables_initializer",
        "variable_scope",
        "get_variable_scope",
        "get_variable",
        "assign",
        "assign_add",
        "assign_sub",
        "gradients",
        "summary",
        "trainable_variables",
        "global_variables",
        "random_uniform",
        "ConfigProto",
        "GPUOptions",
        "get_default_session",
        "get_collection",
        "get_collection_ref",
        "log",
    ]
    for name in compat_map:
        if not hasattr(tf, name) and hasattr(tf.compat.v1, name):
            setattr(tf, name, getattr(tf.compat.v1, name))

    if not hasattr(tf, "GraphKeys") and hasattr(tf.compat.v1, "GraphKeys"):
        tf.GraphKeys = tf.compat.v1.GraphKeys

    # Force TF1-style module namespaces expected by stable-baselines.
    if hasattr(tf.compat.v1, "summary"):
        tf.summary = tf.compat.v1.summary
    if hasattr(tf.compat.v1, "train"):
        tf.train = tf.compat.v1.train
    if hasattr(tf.compat.v1, "losses"):
        tf.losses = tf.compat.v1.losses
    if hasattr(tf.compat.v1, "metrics"):
        tf.metrics = tf.compat.v1.metrics

    if (
        hasattr(tf, "nn")
        and not hasattr(tf.nn, "softmax_cross_entropy_with_logits_v2")
        and hasattr(tf.nn, "softmax_cross_entropy_with_logits")
    ):
        tf.nn.softmax_cross_entropy_with_logits_v2 = tf.nn.softmax_cross_entropy_with_logits

    # Build minimal tf.contrib shim used by the bundled stable-baselines fork.
    contrib_module = sys.modules.get("tensorflow.contrib")
    if contrib_module is None:
        contrib_module = types.ModuleType("tensorflow.contrib")
        sys.modules["tensorflow.contrib"] = contrib_module

    layers_module = sys.modules.get("tensorflow.contrib.layers")
    if layers_module is None:
        layers_module = types.ModuleType("tensorflow.contrib.layers")
        sys.modules["tensorflow.contrib.layers"] = layers_module

    if not hasattr(layers_module, "fully_connected"):
        def _fully_connected(inputs, num_outputs, activation_fn=tf.identity):
            in_dim = inputs.get_shape().as_list()[-1]
            if in_dim is None:
                raise ValueError("fully_connected requires statically known input dim")
            with tf.variable_scope(None, default_name="fully_connected"):
                weight = tf.get_variable(
                    "weights",
                    [in_dim, num_outputs],
                    initializer=tf.compat.v1.glorot_uniform_initializer(),
                )
                bias = tf.get_variable(
                    "biases",
                    [num_outputs],
                    initializer=tf.zeros_initializer(),
                )
                output = tf.matmul(inputs, weight) + bias
                if activation_fn is not None:
                    output = activation_fn(output)
                return output
        layers_module.fully_connected = _fully_connected

    if not hasattr(layers_module, "layer_norm"):
        def _layer_norm(inputs, center=True, scale=True):
            return tf.keras.layers.LayerNormalization(center=center, scale=scale)(inputs)
        layers_module.layer_norm = _layer_norm

    contrib_module.layers = layers_module
    tf.contrib = contrib_module


def _infer_bc_obs_dim(model):
    try:
        shape = model.observation_space.shape
        if shape is None or len(shape) == 0:
            return None
        return int(shape[0])
    except Exception:
        return None


def _to_bc_obs(obs):
    """
    Convert modern Overcooked featurization to legacy BC input width when needed.
    BC models here were trained on 62-dim features, while newer env featurization can be 96-dim.
    """
    obs = np.asarray(obs)
    if obs.ndim != 2:
        return obs
    if obs.shape[1] == 62:
        return obs
    # Keep the first 62 channels for legacy BC compatibility.
    if obs.shape[1] > 62:
        return obs[:, :62]
    return obs


def load_custom_bc_agent(run_name):
    if not ENABLE_GAIL_BC:
        raise IOError(
            "Legacy GAIL BC loader is disabled (ENABLE_GAIL_BC=0). "
            "Use a prebuilt BC agent.pickle under static/assets/agents/BC, "
            "or explicitly set ENABLE_GAIL_BC=1 if you accept legacy TF/MPI deps."
        )

    bc_root = get_custom_bc_model_root()
    run_dir = os.path.join(bc_root, run_name)
    if not os.path.isdir(run_dir):
        raise IOError(
            "BC run not found: {} (expected under {})".format(run_name, bc_root)
        )

    _ensure_stable_baselines_importable()
    _patch_tensorflow_for_legacy_stable_baselines()

    try:
        from stable_baselines import GAIL
    except Exception as e:
        raise IOError(
            "stable_baselines.GAIL not available: {}. "
            "Please mount ProAgent stable-baselines to /app/proagent_lib "
            "(or /app/proagent_lib_ori).".format(e)
        )

    metadata_path = os.path.join(run_dir, "bc_metadata.pickle")
    model_path = os.path.join(run_dir, "model.pkl")
    if not os.path.exists(metadata_path):
        raise IOError("bc_metadata missing: {}".format(metadata_path))
    if not os.path.exists(model_path):
        raise IOError("model.pkl missing: {}".format(model_path))

    bc_metadata = load_pickle(metadata_path)
    bc_params = bc_metadata["bc_params"]
    # Legacy BC metadata may use historical layout ids not present in the
    # current mounted overcooked layouts.
    bc_mdp_params = dict(bc_params.get("mdp_params", {}))
    bc_layout_name = bc_mdp_params.get("layout_name")
    bc_layout_aliases = {
        "simple": "simple_o",
        "unident_s": "unident",
    }
    if bc_layout_name in bc_layout_aliases:
        bc_mdp_params["layout_name"] = bc_layout_aliases[bc_layout_name]
        print(
            "BC layout alias applied: {} -> {}".format(
                bc_layout_name, bc_mdp_params["layout_name"]
            )
        )
    model = GAIL.load(model_path)
    mdp = OvercookedGridworld.from_layout_name(**bc_mdp_params)
    planner = MediumLevelPlanner.from_pickle_or_compute(
        mdp, NO_COUNTERS_PARAMS, force_compute=False
    )
    mlam = planner.ml_action_manager

    def encoded_state_policy(observations, include_waits=True, stochastic=False):
        expected_dim = _infer_bc_obs_dim(model)
        obs_arr = np.asarray(observations)
        if expected_dim == 62 and obs_arr.ndim == 2 and obs_arr.shape[1] != 62:
            obs_arr = _to_bc_obs(obs_arr)
        action_probs_n = model.action_probability(obs_arr)
        if stochastic:
            return [
                np.random.choice(len(action_probs[i]), p=action_probs[i])
                for i in range(len(action_probs))
            ]
        return action_probs_n

    def state_policy(mdp_states, agent_indices, include_waits, stochastic=False):
        encode_fn = lambda s: mdp.featurize_state(s, mlam)
        obs = []
        for agent_idx, s in zip(agent_indices, mdp_states):
            ob = encode_fn(s)[agent_idx]
            obs.append(ob)
        obs = np.array(obs)
        return encoded_state_policy(obs, include_waits, stochastic)

    class _BCImitationAgent(AgentFromPolicy):
        def __init__(
            self,
            state_policy,
            direct_policy,
            mlp=None,
            stochastic=True,
            no_waits=False,
            stuck_time=3,
        ):
            super().__init__(state_policy, direct_policy)
            self.stuck_time = stuck_time
            self.history_length = stuck_time + 1
            self.stochastic = stochastic
            self.action_probs = False
            self.no_waits = no_waits
            self.will_unblock_if_stuck = False if stuck_time == 0 else True
            self.mlp = mlp
            self.reset()

        def action(self, state):
            return self.actions(state)

        def actions(self, states, agent_indices=None):
            if agent_indices is None:
                assert isinstance(states, OvercookedState)
                agent_indices = [self.agent_index]
                states = [states]

            assert len(states) > 0
            all_actions = self.multi_action(states, agent_indices)
            if len(agent_indices) > 1:
                return all_actions
            return all_actions[0]

        def multi_action(self, states, agent_indices):
            try:
                action_probs_n = list(
                    self.state_policy(states, agent_indices, not self.no_waits)
                )
            except AttributeError:
                raise AttributeError(
                    "Need to set the agent_index or mdp of the Agent before using it"
                )

            all_actions = []
            for parallel_agent_idx, curr_agent_action_probs in enumerate(action_probs_n):
                curr_agent_idx = agent_indices[parallel_agent_idx]
                curr_agent_state = states[parallel_agent_idx]
                self.set_agent_index(curr_agent_idx)

                if self.no_waits:
                    curr_agent_action_probs = self.remove_indices_and_renormalize(
                        curr_agent_action_probs,
                        [Action.ACTION_TO_INDEX[Direction.STAY]],
                    )

                if self.will_unblock_if_stuck:
                    curr_agent_action_probs = self.unblock_if_stuck(
                        curr_agent_state, curr_agent_action_probs
                    )

                if self.stochastic:
                    action_idx = np.random.choice(
                        len(curr_agent_action_probs), p=curr_agent_action_probs
                    )
                else:
                    action_idx = np.argmax(curr_agent_action_probs)
                curr_agent_action = Action.INDEX_TO_ACTION[action_idx]
                self.add_to_history(curr_agent_state, curr_agent_action)

                if self.action_probs:
                    all_actions.append(curr_agent_action_probs)
                else:
                    all_actions.append(curr_agent_action)
            return all_actions

        def unblock_if_stuck(self, state, action_probs):
            stuck, last_actions = self.is_stuck(state)
            if stuck:
                assert any(
                    [a not in last_actions for a in Direction.ALL_DIRECTIONS]
                ), last_actions
                last_action_idxes = [Action.ACTION_TO_INDEX[a] for a in last_actions]
                action_probs = self.remove_indices_and_renormalize(
                    action_probs, last_action_idxes
                )
            return action_probs

        def is_stuck(self, state):
            if None in self.history[self.agent_index]:
                return False, []

            last_states = [
                s_a[0]
                for s_a in self.history[self.agent_index][-self.stuck_time :]
            ]
            last_actions = [
                s_a[1]
                for s_a in self.history[self.agent_index][-self.stuck_time :]
            ]
            player_states = [s.players[self.agent_index] for s in last_states]
            pos_and_ors = [p.pos_and_or for p in player_states] + [
                state.players[self.agent_index].pos_and_or
            ]
            if self.check_equal(pos_and_ors):
                return True, last_actions
            return False, []

        @staticmethod
        def remove_indices_and_renormalize(probs, indices):
            if len(np.array(probs).shape) > 1:
                probs = np.array(probs)
                for row_idx, row in enumerate(indices):
                    for idx in indices:
                        probs[row_idx][idx] = 0
                norm_probs = probs.T / np.sum(probs, axis=1)
                return norm_probs.T
            else:
                for idx in indices:
                    probs[idx] = 0
                return probs / sum(probs)

        @staticmethod
        def check_equal(iterator):
            first_pos_and_or = iterator[0]
            for curr_pos_and_or in iterator:
                if (
                    curr_pos_and_or[0] != first_pos_and_or[0]
                    or curr_pos_and_or[1] != first_pos_and_or[1]
                ):
                    return False
            return True

        def add_to_history(self, state, action):
            assert len(self.history[self.agent_index]) == self.history_length
            self.history[self.agent_index].append((state, action))
            self.history[self.agent_index] = self.history[self.agent_index][1:]

        def reset(self):
            self.history = defaultdict(lambda: [None] * self.history_length)

    agent = _BCImitationAgent(state_policy, encoded_state_policy, mlp=mlam)

    return agent


def normalize_policy_action(result):
    """
    Normalize heterogeneous agent return values to a single Overcooked action.

    Supported patterns:
    - action
    - (action, info)
    - (goal, action)
    - (None, action)
    """
    if result is None:
        return Action.STAY

    if result in Action.ALL_ACTIONS:
        return result

    if isinstance(result, tuple):
        if len(result) == 0:
            return Action.STAY
        if len(result) == 1:
            return normalize_policy_action(result[0])
        if result[1] in Action.ALL_ACTIONS:
            return result[1]
        if result[0] in Action.ALL_ACTIONS:
            return result[0]

    if isinstance(result, list) and len(result) == 1:
        return normalize_policy_action(result[0])

    raise ValueError("Unsupported policy action format: {!r}".format(result))


def is_decision_tree_policy(policy):
    return (
        type(policy).__name__ == "DecisionTreeAgent"
        or "decision_tree_agent" in type(policy).__module__
    )


def get_decision_tree_policy_dir():
    candidates = [
        "/workspace/src1/policy",
        "/app/policy",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "policy")),
        "/policy",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _decision_tree_mlam_matches(mlp, expected_params):
    """Validate a cached planner before reusing it for decision-tree agents."""
    params = getattr(mlp, "params", None) or {}
    scalar_keys = ("start_orientations", "wait_allowed", "same_motion_goals")
    for key in scalar_keys:
        if params.get(key) != expected_params[key]:
            return False, "{} mismatch: cached={} expected={}".format(
                key, params.get(key), expected_params[key]
            )

    list_keys = ("counter_goals", "counter_drop", "counter_pickup")
    for key in list_keys:
        cached = sorted(params.get(key, []))
        expected = sorted(expected_params[key])
        if cached != expected:
            return False, "{} mismatch: cached={} expected={}".format(
                key, cached, expected
            )

    return True, "ok"


def build_decision_tree_mlam(mdp):
    mlam_params = {
        "start_orientations": False,
        "wait_allowed": True,
        "counter_goals": [],
        "counter_drop": [],
        "counter_pickup": [],
        "same_motion_goals": True,
    }
    counter_locations = mdp.get_counter_locations()
    mlam_params["counter_goals"] = counter_locations
    mlam_params["counter_drop"] = counter_locations
    mlam_params["counter_pickup"] = counter_locations
    custom_filename = "{}_computed_dt_am.pkl".format(mdp.layout_name)

    try:
        mlp = MediumLevelPlanner.from_pickle_or_compute(
            mdp,
            mlam_params,
            custom_filename=custom_filename,
            force_compute=False,
        )
        matches, reason = _decision_tree_mlam_matches(mlp, mlam_params)
        if matches:
            print(
                "Loaded decision-tree MLAM from dedicated planner file {}".format(
                    custom_filename
                )
            )
            return mlp.ml_action_manager

        print(
            "Decision-tree MLAM cache {} is stale ({}). Recomputing.".format(
                custom_filename, reason
            )
        )
    except Exception as e:
        print(
            "Failed to load decision-tree MLAM cache {}: {}. Recomputing.".format(
                custom_filename, e
            )
        )

    return MediumLevelPlanner.from_pickle_or_compute(
        mdp,
        mlam_params,
        custom_filename=custom_filename,
        force_compute=True,
    ).ml_action_manager


def legacy_marl_lossless_state_encoding(mdp, overcooked_state):
    """
    Reconstruct the legacy 20-channel observation expected by the ProAgent
    MARL models directly from the current state, instead of approximating it
    from the modern 26-channel encoding.
    """
    base_map_features = [
        "pot_loc",
        "counter_loc",
        "onion_disp_loc",
        "dish_disp_loc",
        "serve_loc",
    ]
    variable_map_features = [
        "onions_in_pot",
        "onions_cook_time",
        "onion_soup_loc",
        "dishes",
        "onions",
    ]
    all_objects = overcooked_state.all_objects_list
    pot_locations = set(mdp.get_pot_locations())

    def make_layer(position, value):
        layer = np.zeros(mdp.shape)
        layer[position] = value
        return layer

    def process_for_player(primary_agent_idx):
        other_agent_idx = 1 - primary_agent_idx
        ordered_player_features = [
            "player_{}_loc".format(primary_agent_idx),
            "player_{}_loc".format(other_agent_idx),
        ] + [
            "player_{}_orientation_{}".format(
                i, Direction.DIRECTION_TO_INDEX[d]
            )
            for i, d in itertools.product(
                [primary_agent_idx, other_agent_idx],
                Direction.ALL_DIRECTIONS,
            )
        ]

        layers = ordered_player_features + base_map_features + variable_map_features
        state_mask_dict = {k: np.zeros(mdp.shape) for k in layers}

        for loc in mdp.get_counter_locations():
            state_mask_dict["counter_loc"][loc] = 1
        for loc in mdp.get_pot_locations():
            state_mask_dict["pot_loc"][loc] = 1
        for loc in mdp.get_onion_dispenser_locations():
            state_mask_dict["onion_disp_loc"][loc] = 1
        for loc in mdp.get_dish_dispenser_locations():
            state_mask_dict["dish_disp_loc"][loc] = 1
        for loc in mdp.get_serving_locations():
            state_mask_dict["serve_loc"][loc] = 1

        for i, player in enumerate(overcooked_state.players):
            player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            state_mask_dict["player_{}_loc".format(i)] = make_layer(
                player.position, 1
            )
            state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)] = make_layer(
                player.position, 1
            )

        for obj in all_objects:
            if obj.name == "soup":
                num_onions = sum(ing == "onion" for ing in obj.ingredients)
                if obj.position in pot_locations:
                    cook_time = 0 if obj.is_idle else obj._cooking_tick
                    if obj.is_ready:
                        cook_time = obj.cook_time
                    state_mask_dict["onions_in_pot"] += make_layer(
                        obj.position, num_onions
                    )
                    state_mask_dict["onions_cook_time"] += make_layer(
                        obj.position, cook_time
                    )
                else:
                    state_mask_dict["onion_soup_loc"] += make_layer(
                        obj.position, 1
                    )
            elif obj.name == "dish":
                state_mask_dict["dishes"] += make_layer(obj.position, 1)
            elif obj.name == "onion":
                state_mask_dict["onions"] += make_layer(obj.position, 1)

        state_mask_stack = np.array(
            [state_mask_dict[layer_id] for layer_id in layers]
        )
        state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
        return np.array(state_mask_stack).astype(int)

    num_players = len(overcooked_state.players)
    return tuple(process_for_player(i) for i in range(num_players))


class Game(ABC):

    """
    Class representing a game object. Coordinates the simultaneous actions of arbitrary
    number of players. Override this base class in order to use.

    Players can post actions to a `pending_actions` queue, and driver code can call `tick` to apply these actions.


    It should be noted that most operations in this class are not on their own thread safe. Thus, client code should
    acquire `self.lock` before making any modifications to the instance.

    One important exception to the above rule is `enqueue_actions` which is thread safe out of the box
    """

    # Possible TODO: create a static list of IDs used by the class so far to verify id uniqueness
    # This would need to be serialized, however, which might cause too great a performance hit to
    # be worth it

    EMPTY = "EMPTY"

    class Status:
        DONE = "done"
        ACTIVE = "active"
        RESET = "reset"
        INACTIVE = "inactive"
        ERROR = "error"

    def __init__(self, *args, **kwargs):
        """
        players (list): List of IDs of players currently in the game
        spectators (set): Collection of IDs of players that are not allowed to enqueue actions but are currently watching the game
        id (int):   Unique identifier for this game
        pending_actions List[(Queue)]: Buffer of (player_id, action) pairs have submitted that haven't been commited yet
        lock (Lock):    Used to serialize updates to the game state
        is_active(bool): Whether the game is currently being played or not
        """
        self.players = []
        self.spectators = set()
        self.pending_actions = []
        self.id = kwargs.get("id", id(self))
        self.lock = Lock()
        self._is_active = False

    @abstractmethod
    def is_full(self):
        """
        Returns whether there is room for additional players to join or not
        """
        pass

    @abstractmethod
    def apply_action(self, player_idx, action):
        """
        Updates the game state by applying a single (player_idx, action) tuple. Subclasses should try to override this method
        if possible
        """
        pass

    @abstractmethod
    def is_finished(self):
        """
        Returns whether the game has concluded or not
        """
        pass

    def is_ready(self):
        """
        Returns whether the game can be started. Defaults to having enough players
        """
        return self.is_full()

    @property
    def is_active(self):
        """
        Whether the game is currently being played
        """
        return self._is_active

    @property
    def reset_timeout(self):
        """
        Number of milliseconds to pause game on reset
        """
        return 3000

    def apply_actions(self):
        """
        Updates the game state by applying each of the pending actions in the buffer. Is called by the tick method. Subclasses
        should override this method if joint actions are necessary. If actions can be serialized, overriding `apply_action` is
        preferred
        """
        for i in range(len(self.players)):
            try:
                while True:
                    action = self.pending_actions[i].get(block=False)
                    self.apply_action(i, action)
            except Empty:
                pass

    def activate(self):
        """
        Activates the game to let server know real-time updates should start. Provides little functionality but useful as
        a check for debugging
        """
        self._is_active = True

    def deactivate(self):
        """
        Deactives the game such that subsequent calls to `tick` will be no-ops. Used to handle case where game ends but
        there is still a buffer of client pings to handle
        """
        self._is_active = False

    def reset(self):
        """
        Restarts the game while keeping all active players by resetting game stats and temporarily disabling `tick`
        """
        if not self.is_active:
            raise ValueError("Inactive Games cannot be reset")
        if self.is_finished():
            return self.Status.DONE
        self.deactivate()
        self.activate()
        return self.Status.RESET

    def needs_reset(self):
        """
        Returns whether the game should be reset on the next call to `tick`
        """
        return False

    def tick(self):
        """
        Updates the game state by applying each of the pending actions. This is done so that players cannot directly modify
        the game state, offering an additional level of safety and thread security.

        One can think of "enqueue_action" like calling "git add" and "tick" like calling "git commit"

        Subclasses should try to override `apply_actions` if possible. Only override this method if necessary
        """
        if not self.is_active:
            return self.Status.INACTIVE
        if self.needs_reset():
            self.reset()
            return self.Status.RESET

        self.apply_actions()
        return self.Status.DONE if self.is_finished() else self.Status.ACTIVE

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            self.pending_actions[player_idx].put(action)
        except Full:
            pass

    def get_state(self):
        """
        Return a JSON compatible serialized state of the game. Note that this should be as minimalistic as possible
        as the size of the game state will be the most important factor in game performance. This is sent to the client
        every frame update.
        """
        return {"players": self.players}

    def to_json(self):
        """
        Return a JSON compatible serialized state of the game. Contains all information about the game, does not need to
        be minimalistic. This is sent to the client only once, upon game creation
        """
        return self.get_state()

    def is_empty(self):
        """
        Return whether it is safe to garbage collect this game instance
        """
        return not self.num_players

    def add_player(self, player_id, idx=None, buff_size=-1):
        """
        Add player_id to the game
        """
        if self.is_full():
            raise ValueError("Cannot add players to full game")
        if self.is_active:
            raise ValueError("Cannot add players to active games")
        if not idx and self.EMPTY in self.players:
            idx = self.players.index(self.EMPTY)
        elif not idx:
            idx = len(self.players)

        padding = max(0, idx - len(self.players) + 1)
        for _ in range(padding):
            self.players.append(self.EMPTY)
            self.pending_actions.append(self.EMPTY)

        self.players[idx] = player_id
        self.pending_actions[idx] = Queue(maxsize=buff_size)

    def add_spectator(self, spectator_id):
        """
        Add spectator_id to list of spectators for this game
        """
        if spectator_id in self.players:
            raise ValueError("Cannot spectate and play at same time")
        self.spectators.add(spectator_id)

    def remove_player(self, player_id):
        """
        Remove player_id from the game
        """
        try:
            idx = self.players.index(player_id)
            self.players[idx] = self.EMPTY
            self.pending_actions[idx] = self.EMPTY
        except ValueError:
            return False
        else:
            return True

    def remove_spectator(self, spectator_id):
        """
        Removes spectator_id if they are in list of spectators. Returns True if spectator successfully removed, False otherwise
        """
        try:
            self.spectators.remove(spectator_id)
        except ValueError:
            return False
        else:
            return True

    def clear_pending_actions(self):
        """
        Remove all queued actions for all players
        """
        for i, player in enumerate(self.players):
            if player != self.EMPTY:
                queue = self.pending_actions[i]
                queue.queue.clear()

    @property
    def num_players(self):
        return len([player for player in self.players if player != self.EMPTY])

    def get_data(self):
        """
        Return any game metadata to server driver.
        """
        return {}


class DummyGame(Game):

    """
    Standin class used to test basic server logic
    """

    def __init__(self, **kwargs):
        super(DummyGame, self).__init__(**kwargs)
        self.counter = 0

    def is_full(self):
        return self.num_players == 2

    def apply_action(self, idx, action):
        pass

    def apply_actions(self):
        self.counter += 1

    def is_finished(self):
        return self.counter >= 100

    def get_state(self):
        state = super(DummyGame, self).get_state()
        state["count"] = self.counter
        return state


class DummyInteractiveGame(Game):

    """
    Standing class used to test interactive components of the server logic
    """

    def __init__(self, **kwargs):
        super(DummyInteractiveGame, self).__init__(**kwargs)
        self.max_players = int(
            kwargs.get("playerZero", "human") == "human"
        ) + int(kwargs.get("playerOne", "human") == "human")
        self.max_count = kwargs.get("max_count", 30)
        self.counter = 0
        self.counts = [0] * self.max_players

    def is_full(self):
        return self.num_players == self.max_players

    def is_finished(self):
        return max(self.counts) >= self.max_count

    def apply_action(self, player_idx, action):
        if action.upper() == Direction.NORTH:
            self.counts[player_idx] += 1
        if action.upper() == Direction.SOUTH:
            self.counts[player_idx] -= 1

    def apply_actions(self):
        super(DummyInteractiveGame, self).apply_actions()
        self.counter += 1

    def get_state(self):
        state = super(DummyInteractiveGame, self).get_state()
        state["count"] = self.counter
        for i in range(self.num_players):
            state["player_{}_count".format(i)] = self.counts[i]
        return state


class OvercookedGame(Game):
    """
    Class for bridging the gap between Overcooked_Env and the Game interface

    Instance variable:
        - max_players (int): Maximum number of players that can be in the game at once
        - mdp (OvercookedGridworld): Controls the underlying Overcooked game logic
        - score (int): Current reward acheived by all players
        - max_time (int): Number of seconds the game should last
        - npc_policies (dict): Maps user_id to policy (Agent) for each AI player
        - npc_state_queues (dict): Mapping of NPC user_ids to LIFO queues for the policy to process
        - curr_tick (int): How many times the game server has called this instance's `tick` method
        - ticker_per_ai_action (int): How many frames should pass in between NPC policy forward passes.
            Note that this is a lower bound; if the policy is computationally expensive the actual frames
            per forward pass can be higher
        - action_to_overcooked_action (dict): Maps action names returned by client to action names used by OvercookedGridworld
            Note that this is an instance variable and not a static variable for efficiency reasons
        - human_players (set(str)): Collection of all player IDs that correspond to humans
        - npc_players (set(str)): Collection of all player IDs that correspond to AI
        - randomized (boolean): Whether the order of the layouts should be randomized

    Methods:
        - npc_policy_consumer: Background process that asynchronously computes NPC policy forward passes. One thread
            spawned for each NPC
        - _curr_game_over: Determines whether the game on the current mdp has ended
    """

    def __init__(
        self,
        layouts=["cramped_room"],
        mdp_params={},
        num_players=2,
        gameTime=30,
        playerZero="human",
        playerOne="human",
        showPotential=False,
        randomized=False,
        ticks_per_ai_action=1,
        **kwargs
    ):
        super(OvercookedGame, self).__init__(**kwargs)
        self.show_potential = showPotential
        self.mdp_params = dict(mdp_params)
        self.layouts = layouts
        self.max_players = int(num_players)
        self.mdp = None
        self.mp = None
        self.score = 0
        self.phi = 0
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.npc_policies = {}
        self.npc_state_queues = {}
        self.action_to_overcooked_action = {
            "STAY": Action.STAY,
            "UP": Direction.NORTH,
            "DOWN": Direction.SOUTH,
            "LEFT": Direction.WEST,
            "RIGHT": Direction.EAST,
            "SPACE": Action.INTERACT,
        }
        self.ticks_per_ai_action = ticks_per_ai_action
        self.curr_tick = 0
        self.human_players = set()
        self.npc_players = set()
        self.threads = []
        self.uses_legacy_marl = False

        if randomized:
            random.shuffle(self.layouts)

        if playerZero != "human":
            player_zero_id = playerZero + "_0"
            self.add_player(player_zero_id, idx=0, buff_size=1, is_human=False)
            self.uses_legacy_marl = self.uses_legacy_marl or is_legacy_marl_agent(
                playerZero
            )
            self.npc_policies[player_zero_id] = self.get_policy(
                playerZero, idx=0
            )
            self.npc_state_queues[player_zero_id] = LifoQueue()

        if playerOne != "human":
            player_one_id = playerOne + "_1"
            self.add_player(player_one_id, idx=1, buff_size=1, is_human=False)
            self.uses_legacy_marl = self.uses_legacy_marl or is_legacy_marl_agent(
                playerOne
            )
            self.npc_policies[player_one_id] = self.get_policy(
                playerOne, idx=1
            )

            self.npc_state_queues[player_one_id] = LifoQueue()
        if self.uses_legacy_marl:
            self.mdp_params.setdefault("old_dynamics", True)
        # Always kill ray after loading agent, otherwise, ray will crash once process exits
        # Only kill ray after loading both agents to avoid having to restart ray during loading
        if ray is not None and ray.is_initialized():
            ray.shutdown()

        if kwargs["dataCollection"]:
            self.write_data = True
            self.write_config = kwargs["collection_config"]
        else:
            self.write_data = False

        self.trajectory = []

    def _curr_game_over(self):
        return time.time() - self.start_time >= self.max_time

    def needs_reset(self):
        return self._curr_game_over() and not self.is_finished()

    def add_player(self, player_id, idx=None, buff_size=-1, is_human=True):
        super(OvercookedGame, self).add_player(
            player_id, idx=idx, buff_size=buff_size
        )
        if is_human:
            self.human_players.add(player_id)
        else:
            self.npc_players.add(player_id)

    def remove_player(self, player_id):
        removed = super(OvercookedGame, self).remove_player(player_id)
        if removed:
            if player_id in self.human_players:
                self.human_players.remove(player_id)
            elif player_id in self.npc_players:
                self.npc_players.remove(player_id)
            else:
                raise ValueError("Inconsistent state")

    def npc_policy_consumer(self, policy_id):
        queue = self.npc_state_queues[policy_id]
        policy_ = self.npc_policies[policy_id]
        while self._is_active:
            state = queue.get()
            if not self._is_active:
                break
            try:
                result = policy_.action(state)
                npc_action = normalize_policy_action(result)
            except Exception as e:
                import traceback
                print("NPC {} action() failed: {}".format(policy_id, e))
                traceback.print_exc()
                npc_action = Action.STAY
            super(OvercookedGame, self).enqueue_action(policy_id, npc_action)


    def is_full(self):
        return self.num_players >= self.max_players

    def is_finished(self):
        val = not self.layouts and self._curr_game_over()
        return val

    def is_empty(self):
        """
        Game is considered safe to scrap if there are no active players or if there are no humans (spectating or playing)
        """
        return (
            super(OvercookedGame, self).is_empty()
            or not self.spectators
            and not self.human_players
        )

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        return super(OvercookedGame, self).is_ready() and not self.is_empty()

    def apply_action(self, player_id, action):
        pass

    def apply_actions(self):
        # Default joint action, as NPC policies and clients probably don't enqueue actions fast
        # enough to produce one at every tick
        joint_action = [Action.STAY] * len(self.players)

        # Synchronize individual player actions into a joint-action as required by overcooked logic
        for i in range(len(self.players)):
            # if this is a human, don't block and inject
            if self.players[i] in self.human_players:
                try:
                    # we don't block here in case humans want to Stay
                    joint_action[i] = self.pending_actions[i].get(block=False)
                except Empty:
                    pass
            else:
                # we block on agent actions to ensure that the agent gets to do one action per state
                joint_action[i] = self.pending_actions[i].get(block=True)

        # Apply overcooked game logic to get state transition
        prev_state = self.state
        transition_result = self.mdp.get_state_transition(prev_state, joint_action)
        if len(transition_result) == 3:
            # The repository's modified Overcooked runtime returns aggregate
            # sparse and shaped rewards as separate values.
            self.state, sparse_reward, shaped_reward = transition_result
            info = {
                "sparse_reward_by_agent": [sparse_reward],
                "shaped_reward_by_agent": [shaped_reward],
            }
        else:
            # Keep compatibility with the newer upstream demo API.
            self.state, info = transition_result
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                prev_state, self.mp, gamma=0.99
            )

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward

        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            "time_left": max(self.max_time - (time.time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time.time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": str(self.start_time),
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
        }

        self.trajectory.append(transition)

        # Return about the current transition
        return prev_state, joint_action, info

    def enqueue_action(self, player_id, action):
        overcooked_action = self.action_to_overcooked_action[action]
        super(OvercookedGame, self).enqueue_action(
            player_id, overcooked_action
        )

    def reset(self):
        status = super(OvercookedGame, self).reset()
        if status == self.Status.RESET:
            # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
            self.start_time += self.reset_timeout / 1000

    def tick(self):
        self.curr_tick += 1
        return super(OvercookedGame, self).tick()



    def activate(self):
        super(OvercookedGame, self).activate()

        # Sanity check
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")
        self.curr_layout = self.layouts.pop()
        # The public repository stores the five paper layouts under the
        # historical ProAgent ids used by the command-line evaluator.
        runtime_layout = get_legacy_layout_name(self.curr_layout)
        self.mdp = OvercookedGridworld.from_layout_name(runtime_layout, **self.mdp_params)
        self.mp = None
        if self.show_potential:
            try:
                self.mp = MotionPlanner.from_pickle_or_compute(self.mdp, counter_goals=[])
            except Exception:
                pass

        self.state = self.mdp.get_standard_start_state()
        self.curr_tick = 0
        self.score = 0
        self.threads = []
        decision_tree_mlam = None

        for npc_id in self.npc_policies:
            policy = self.npc_policies[npc_id]
            player_idx = self.players.index(npc_id)
            if hasattr(policy, "set_agent_index"):
                policy.set_agent_index(player_idx)
            if hasattr(policy, 'set_mdp'):
                policy.set_mdp(self.mdp)
            elif hasattr(policy, "mdp"):
                policy.mdp = self.mdp
            if is_decision_tree_policy(policy):
                if decision_tree_mlam is None:
                    decision_tree_mlam = build_decision_tree_mlam(self.mdp)
                policy.mlam = decision_tree_mlam
                policy.mdp = self.mdp
                # Force policy layout to match the current game layout instead of
                # using the value serialized in pickle.
                if hasattr(policy, "layout"):
                    policy.layout = self.curr_layout
                policy.policy_dir = get_decision_tree_policy_dir()
                policy.policy_file = os.path.join(
                    policy.policy_dir,
                    "{}_{}.txt".format(self.curr_layout, policy.agent_index),
                )
                # Decision tree code is lazily loaded and cached. Reset it on every
                # new game so layout/policy file changes take effect immediately.
                if hasattr(policy, "_predict_teammate_and_plan"):
                    policy._predict_teammate_and_plan = None
                print(
                    "DecisionTree policy configured: layout={}, dir={}, file={}, exists={}".format(
                        self.curr_layout,
                        policy.policy_dir,
                        policy.policy_file,
                        os.path.exists(policy.policy_file),
                    )
                )
            policy.reset()

        # Planner initialization can take several seconds on the first run.
        # Start the episode clock only after every policy is ready.
        self.start_time = time.time()

        for npc_id in self.npc_policies:
            self.npc_state_queues[npc_id].put(self.state)
            t = Thread(target=self.npc_policy_consumer, args=(npc_id,))
            t.daemon = True
            self.threads.append(t)
            t.start()



    def deactivate(self):
        super(OvercookedGame, self).deactivate()
        # Ensure the background consumers do not hang
        for npc_policy in self.npc_policies:
            try:
                self.npc_state_queues[npc_policy].put(self.state, block=False)
            except Full:
                pass

        # Wait for all background threads to exit
        for t in self.threads:
            t.join()

        # Clear all action queues
        self.clear_pending_actions()

    def get_state(self):
        state_dict = {}
        state_dict["potential"] = self.phi if self.show_potential else None
        state_dict["state"] = self.state.to_dict()
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time.time() - self.start_time), 0
        )
        return state_dict

    def to_json(self):
        obj_dict = {}
        obj_dict["terrain"] = self.mdp.terrain_mtx if self._is_active else None
        obj_dict["state"] = self.get_state() if self._is_active else None
        return obj_dict

    def get_policy(self, npc_id, idx=0):
        if npc_id.lower() == "llmtree":
            return DecisionTreeAgent(
                mlam=None,
                layout=self.layouts[-1],
                agent_index=idx,
                policy_dir=get_decision_tree_policy_dir(),
            )
        if npc_id.lower().startswith("rllib"):
            if load_agent is None:
                raise RuntimeError(
                    "RLlib support is not installed in the open-source human demo"
                )
            try:
                fpath = os.path.join(AGENT_DIR, npc_id, "agent")
                fix_bc_path(fpath)
                agent = load_agent(fpath, agent_index=idx)
                return agent
            except Exception as e:
                raise IOError(
                    "Error loading Rllib Agent\n{}".format(e.__repr__())
                )
        elif is_custom_bc_agent(npc_id):
            try:
                layout_name = self.layouts[-1]
                run_name = resolve_bc_run_name(npc_id, layout_name)
                # First, try loading a prebuilt pickle BC agent to avoid the
                # legacy stable-baselines runtime path.
                pickle_candidates = [
                    os.path.join(AGENT_DIR, npc_id, "agent.pickle"),
                    os.path.join(AGENT_DIR, "BC_{}".format(run_name), "agent.pickle"),
                    os.path.join(AGENT_DIR, "BC", "agent.pickle"),
                ]
                for pickle_path in pickle_candidates:
                    if os.path.exists(pickle_path):
                        try:
                            with open(pickle_path, "rb") as f:
                                agent = pickle.load(f)
                        except Exception as load_e:
                            raise IOError(
                                "Found BC pickle but failed to load: {}\n{}".format(
                                    pickle_path, repr(load_e)
                                )
                            )
                        if hasattr(agent, "set_agent_index"):
                            agent.set_agent_index(idx)
                        if hasattr(self, "mdp") and self.mdp is not None and hasattr(agent, "set_mdp"):
                            agent.set_mdp(self.mdp)
                        print("Loaded BC from pickle: {}".format(pickle_path))
                        return agent

                missing_msg = (
                    "BC agent.pickle not found.\n"
                    "Tried paths:\n- {}\n"
                    "Expected one of these to exist.\n"
                    "Recommended: place your prebuilt BC pickle at {}\n"
                    "Note: legacy GAIL BC runtime is disabled by default to avoid breaking SP/MEP."
                ).format(
                    "\n- ".join(pickle_candidates),
                    os.path.join(AGENT_DIR, "BC", "agent.pickle"),
                )

                if not ENABLE_GAIL_BC:
                    raise FileNotFoundError(missing_msg)

                print(missing_msg)
                print(
                    "Loading custom BC (legacy runtime): agent='{}' layout='{}' run='{}' root='{}'".format(
                        npc_id, layout_name, run_name, get_custom_bc_model_root()
                    )
                )
                agent = load_custom_bc_agent(run_name)
                agent.set_agent_index(idx)
                if hasattr(self, "mdp") and self.mdp is not None:
                    agent.set_mdp(self.mdp)
                return agent
            except Exception as e:
                raise IOError("Error loading custom BC agent\n{}".format(e.__repr__()))
        elif is_legacy_marl_agent(npc_id):
            try:
                layout_name = self.layouts[-1]
                pickle_layout = get_legacy_layout_name(layout_name)
                agent_root = os.path.join(
                    AGENT_DIR, "models", pickle_layout, npc_id
                )
                config_path = os.path.join(agent_root, "params.txt")
                if not os.path.exists(config_path):
                    raise IOError(f"params.txt not found in {agent_root}")

                config = load_dict_from_file(config_path)
                sim_threads = config.get("sim_threads", 1)

                agent = get_agent_from_saved_model(agent_root, sim_threads)
                agent.set_agent_index(idx)
                if hasattr(self, "mdp") and self.mdp is not None:
                    agent.set_mdp(self.mdp)
                return agent
            except Exception as e:
                raise IOError("Error loading agent\n{}".format(e.__repr__()))
        else:
            try:
                fpath = os.path.join(AGENT_DIR, npc_id, "agent.pickle")
                with open(fpath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise IOError("Error loading agent\n{}".format(e.__repr__()))
    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        data = {
            "uid": str(time.time()),
            "trajectory": self.trajectory,
        }
        self.trajectory = []
        # if we want to store the data and there is data to store
        if self.write_data and len(data["trajectory"]) > 0:
            configs = self.write_config
            # create necessary dirs
            data_path = create_dirs(configs, self.curr_layout)
            # the 3-layer-directory structure should be able to uniquely define any experiment
            with open(os.path.join(data_path, "result.pkl"), "wb") as f:
                pickle.dump(data, f)
        return data


class OvercookedTutorial(OvercookedGame):

    """
    Wrapper on OvercookedGame that includes additional data for tutorial mechanics, most notably the introduction of tutorial "phases"

    Instance Variables:
        - curr_phase (int): Indicates what tutorial phase we are currently on
        - phase_two_score (float): The exact sparse reward the user must obtain to advance past phase 2
    """

    def __init__(
        self,
        layouts=["tutorial_0"],
        mdp_params={},
        playerZero="human",
        playerOne="AI",
        phaseTwoScore=15,
        **kwargs
    ):
        super(OvercookedTutorial, self).__init__(
            layouts=layouts,
            mdp_params=mdp_params,
            playerZero=playerZero,
            playerOne=playerOne,
            showPotential=False,
            **kwargs
        )
        self.phase_two_score = phaseTwoScore
        self.phase_two_finished = False
        self.max_time = 0
        self.max_players = 2
        self.ticks_per_ai_action = 1
        self.curr_phase = 0
        # we don't collect tutorial data
        self.write_data = False

    @property
    def reset_timeout(self):
        return 1

    def needs_reset(self):
        if self.curr_phase == 0:
            return self.score > 0
        elif self.curr_phase == 1:
            return self.score > 0
        elif self.curr_phase == 2:
            return self.phase_two_finished
        return False

    def is_finished(self):
        return not self.layouts and self.score >= float("inf")

    def reset(self):
        super(OvercookedTutorial, self).reset()
        self.curr_phase += 1

    def get_policy(self, *args, **kwargs):
        return TutorialAI()

    def apply_actions(self):
        """
        Apply regular MDP logic with retroactive score adjustment tutorial purposes
        """
        _, _, info = super(OvercookedTutorial, self).apply_actions()

        human_reward, ai_reward = info["sparse_reward_by_agent"]

        # We only want to keep track of the human's score in the tutorial
        self.score -= ai_reward

        # Phase two requires a specific reward to complete
        if self.curr_phase == 2:
            self.score = 0
            if human_reward == self.phase_two_score:
                self.phase_two_finished = True


class DummyOvercookedGame(OvercookedGame):
    """
    Class that hardcodes the AI to be random. Used for debugging
    """

    def __init__(self, layouts=["cramped_room"], **kwargs):
        super(DummyOvercookedGame, self).__init__(layouts, **kwargs)

    def get_policy(self, *args, **kwargs):
        return DummyAI()


class DummyAI:
    """
    Randomly samples actions. Used for debugging
    """

    def action(self, state):
        [action] = random.sample(
            [
                Action.STAY,
                Direction.NORTH,
                Direction.SOUTH,
                Direction.WEST,
                Direction.EAST,
                Action.INTERACT,
            ],
            1,
        )
        return action, None

    def reset(self):
        pass


class DummyComputeAI(DummyAI):
    """
    Performs simulated compute before randomly sampling actions. Used for debugging
    """

    def __init__(self, compute_unit_iters=1e5):
        """
        compute_unit_iters (int): Number of for loop cycles in one "unit" of compute. Number of
                                    units performed each time is randomly sampled
        """
        super(DummyComputeAI, self).__init__()
        self.compute_unit_iters = int(compute_unit_iters)

    def action(self, state):
        # Randomly sample amount of time to busy wait
        iters = random.randint(1, 10) * self.compute_unit_iters

        # Actually compute something (can't sleep) to avoid scheduling optimizations
        val = 0
        for i in range(iters):
            # Avoid branch prediction optimizations
            if i % 2 == 0:
                val += 1
            else:
                val += 2

        # Return randomly sampled action
        return super(DummyComputeAI, self).action(state)


class StayAI:
    """
    Always returns "stay" action. Used for debugging
    """

    def action(self, state):
        return Action.STAY, None

    def reset(self):
        pass


class TutorialAI:
    COOK_SOUP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab second onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab third onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Cook soup
        Action.INTERACT,
        # Grab plate
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        Direction.WEST,
        Direction.NORTH,
        # Deliver soup
        Action.INTERACT,
        Direction.EAST,
        Direction.EAST,
        Direction.EAST,
        Action.INTERACT,
        Direction.WEST,
    ]

    COOK_SOUP_COOP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        # Move to start so this loops
        Direction.EAST,
        Direction.EAST,
        # Pause to make cooperation more real time
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
    ]

    def __init__(self):
        self.curr_phase = -1
        self.curr_tick = -1

    def action(self, state):
        self.curr_tick += 1
        if self.curr_phase == 0:
            return (
                self.COOK_SOUP_LOOP[self.curr_tick % len(self.COOK_SOUP_LOOP)],
                None,
            )
        elif self.curr_phase == 2:
            return (
                self.COOK_SOUP_COOP_LOOP[
                    self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)
                ],
                None,
            )
        return Action.STAY, None

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1

def get_agent_from_saved_model(save_dir, sim_threads):
    """Get Agent corresponding to a saved model"""
    # NOTE: Could remove dependency on sim_threads if get the sim_threads from config or dummy env
    state_policy, processed_obs_policy = get_model_policy_from_saved_model(save_dir, sim_threads)
    return AgentFromPolicy(state_policy, processed_obs_policy)

def get_agent_from_model(model, sim_threads, is_joint_action=False):
    """Get Agent corresponding to a loaded model"""
    state_policy, processed_obs_policy = get_model_policy_from_model(model, sim_threads, is_joint_action=is_joint_action)
    return AgentFromPolicy(state_policy, processed_obs_policy)

# def get_model_policy_from_saved_model(save_dir, sim_threads):
#     """Get a policy function from a saved model"""
#     if tf.__version__.startswith('1'):
#         predictor = tf.contrib.predictor.from_saved_model(save_dir) 
#         step_fn = lambda obs: predictor({"obs": obs})["action_probs"] 
#     elif tf.__version__.startswith('2'):
#         model = tf.saved_model.load(export_dir=save_dir, tags=[tag_constants.SERVING])
#         predictor = model.signatures['serving_default'] 
#         step_fn = lambda obs: predictor(**{"obs": tf.convert_to_tensor(obs, dtype=tf.float32)})["action_probs"] 
#     return get_model_policy(step_fn, sim_threads)
def get_model_policy_from_saved_model(save_dir, sim_threads):
    # Legacy MARL agents bundled with ProAgent were exported with an
    # observation signature of shape (sim_threads, H, W, 20). The current
    # demo emits a richer 26-channel encoding, so we keep the historical
    # channel subset that these agents were previously run with.
    keep_idx = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 14, 15,
        16, 20, 18, 22, 23,
    ]

    def preprocess_obs(obs):
        if obs.shape[-1] == len(keep_idx):
            return obs
        return obs[..., keep_idx]

    if tf.__version__.startswith('1'):
        predictor = tf.contrib.predictor.from_saved_model(save_dir)
        step_fn = lambda obs: predictor({"obs": preprocess_obs(obs)})[
            "action_probs"
        ]

    elif tf.__version__.startswith('2'):
        model = tf.saved_model.load(export_dir=save_dir, tags=[tag_constants.SERVING])
        predictor = model.signatures['serving_default']
        step_fn = lambda obs: predictor(
            **{
                "obs": tf.convert_to_tensor(
                    preprocess_obs(obs), dtype=tf.float32
                )
            }
        )["action_probs"]

    return get_model_policy(step_fn, sim_threads)

def get_model_policy_from_model(model, sim_threads, is_joint_action=False):
    def step_fn(obs):
        action_probs = model.act_model.step(obs, return_action_probs=True)
        return action_probs
    return get_model_policy(step_fn, sim_threads, is_joint_action=is_joint_action)

def get_model_policy(step_fn, sim_threads, is_joint_action=False):
    def encoded_state_policy(observations, stochastic=True, return_action_probs=False):
        """Takes in SIM_THREADS many losslessly encoded states and returns corresponding actions"""
        action_probs_n = step_fn(observations)

        if return_action_probs:
            return action_probs_n
        
        if stochastic:
            action_idxs = [np.random.choice(len(Action.ALL_ACTIONS), p=action_probs) for action_probs in action_probs_n]
        else:
            action_idxs = [np.argmax(action_probs) for action_probs in action_probs_n]

        return np.array(action_idxs)

    def state_policy(mdp_state, mdp, agent_index, stochastic=True, return_action_probs=False):
        """Takes in a Overcooked state object and returns the corresponding action"""
        obs = legacy_marl_lossless_state_encoding(mdp, mdp_state)[agent_index]
        padded_obs = np.array([obs] + [np.zeros(obs.shape)] * (sim_threads - 1))
        action_probs = step_fn(padded_obs)[0] # Discards all padding predictions
        if isinstance(action_probs, tf.Tensor):
            action_probs = action_probs.numpy()

        if return_action_probs:
            return action_probs

        if stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)

        if is_joint_action:
            action_idxs = Action.INDEX_TO_ACTION_INDEX_PAIRS[action_idx]
            joint_action = [Action.INDEX_TO_ACTION[i] for i in action_idxs]
            return joint_action

        return Action.INDEX_TO_ACTION[action_idx]

    return state_policy, encoded_state_policy
