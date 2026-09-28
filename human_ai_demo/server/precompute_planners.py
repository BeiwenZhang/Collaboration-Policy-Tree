"""Precompute the planner caches used by the five public human-study layouts."""

from game import build_decision_tree_mlam, get_legacy_layout_name
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


PUBLIC_LAYOUTS = (
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
)


for public_layout in PUBLIC_LAYOUTS:
    runtime_layout = get_legacy_layout_name(public_layout)
    mdp = OvercookedGridworld.from_layout_name(runtime_layout)
    build_decision_tree_mlam(mdp)
    print("Planner ready: {} -> {}".format(public_layout, runtime_layout))

