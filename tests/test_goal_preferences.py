"""Regression checks for Co-π-tree's motion goal preferences."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src1"))
import project_paths  # noqa: F401 - selects the bundled Overcooked-AI package
from overcooked_ai_py.planning.planners import MediumLevelActionManager


NEAR_GOAL = ((0, 0), (1, 0))
COUNTER_GOAL = ((1, 0), (1, 0))
FAR_GOAL = ((3, 0), (-1, 0))
DISPENSER = (1, 0)
COUNTER = (2, 0)
PLAYERS = [((0, 0), (1, 0)), ((4, 0), (-1, 0))]


class FakeMotionPlanner:
    def is_valid_motion_start_goal_pair(self, start, goal):
        return goal[0][0] <= 1


def make_manager(counter_goal):
    manager = MediumLevelActionManager.__new__(MediumLevelActionManager)
    manager.motion_planner = FakeMotionPlanner()
    manager.mdp = SimpleNamespace(
        terrain_mtx=[list("  X  ")],
        get_onion_dispenser_locations=lambda: [DISPENSER],
        get_dish_dispenser_locations=lambda: [DISPENSER],
    )
    manager._get_ml_actions_for_positions = lambda positions: [
        NEAR_GOAL if position == DISPENSER else counter_goal
        for position in positions
    ]
    return manager


class GoalPreferenceTests(unittest.TestCase):
    def test_unreachable_counter_falls_back_to_dispenser_for_co_pi_tree(self):
        manager = make_manager(FAR_GOAL)
        for item, method in (
            ("onion", manager.pickup_onion_actions_new),
            ("dish", manager.pickup_dish_actions_new),
        ):
            objects = {item: [COUNTER]}
            self.assertEqual(method(None, objects, PLAYERS, 0), [FAR_GOAL])
            self.assertEqual(
                method(None, objects, PLAYERS, 0, prefer_reachable_counter=True),
                [NEAR_GOAL],
            )

    def test_reachable_counter_is_preferred(self):
        manager = make_manager(COUNTER_GOAL)
        for item, method in (
            ("onion", manager.pickup_onion_actions_new),
            ("dish", manager.pickup_dish_actions_new),
        ):
            self.assertEqual(
                method(None, {item: [COUNTER]}, PLAYERS, 0,
                       prefer_reachable_counter=True),
                [COUNTER_GOAL],
            )

    def test_ready_pot_is_preferred_over_cooking_pot(self):
        manager = make_manager(FAR_GOAL)
        pots = {
            "onion": {"ready": [DISPENSER], "cooking": [COUNTER]},
            "tomato": {"ready": [], "cooking": []},
        }
        self.assertEqual(
            manager.pickup_soup_with_dish_actions(pots, only_nearly_ready=True),
            [NEAR_GOAL, FAR_GOAL],
        )
        self.assertEqual(
            manager.pickup_soup_with_dish_actions(
                pots, only_nearly_ready=True, prefer_ready=True,
                start_pos_and_or=PLAYERS[0]
            ),
            [NEAR_GOAL],
        )

        pots["onion"]["ready"] = [COUNTER]
        pots["onion"]["cooking"] = [DISPENSER]
        self.assertEqual(
            manager.pickup_soup_with_dish_actions(
                pots, only_nearly_ready=True, prefer_ready=True,
                start_pos_and_or=PLAYERS[0]
            ),
            [NEAR_GOAL],
        )


if __name__ == "__main__":
    unittest.main()
