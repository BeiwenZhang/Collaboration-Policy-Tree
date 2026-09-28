import itertools, os, json, re
from collections import defaultdict
import numpy as np
import pkg_resources
import sys 
import copy
from project_paths import POLICY_DIR as PROJECT_POLICY_DIR

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.search import find_path 
from overcooked_ai_py.planning.search import get_intersect_counter 
from overcooked_ai_py.planning.search import query_counter_states 

# no inferred teammate skill
POLICY_DIR = str(PROJECT_POLICY_DIR)


NAME_TO_ACTION = {
	"NORTH": Direction.NORTH,
	"SOUTH": Direction.SOUTH,
	"EAST": Direction.EAST,
	"WEST": Direction.WEST,
	"INTERACT": Action.INTERACT,
	"STAY": Action.STAY
}


class LlmtreeAgent:
	def __init__(self, mlam, layout, agent_index=None,auto_unstuck=True):
		self.trace = True 
		self.debug_mode = 'Y'
		self.controller_mode = 'new' 
		self.mlam = mlam
		self.layout = layout
		self.mdp = self.mlam.mdp
		self.agent_index = agent_index

		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.pot_id_to_pos = []
		self.teammate_last_completed_skill = None
		self.self_last_skill = None
		self.last_planned_self_skill = None
		self.last_validation_reason = ''
		self.last_motion_goal_source = ''
		self.last_motion_goal_debug = []
		self.auto_unstuck = auto_unstuck
		func_file = os.path.join(POLICY_DIR, f'{layout}_{agent_index}.txt')
		with open(func_file, "r") as f:
			code = f.read()
		local_vars = {}
		exec(code, {}, local_vars)
		self.predict_teammate_and_plan = local_vars["PredictTeammateThenPlan"]

	def set_mdp(self, mdp):
		self.mdp = mdp

	def reset(self):
		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.current_timestep = 0
		self.teammate_ml_actions_dict = {}
		self.teammate_intentions_dict = {}
		self.teammate_last_completed_skill = None
		self.self_last_skill = None
		self.last_planned_self_skill = None
		self.last_validation_reason = ''
		self.last_motion_goal_source = ''
		self.last_motion_goal_debug = []
		self.prev_state = None
		self.prev_prev_state = None

	def set_agent_index(self, agent_index):
		self.agent_index = agent_index

	def _format_goal(self, goal):
		if goal is None:
			return "None"
		try:
			return f"pos={goal[0]}, orient={goal[1]}"
		except Exception:
			return str(goal)

	def _format_action_for_log(self, action):
		if action is None:
			return "None"
		if isinstance(action, tuple):
			return str(action)
		try:
			return Action.to_char(action)
		except Exception:
			return str(action)

	def _append_motion_goal_log(self, lines):
		self.last_motion_goal_debug = list(lines)


	# def action(self, state,**kwargs):
	# 	if self.prev_state is None:
	# 		self.prev_state = state
	# 	self.state_prompt = self.generate_state_prompt(self.prev_state)
	# 	self.state_str=self.mdp.state_string(self.prev_state).replace('ø', 'o')
	# 	self.predict_prompt = ''
	# 	self.plan_prompt = ''
		
	# 	# ((pos1, orient1), (pos2, orient2))
	# 	start_pos_and_or = state.players_pos_and_or[self.agent_index]
	# 	# only use to record the teammate ml_action, 
	# 	# if teammate finish ml_action in t-1, it will record in s_t, 
	# 	# otherwise, s_t will just record None
	# 	self.current_timestep = state.timestep


	# 	if state.ml_actions[1-self.agent_index] != None:
	# 		self.teammate_ml_actions_dict[str(self.current_timestep-1)] = state.ml_actions[1-self.agent_index]
	# 		self.teammate_last_completed_skill = state.ml_actions[1-self.agent_index]

	# 	if self.current_ml_action is None:
	# 		self.current_ml_action = self.generate_ml_action(state)
	# 	if self.current_ml_action_steps > 0 and self.check_current_ml_action_done(state):
	# 		self.current_ml_action = self.generate_ml_action(state)
	# 	# print(f'self action: {self.current_ml_action}')
	# 	if not self.validate_current_ml_action(state):
	# 		# print(f'self action: {self.current_ml_action} is not valid')
	# 		self.plan_prompt = self.generate_plan_feedback(self.agent_index, self.current_ml_action)
	# 		self.current_ml_action = "wait(1)"
	# 		self.time_to_wait = 1
	# 	# else:
	# 	# 	self.plan_prompt = self.generate_plan_feedback(self.agent_index, self.current_ml_action)

	# 	if "wait" in self.current_ml_action:
	# 		self.current_ml_action_steps += 1
	# 		self.time_to_wait -= 1
	# 		chosen_action = Action.STAY
	# 	else:
	# 		#The method mainly uses self.current_ml_action internally
	# 		possible_motion_goals = self.find_motion_goals(state)  
	# 		current_motion_goal, chosen_action = self.choose_motion_goal(
	# 			start_pos_and_or, 
	# 			possible_motion_goals, 
	# 			state
	# 		)



	# 	if self.auto_unstuck and chosen_action != Action.INTERACT:

	# 		if (self.prev_state is not None and 
	# 			state.players_pos_and_or== self.prev_state.players_pos_and_or):


	# 			# Create possible joint actions to unblock the agent
	# 			if self.agent_index == 0:
	# 				joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
	# 			else:  # agent_index == 1
	# 				joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
				
	# 			# Find actions that would change the agent's position
	# 			unblocking_joint_actions = []
	# 			for j_a in joint_actions:
	# 				# Skip action combinations involving INTERACT as they might interfere with normal operations
	# 				if j_a != (Action.INTERACT, Action.STAY) and j_a != (Action.STAY, Action.INTERACT):
	# 					if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
	# 						new_state, _ = self.mdp.get_state_transition(state, j_a)
	# 					elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
	# 						new_state, _, _ = self.mdp.get_state_transition(state, j_a)
							
	# 					if new_state.players_pos_and_or!= self.prev_state.players_pos_and_or:
	# 						unblocking_joint_actions.append(j_a)
				
	# 			# Add STAY action as fallback
	# 			if not unblocking_joint_actions:
	# 				unblocking_joint_actions.append((Action.STAY, Action.STAY))
				
	# 			# Randomly select an unblocking action
	# 			if unblocking_joint_actions:
	# 				chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][self.agent_index]
	# 				# print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')

	# 	if chosen_action is None:
	# 		self.current_ml_action = "wait(1)"
	# 		self.time_to_wait = 1
	# 		chosen_action = Action.STAY
	# 	self.current_ml_action_steps += 1

	# 	# Update the previous state
	# 	self.prev_state = state

	# 	# print(f'ml_action = {self.current_ml_action}') 
	# 	# print(f'teammate_action = {state.ml_actions[1-self.agent_index]}') 
	# 	# print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')
	# 	if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
	# 		return chosen_action, {}
	# 	elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
	# 		return chosen_action
		
	def action(self, state, **kwargs):
			# 1. State initialization and maintenance
			if self.prev_state is None:
				self.prev_state = state
			
			# Ensure prev_prev_state exists on the class instance
			if not hasattr(self, 'prev_prev_state'):
				self.prev_prev_state = None

				# Keep the logged scene aligned with the same state used for planning.
			self.state_prompt = self.generate_state_prompt(state)
			self.state_str = self.mdp.state_string(state).replace('ø', 'o')
			self.predict_prompt = ''
			self.plan_prompt = ''
			
			# Get the current agent position and orientation
			start_pos_and_or = state.players_pos_and_or[self.agent_index]
			
			self.current_timestep = state.timestep

			# Process records of completed teammate skills
			if state.ml_actions[1-self.agent_index] is not None:
				self.teammate_ml_actions_dict[str(self.current_timestep-1)] = state.ml_actions[1-self.agent_index]
				self.teammate_last_completed_skill = state.ml_actions[1-self.agent_index]

			# 2. Generate/validate the current high-level action (ML Action)
			# if self.current_ml_action is None:
			# 	self.current_ml_action = self.generate_ml_action(state)
			
			# if self.current_ml_action_steps > 0 and self.check_current_ml_action_done(state):
			self.current_ml_action = self.generate_ml_action(state)
			is_valid, validation_reason = self.validate_current_ml_action(state)
			self.last_validation_reason = validation_reason
			self.plan_prompt = self.generate_plan_feedback(
				self.agent_index,
				self.current_ml_action,
				self.last_planned_self_skill,
				is_valid,
				validation_reason
			)
				
			if not is_valid:
				self.current_ml_action = "wait(1)"
				self.time_to_wait = 1

			# 3. Convert the high-level action into a concrete primitive action (chosen_action)
			if "wait" in self.current_ml_action:
				self.current_ml_action_steps += 1
				self.time_to_wait -= 1
				chosen_action = Action.STAY
			else:
				possible_motion_goals = self.find_motion_goals(state)  
				current_motion_goal, chosen_action = self.choose_motion_goal(
					start_pos_and_or, 
					possible_motion_goals, 
					state
				)

			# 4. Anti-congestion logic (Unstuck Logic)
			# Logic A: position unchanged for three consecutive frames (highest priority, ignores INTERACT)
			force_unstick = False
			if self.prev_prev_state is not None and self.prev_state is not None:
				if state.players_pos_and_or == self.prev_state.players_pos_and_or == self.prev_prev_state.players_pos_and_or:
					force_unstick = True

			# Logic B: original two-frame no-movement check (limited to non-INTERACT actions)
			is_stuck_two_steps = (self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or)

			# Compute the unstuck action
			if self.auto_unstuck and (force_unstick or (is_stuck_two_steps and chosen_action != Action.INTERACT)):
				# Determine the joint action space by agent index and force the other agent to STAY
				if self.agent_index == 0:
					joint_actions_space = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
				else:  # agent_index == 1
					joint_actions_space = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
				
				unblocking_joint_actions = []
				for j_a in joint_actions_space:
					# Avoid INTERACT action combinations that may affect the environment state
					if j_a != (Action.INTERACT, Action.STAY) and j_a != (Action.STAY, Action.INTERACT):
						# Compatible with different versions of the Overcooked library
						if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
							new_state, _ = self.mdp.get_state_transition(state, j_a)
						elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
							new_state, _, _ = self.mdp.get_state_transition(state, j_a)
							
						# Check whether this action can change the current position
						if new_state.players_pos_and_or != state.players_pos_and_or:
							unblocking_joint_actions.append(j_a)
				
				# Randomly select an action that can break the stalemate
				if unblocking_joint_actions:
					chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][self.agent_index]
				else:
					chosen_action = Action.STAY

			# 5. Final fallback and state update
			if chosen_action is None:
				self.current_ml_action = "wait(1)"
				self.time_to_wait = 1
				chosen_action = Action.STAY
				
			self.current_ml_action_steps += 1

			# Update the historical state chain
			self.prev_prev_state = self.prev_state
			self.prev_state = state

			# Return the action
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				return chosen_action, {}
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				return chosen_action

	def Parse_state(self, state):
		state_dict = {'hold': [None, None],
					'pot': [],
					'any_soup_ready': None, 
					'any_pot_not_full': None,
					'teammate_last_completed_skill': None,
					'num_empty_counters': 0,
					'num_onion_counters':0,
					'num_dish_counters':0,
					'self_last_skill': None,}

		# hold: onion, dish, soup
		ego = state.players[self.agent_index]
		state_dict['hold'][self.agent_index] = ego.held_object.name if ego.held_object else 'empty'
		teammate = state.players[1 - self.agent_index]
		state_dict['hold'][1 - self.agent_index] = teammate.held_object.name if teammate.held_object else 'empty'

		# pot: count, state, timer
		for pot in self.mdp.get_pot_locations():
			if state.has_object(pot):
				soup = state.get_object(pot)
				soup_type, count, cook = soup.state
				if count < self.mdp.num_items_for_soup:
					st, timer = 'idle', None
				elif cook < self.mdp.soup_cooking_time:
					st, timer = 'cooking', (self.mdp.soup_cooking_time - cook)
				else:
					st, timer = 'ready', 0
			else:
				count, st, timer = 0, 'idle', None
			state_dict['pot'].append({'count': count, 'state': st, 'timers': timer})

		# any_soup_ready, any_pot_not_full
		state_dict['any_soup_ready'] = any(p['state'] == 'ready' for p in state_dict['pot'])

		# all_empty_counters = set(self.mdp.get_empty_counter_locations(state))
		# counter_list = get_intersect_counter(state.players_pos_and_or[self.agent_index],
		# 				state.players_pos_and_or[1 - self.agent_index], 
		# 				self.mdp, 
		# 				self.mlam)   

		# available_counters = all_empty_counters.intersection(counter_list)
		# state_dict['num_empty_counters'] = len(available_counters)

		counter_states = query_counter_states(self.mdp, state)
		shared_counters = get_intersect_counter(
			state.players_pos_and_or[self.agent_index],
			state.players_pos_and_or[1 - self.agent_index],
			self.mdp,
			self.mlam
		)

		shared_counter_states = {pos: counter_states[pos] for pos in shared_counters}
		shared_empty_counters = [pos for pos in shared_counters if counter_states[pos] == ' ']
		shared_onion_counters = [pos for pos in shared_counters if counter_states[pos] == 'onion']
		shared_dish_counters = [pos for pos in shared_counters if counter_states[pos] == 'dish']
		state_dict['num_empty_counters'] = len(shared_empty_counters)
		state_dict['num_onion_counters'] = len(shared_onion_counters)
		state_dict['num_dish_counters'] = len(shared_dish_counters)
		# print("shared_empty_counters:",len(shared_empty_counters))
		# print("num_onion_counters:",len(shared_onion_counters))
		# print("num_dish_counters:",len(shared_dish_counters))
		state_dict['any_pot_not_full'] = any(p['count'] < self.mdp.num_items_for_soup for p in state_dict['pot'])

		state_dict['teammate_last_completed_skill'] = self.teammate_last_completed_skill
		state_dict['self_last_skill'] = self.self_last_skill

		return state_dict

	# def Parse_state(self, state):
	# 	state_dict = {
	# 		'hold': [None, None],
	# 		'pot': [],
	# 		'any_soup_ready': None,
	# 		'any_pot_not_full': None,
	# 		'empty_counter_locations': [],
	# 		'num_empty_counters': 0,
	# 		'teammate_last_completed_skill': None,
	# 		'self_last_skill': None,
	# 	}

	# 	ego = state.players[self.agent_index]
	# 	state_dict['hold'][self.agent_index] = ego.held_object.name if ego.held_object else 'empty'
	# 	teammate = state.players[1 - self.agent_index]
	# 	state_dict['hold'][1 - self.agent_index] = teammate.held_object.name if teammate.held_object else 'empty'

	# 	for pot in self.mdp.get_pot_locations():
	# 		if state.has_object(pot):
	# 			soup = state.get_object(pot)
	# 			soup_type, count, cook = soup.state
	# 			if count < self.mdp.num_items_for_soup:
	# 				st, timer = 'idle', None
	# 			elif cook < self.mdp.soup_cooking_time:
	# 				st, timer = 'cooking', (self.mdp.soup_cooking_time - cook)
	# 			else:
	# 				st, timer = 'ready', 0
	# 		else:
	# 			count, st, timer = 0, 'idle', None
	# 		state_dict['pot'].append({'count': count, 'state': st, 'timers': timer})

	# 	empty_counters = list(self.mdp.get_empty_counter_locations(state))
	# 	state_dict['empty_counter_locations'] = empty_counters
	# 	state_dict['num_empty_counters'] = len(empty_counters)

	# 	state_dict['any_soup_ready'] = any(p['state'] == 'ready' for p in state_dict['pot'])
	# 	state_dict['any_pot_not_full'] = any(p['count'] < self.mdp.num_items_for_soup for p in state_dict['pot'])

	# 	state_dict['teammate_last_completed_skill'] = self.teammate_last_completed_skill
	# 	state_dict['self_last_skill'] = self.self_last_skill

	# 	return state_dict

	def generate_ml_action(self, state):
		self_skill = self.predict_teammate_and_plan(self.Parse_state(state))
		if self_skill not in ['pickup_onion', 'put_onion_in_pot', 'pickup_dish', 'fill_dish_with_soup', 'deliver_soup', 'place_obj_on_counter', 'wait(1)',None]:
			raise ValueError(f"Invalid action string: {self_skill}")

		self.current_ml_action_steps = 0
		self.self_last_skill = self.current_ml_action
		self.last_planned_self_skill = self_skill if self_skill is not None else 'wait(1)'
		return self_skill if self_skill is not None else 'wait(1)'

	def check_current_ml_action_done(self,state):
		player = state.players[self.agent_index]
		# pot_states_dict = self.mlam.mdp.get_pot_states(state)
		if "pickup" in self.current_ml_action:
			pattern = r"pickup(?:[(]|_)(\w+)(?:[)]|)" # fit both pickup(onion) and pickup_onion
			obj_str = re.search(pattern, self.current_ml_action).group(1)
			return player.has_object() and player.get_object().name == obj_str
		
		elif "fill" in self.current_ml_action:
			return player.held_object.name == 'soup'
		
		elif "put" in self.current_ml_action or "place" in self.current_ml_action:
			return not player.has_object()
		
		elif "deliver" in self.current_ml_action:
			return not player.has_object()
		
		elif "wait" in self.current_ml_action:
			return self.time_to_wait == 0
		
	def validate_current_ml_action(self, state):
		if self.current_ml_action is None:
			return False, "no macro action was generated"

		pot_states_dict = self.mdp.get_pot_states(state)
		player = state.players[self.agent_index]
		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			soup_cooking = len(pot_states_dict['cooking']) > 0
			soup_ready = len(pot_states_dict['ready']) > 0
			pot_not_full = pot_states_dict["empty"] + self.mdp.get_partially_full_pots(pot_states_dict)
			cookable_pots = self.mdp.get_full_but_not_cooking_pots(pot_states_dict)
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			soup_cooking = len(pot_states_dict['onion']['cooking'])+len(pot_states_dict['tomato']['cooking']) > 0
			soup_ready = len(pot_states_dict['onion']['ready'])+len(pot_states_dict['tomato']['ready']) > 0
			pot_not_full = pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full'] + pot_states_dict["tomato"]['partially_full']
			cookable_pots = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)] # pot has max onions/tomotos

		has_onion = False
		has_tomato = False
		has_dish = False
		has_soup = False
		has_object = player.has_object()
		if has_object:
			has_onion = player.get_object().name == 'onion'
			has_tomato = player.get_object().name == 'tomato'
			has_dish = player.get_object().name == 'dish'
			has_soup = player.get_object().name == 'soup'
		empty_counter = self.mdp.get_empty_counter_locations(state)

		if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:   
			flag2 = len(self.find_motion_goals(state)) == 0 
			if flag2: 
				return False, "no reachable onion pickup motion goal"
			if has_object:
				return False, "cannot pickup onion while already holding an object"
			if len(self.mdp.get_onion_dispenser_locations()) == 0:
				return False, "no onion dispenser available"
			return True, "pickup_onion is executable"
		if self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
			if has_object:
				return False, "cannot pickup tomato while already holding an object"
			if len(self.mdp.get_tomato_dispenser_locations()) == 0:
				return False, "no tomato dispenser available"
			return True, "pickup_tomato is executable"
		elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
			flag2 = len(self.find_motion_goals(state)) == 0 
			if flag2: 
				return False, "no reachable dish pickup motion goal"
			if has_object:
				return False, "cannot pickup dish while already holding an object"
			if len(self.mdp.get_dish_dispenser_locations()) == 0:
				return False, "no dish dispenser available"
			return True, "pickup_dish is executable"
		elif "put_onion_in_pot" in self.current_ml_action:
			if not has_onion:
				return False, "cannot put onion in pot without holding an onion"
			if len(pot_not_full) == 0:
				return False, "all pots are already full"
			return True, "put_onion_in_pot is executable"
		elif "put_tomato_in_pot" in self.current_ml_action:
			if not has_tomato:
				return False, "cannot put tomato in pot without holding a tomato"
			if len(pot_not_full) == 0:
				return False, "all pots are already full"
			return True, "put_tomato_in_pot is executable"
		elif "place_obj_on_counter" in self.current_ml_action:
			if not has_object:
				return False, "cannot place object on counter with empty hands"
			if len(empty_counter) == 0:
				return False, "no empty counter is available"
			return True, "place_obj_on_counter is executable"
		elif "fill_dish_with_soup" in self.current_ml_action:
			if not has_dish:
				return False, "cannot fill dish with soup without holding a dish"
			if not (soup_ready or soup_cooking):
				return False, "no soup is ready or cooking"
			return True, "fill_dish_with_soup is executable"
		elif "deliver_soup" in self.current_ml_action:
			if not has_soup:
				return False, "cannot deliver soup without holding soup"
			return True, "deliver_soup is executable"
		elif "wait" in self.current_ml_action:
			try:
				wait_time = int(self.current_ml_action.split('(')[1][:-1])
				if not (0 < wait_time <= 20):
					return False, "wait time must be between 1 and 20"
				return True, f"wait({wait_time}) is executable"
			except (IndexError, ValueError):
				return False, "invalid wait action format"

		return False, f"unknown macro action: {self.current_ml_action}"

	def generate_layout_prompt(self):
		layout_prompt_dict = {
			"onion_dispenser": " <Onion Dispenser {id}>",
			"dish_dispenser": " <Dish Dispenser {id}>",
			"tomato_dispenser": " <Tomato Dispenser {id}>",
			"serving": " <Serving Loc {id}>",
			"pot": " <Pot {id}>",
		}
		layout_prompt = "Here's the layout of the kitchen:"
		for obj_type, prompt_template in layout_prompt_dict.items():
			locations = getattr(self.mdp, f"get_{obj_type}_locations")()
			for obj_id, obj_pos in enumerate(locations):
				layout_prompt += prompt_template.format(id=obj_id) + ","
				if obj_type == "pot":
					self.pot_id_to_pos.append(obj_pos)
		layout_prompt = layout_prompt[:-1] + ".\n"
		return layout_prompt

	def generate_state_prompt(self, state):
		ego = state.players[self.agent_index]
		teammate = state.players[1 - self.agent_index]

		time_prompt = f"Scene {state.timestep}: "
		ego_object = ego.held_object.name if ego.held_object else "nothing"
		teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
		ego_state_prompt = f"<Player {self.agent_index}> holds "
		if ego_object == 'soup':
			ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
		elif ego_object == 'nothing':
			ego_state_prompt += f"{ego_object}. "
		else:
			ego_state_prompt += f"one {ego_object}. "
		teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
		if teammate_object == 'soup':
			teammate_state_prompt += f"a dish with {teammate_object}. "
		elif teammate_object == "nothing":
			teammate_state_prompt += f"{teammate_object}. "
		else:
			teammate_state_prompt += f"one {teammate_object}. "

		kitchen_state_prompt = "Kitchen states: "
		prompt_dict = {
			"empty": "<Pot {id}> is empty; ",
			"cooking": "<Pot {id}> starts cooking, the soup will be ready after {t} timesteps; ",
			"ready": "<Pot {id}> has already cooked the soup; ",
			"1_items": "<Pot {id}> has 1 onion; ",
			"2_items": "<Pot {id}> has 2 onions; ",
			"3_items": "<Pot {id}> has 3 onions and is full; "
		}
		pot_states_dict = self.mdp.get_pot_states(state)   

		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			for key in pot_states_dict.keys():
				if key == "cooking":
					for pos in pot_states_dict[key]:
						pot_id = self.pot_id_to_pos.index(pos)
						soup_object = state.get_object(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id, t=soup_object.cook_time_remaining)
				else:
					for pos in pot_states_dict[key]:
						pot_id = self.pot_id_to_pos.index(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id) 
		
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			for key in pot_states_dict.keys():
				if key == "empty":
					for pos in pot_states_dict[key]: 
						# print("pos:",pos)
						pot_id = self.pot_id_to_pos.index(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id)     
				else: # key = 'onion' or 'tomota'
					for soup_key in pot_states_dict[key].keys():
						# soup_key: ready, cooking, partially_full
						for pos in pot_states_dict[key][soup_key]:
							pot_id = self.pot_id_to_pos.index(pos)
							soup_object = state.get_object(pos)
							soup_type, num_items, cook_time = soup_object.state
							if soup_key == "cooking":
								kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id, t=self.mdp.soup_cooking_time-cook_time)
							elif soup_key == "partially_full":
								pass
							else:
								kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id)
 
		intersect_counters = get_intersect_counter(
								state.players_pos_and_or[self.agent_index], 
								state.players_pos_and_or[1 - self.agent_index], 
								self.mdp, 
								self.mlam
							)
		counter_states = query_counter_states(self.mdp, state)  

		if self.layout == 'forced_coordination': 
			kitchen_state_prompt += '{} counters can be visited by <Player {}>. Their states are as follows: '.format(len(intersect_counters), self.agent_index)
			count_states = {}  
			for i in intersect_counters:  
				obj_i = 'nothing' 
				if counter_states[i] != ' ': 
					obj_i = counter_states[i]                
				if obj_i in count_states:  
					count_states[obj_i] += 1
				else: 
					count_states[obj_i]  = 1 
			total_obj = ['onion', 'dish']
			for i in count_states:   
				if i == 'nothing': 
					continue 
				kitchen_state_prompt += f'{count_states[i]} counters have {i}. '   
			for i in total_obj: 
				if i not in count_states:        
					kitchen_state_prompt += f'No counters have {i}. ' 

		if self.layout == 'forced_coordination': 
			teammate_state_prompt = ""
		return (time_prompt + ego_state_prompt + teammate_state_prompt + kitchen_state_prompt)

	def generate_plan_feedback(self, agent_index, current_ml_action,
							planned_self_skill=None,
							is_valid=None, validation_reason=""):
		parts = [
			f'<Player {agent_index}> planned self skill "{planned_self_skill}".'
		]
		if is_valid is True:
			parts.append(f'The macro action is valid: {validation_reason}.')
		elif is_valid is False:
			parts.append(f'The macro action is invalid: {validation_reason}.')
		return " ".join(parts)


	def generate_step_feedback(self):
		if self.state_str != '' or self.predict_prompt != '' or self.plan_prompt != '':
			return {'state': self.state_prompt,
			        # 'state_demo': "\n"+self.state_str,
					'predict': self.predict_prompt,
					'plan': self.plan_prompt,}
		else:
			return None

	##################
	'''
	The followings are the Controller part almost inherited from GreedyHumanModel class
	'''
	##################	
		
	def find_shared_counters(self, state, mlam):  
		counter_dicts = query_counter_states(self.mdp, state) 

		counter_list  = get_intersect_counter(state.players_pos_and_or[self.agent_index],
						state.players_pos_and_or[1 - self.agent_index], 
						self.mdp, 
						self.mlam
					)    

		# print('counter_list = {}'.format(counter_list))  
		lis = [] 
		for i in counter_list:  
			if counter_dicts[i] == ' ':  
				lis.append(i)       
		available_plans = mlam._get_ml_actions_for_positions(lis)
		return available_plans          
	#Directly find the optimal primitive action for the macro action
	def find_motion_goals(self, state):
		"""
		Generates the motion goals for the given medium level action.
		:param state:
		:return:
		"""
		am = self.mlam
		motion_goals = []
		player = state.players[self.agent_index]
		pot_states_dict = self.mdp.get_pot_states(state)
		counter_objects = self.mdp.get_counter_objects_dict(
			state, list(self.mdp.terrain_pos_dict["X"])
		)
		if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:
			self.last_motion_goal_source = "pickup_onion_actions_new"
			motion_goals = am.pickup_onion_actions_new(
				state, counter_objects, state.players_pos_and_or, self.agent_index,
				prefer_reachable_counter=True
			)

		elif self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
			self.last_motion_goal_source = "pickup_tomato_actions"
			motion_goals = am.pickup_tomato_actions(state, counter_objects)
		elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
			self.last_motion_goal_source = "pickup_dish_actions_new"
			motion_goals = am.pickup_dish_actions_new(
				state, counter_objects, state.players_pos_and_or, self.agent_index,
				prefer_reachable_counter=True
			)
		elif "put_onion_in_pot" in self.current_ml_action:
			self.last_motion_goal_source = "put_onion_in_pot_actions"
			motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
		elif "put_tomato_in_pot" in self.current_ml_action:
			self.last_motion_goal_source = "put_tomato_in_pot_actions"
			motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
		elif "place_obj_on_counter" in self.current_ml_action:  
			self.last_motion_goal_source = "find_shared_counters"
			motion_goals = self.find_shared_counters(state, self.mlam)     
			if len(motion_goals) == 0: 
				self.last_motion_goal_source = "place_obj_on_counter_actions_fallback"
				motion_goals = am.place_obj_on_counter_actions(state)

		elif "start_cooking" in self.current_ml_action:
			self.last_motion_goal_source = "start_cooking_actions"
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				next_order = list(state.all_orders)[0]
				soups_ready_to_cook_key = "{}_items".format(len(next_order.ingredients))
				soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				soups_ready_to_cook = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)]
			only_pot_states_ready_to_cook = defaultdict(list)
			only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
			motion_goals = am.start_cooking_actions(only_pot_states_ready_to_cook)
		elif "fill_dish_with_soup" in self.current_ml_action:
			self.last_motion_goal_source = "pickup_soup_with_dish_actions_prefer_ready"
			motion_goals = am.pickup_soup_with_dish_actions(
				pot_states_dict, only_nearly_ready=True,
				prefer_ready=True, start_pos_and_or=player.pos_and_or
			)
		elif "deliver_soup" in self.current_ml_action:
			self.last_motion_goal_source = "deliver_soup_actions"
			motion_goals = am.deliver_soup_actions()
		elif "wait" in self.current_ml_action:
			self.last_motion_goal_source = "wait_actions"
			motion_goals = am.wait_actions(player)
		else:
			raise ValueError("Invalid action: {}".format(self.current_ml_action))

		motion_goals = [
			mg
			for mg in motion_goals
			if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
				player.pos_and_or, mg
			)
		]

		return motion_goals

	def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
		"""
		For each motion goal, consider the optimal motion plan that reaches the desired location.
		Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
		or rationally), and returns the plan and the corresponding first action on that plan.
		"""

		if self.controller_mode == 'new':
			(
				chosen_goal,
				chosen_goal_action,
			) = self.get_lowest_cost_action_and_goal_new(
				start_pos_and_or, motion_goals, state
			)
		else: 
			(
				chosen_goal,
				chosen_goal_action,
			) = self.get_lowest_cost_action_and_goal(
				start_pos_and_or, motion_goals
			)
		return chosen_goal, chosen_goal_action
	
	def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""
		min_cost = np.Inf
		best_action, best_goal = None, None
		log_lines = [
			f"[t={self.current_timestep}] action={self.current_ml_action} source={self.last_motion_goal_source}",
			f"state={self.state_prompt}",
			f"start={start_pos_and_or}",
			f"mode=static candidate_count={len(motion_goals)}",
		]
		for goal in motion_goals:
			action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
				start_pos_and_or, goal
			)
			first_action = action_plan[0] if action_plan else None
			log_lines.append(
				f"candidate goal={self._format_goal(goal)} cost={plan_cost} first_action={self._format_action_for_log(first_action)}"
			)
			if plan_cost < min_cost:
				best_action = action_plan[0]
				min_cost = plan_cost
				best_goal = goal
		log_lines.append(
			f"chosen goal={self._format_goal(best_goal)} cost={min_cost} action={self._format_action_for_log(best_action)}"
		)
		self._append_motion_goal_log(log_lines)
		return best_goal, best_action

	def get_lowest_cost_action_and_goal_new(self, start_pos_and_or, motion_goals, state): 
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""   
		min_cost = np.Inf
		best_action, best_goal = None, None
		log_lines = [
			f"[t={self.current_timestep}] action={self.current_ml_action} source={self.last_motion_goal_source}",
			f"state={self.state_prompt}",
			f"start={start_pos_and_or}",
			f"other={state.players_pos_and_or[1 - self.agent_index]}",
			f"mode=real_time candidate_count={len(motion_goals)}",
		]
		for goal in motion_goals:   
			action_plan, plan_cost = self.real_time_planner(
				start_pos_and_or, goal, state
			)     
			log_lines.append(
				f"candidate goal={self._format_goal(goal)} cost={plan_cost} first_action={self._format_action_for_log(action_plan)}"
			)
			if plan_cost < min_cost:
				best_action = action_plan
				min_cost = plan_cost
				best_goal = goal     
		if best_action is None: 
			if np.random.rand() < 0.5:  
				log_lines.append("chosen goal=None cost=inf action=STAY fallback=random")
				self._append_motion_goal_log(log_lines)
				return None, Action.STAY
			else: 
				log_lines.append("fallback_to_static_planner=True")
				self._append_motion_goal_log(log_lines)
				return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
		log_lines.append(
			f"chosen goal={self._format_goal(best_goal)} cost={min_cost} action={self._format_action_for_log(best_action)}"
		)
		self._append_motion_goal_log(log_lines)
		return best_goal, best_action

	def real_time_planner(self, start_pos_and_or, goal, state):   
		terrain_matrix = {
			'matrix': copy.deepcopy(self.mlam.mdp.terrain_mtx), 
			'height': len(self.mlam.mdp.terrain_mtx), 
			'width' : len(self.mlam.mdp.terrain_mtx[0]) 
		}
		other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
		action_plan, plan_cost = find_path(start_pos_and_or, other_pos_and_or, goal, terrain_matrix) 

		return action_plan, plan_cost
	
