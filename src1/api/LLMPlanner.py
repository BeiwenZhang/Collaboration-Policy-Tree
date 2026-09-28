#from openai import OpenAI
import openai
import os
import re
import json
from LLM.call_llm_api.call_llm import TextChatbot
from project_paths import POLICY_DIR, PROMPTS_DIR

PLANNER_PROMPT_DIR = PROMPTS_DIR / "planner"
#Final experimental version (AB version); supports both merge and non-merge modes
class LLMPlanner:

    def __init__(self, layout, agent_index):
        
        self.planner_bot = TextChatbot("planner")
  
        prompt_file = PLANNER_PROMPT_DIR / f'{layout}_{agent_index}.txt'
        with open(prompt_file, "r") as f:
            self.system_content = f.read() # [{"role": "system", "content": f.read()}]
        
        self.task_content = '''
You should provide me with one decision-tree function description in the required format.
Make sure the conditions are not conflict with each other.
'''

        self.tactics_history = []

    def _extract_promotion_payload(self, message):
        """Support both fenced JSON and raw JSON strings from the summarizer."""
        if message is None:
            return None, message

        if isinstance(message, dict):
            return message, json.dumps(message, ensure_ascii=False, indent=2)

        if not isinstance(message, str):
            return None, message

        text = message.strip()
        candidates = []

        match = re.search(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(1))
        candidates.append(text)

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                return data, json.dumps(data, ensure_ascii=False, indent=2)
            except Exception:
                continue

        return None, message


    # def plan(self, tactics=None, message=None):

    #     if tactics is not None and message is not None:
    #         result_message = f'The current decision tree is:\n{tactics}'
    #         result_message += f'\n\nBut it has the following issue:\n{message}'
    #         result_message += '\n\nYou should provide me with one improved decision-tree function description in the required format. '
    #         result_message += 'Make sure the conditions are not conflict with each other.'

    #     response = self.planner_bot.query(self.system_content, self.task_content if message==None else result_message, maintain_history=True)

    #     return response

    #
    def plan(self, tactics=None, message=None, planner_merge=True,use_last=True,stuck_steps=0): 
        base_prompt=""
        # --- Process only when message has content ---
        #Storage happens after a message is available, so persistence is deferred
        if message is not None and use_last:
            parsed_message, normalized_message = self._extract_promotion_payload(message)
            if parsed_message is not None:
                try:
                    data = dict(parsed_message)
                    decision_summary = data.pop("Decision_Tree_Summary", None)

                    final_score = ""
                    if decision_summary is not None:
                        summary_text = decision_summary.get("Summary", "")
                        final_score = decision_summary.get("Final_Score", "")
                        self.tactics_history.append(
                            f"Summary: {summary_text}\nFinal_Score: {final_score}"
                        )

                    message = f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"

                    if final_score:
                        tactics = f"{tactics}\n\n(Previous Final_Score: {final_score})"
                except Exception as e:
                    print(f"[Warning] Failed to parse promotion JSON: {e}")
                    message = normalized_message
            else:
                print("[Warning] Promotion is not valid JSON, keeping raw message.")
        if tactics is not None and message is not None and use_last:
            # --- Concatenate the historical tactic pool ---
            history_text = "\n\n".join(self.tactics_history) if self.tactics_history else "No previous tactics."
            print("history:", history_text)

            # --- Build the shared prompt header ---
            base_prompt = f"""<DECISION_TREE_MEMORY>
You are given some [previous decision tree memorys].
ALWAYS prefer decision trees with higher Final_Score, and reuse these to generate new one.
Use the following memory for reference:
{history_text}

<THE_LAST_DECISION_TREE>
{tactics}

<LAST_ISSUE_INFORMATION>
Analyze and modify the last decision tree using these improvement requirements:
{message}
            """


        # --- If merge mode is enabled ---
        if planner_merge:
            result_message=""
            if base_prompt:
                result_message+=base_prompt+"\n"
            result_message +="""
<OUTPUT_REQUIREMENTS>    
- The output must start directly with "### FunctionDescription:" and end after the last line of the **Self Skill Selection Tree**. Please ensure that the generated decision tree covers the vast majority of branches in the game.
- Decide whether coordination is required. If coordination is required, condition self_action_now on the inferred_teammate_current_skill in **Partner Inference Tree**. If not, ignore inferred_teammate_current_skill.
- Now you should provide me with one **modified** decision-tree function description strictly following the **Required output format** and **<LAST_ISSUE_INFORMATION>**. If you don't find them, just ignore this point.  
- Do NOT include any explanations, introductions, or commentary.
- Make sure the conditions are not conflict with each other.
- If any_soup_ready, pay attention to it.
    """#- If self_action_now=self_last_skill, consider whether a collision has occurred and attempt to execute action "None".
#   - When generating the **Self Skill Selection Tree**, explicitly include conditional reasoning of the form: “If inferred_teammate_current_skill = [Skill_X], then self_action_now = [Skill_Y]”, so that the tree reflects how your action should adapt to the predicted teammate’s current skill.
# #- Required output format:
### FunctionDescription:
# Name: PredictTeammateThenPlan
# Inputs:
#     - current_scene:
#         - holdings: {self: empty/onion/dish/soup, teammate: empty/onion/dish/soup}
#         - pots: for each <Pot>: onion_count: {0,1,2,3}, state: {idle,cooking,ready}, timers
#         - derived flags: any_soup_ready, any_pot_not_full
#     - teammate_last_completed_skill: one of the 6 skills or None
#     - teammate_last_inferred_skill: one of the 6 skills or None
#     - self_last_skill: one of the 6 skills or None
# Outputs:
#     - inferred_teammate_current_skill: one of the 6 skills
#     - self_action_now: one of the 6 skills

# **Partner Inference Tree**

# **Self Skill Selection Tree**     
            print("🟢 [Merged Planner Mode] Sending one combined query...")
            full_prompt = f"{self.system_content}\n\n{result_message}"
            # with open(POLICY_DIR / "full_prompt.txt", "w", encoding="utf-8") as f:
            #     f.write(full_prompt)
            # print(full_prompt)
            response = self.planner_bot.query(
                self.system_content,
                result_message,
                maintain_history=False
            )
            return response

        # --- Otherwise use two steps; note that the prompt for this case has not been updated yet
        else:
            print("🟡 [Two-Step Planner Mode] Step 1: generating FunctionDescription + Partner Inference Tree...")

            # Step 1 prompt
            prompt1 = (
                base_prompt
                + """
- Now please generate the first part, ### FunctionDescription and **Partner Inference Tree**. The output must start directly with "### FunctionDescription:" and end after the last line of the **Partner Inference Tree**. You do not need to generate **Self Skill Selection Tree** this step.
- Now you should provide me with one **modified** decision-tree function description strictly following the **Required output format** and **<LAST_ISSUE_INFORMATION>**. If you don't find them, just ignore this point. 
- Do NOT include any explanations, introductions, or commentary.

    """
            )

            response1 = self.planner_bot.query(
                self.system_content,
                prompt1,
                maintain_history=False,  # Do not retain conversation memory
                stuck_steps=stuck_steps
            )
            # print("response1",response1)

            # Step 2 prompt
            print("🟡 [Two-Step Planner Mode] Step 2: generating Self Skill Selection Tree...")
            prompt2 = (
                base_prompt
                + "\n\nHere is what you have generated in the previous step:\n"
                + f"{response1}\n\n"
                #+"Decide whether coordination is required. If coordination is required, condition self_action_now on the inferred_teammate_current_skill in **Partner Inference Tree**.\n"
                + "### Please now continue and generate only the following part:\n"
                + "**Self Skill Selection Tree**\n"
                + 'The output must start directly with "**Self Skill Selection Tree**" and end after the last line of the Self Skill Selection Tree. Please ensure that the generated decision tree covers the vast majority of branches in the game.\n'
                +'Decide whether coordination is required. If coordination is required, condition self_action_now on the inferred_teammate_current_skill in **Partner Inference Tree**. If not, ignore inferred_teammate_current_skill.\n'
                +'Make sure the conditions are not conflict with each other. If any_soup_ready, pay attention to it.\n'
                +'PAY ATTENTION TO **<LAST_ISSUE_INFORMATION>**, If you dont find it, just ignore this point.\n'
            )

            response2 = self.planner_bot.query(
                self.system_content,
                prompt2,
                maintain_history=False,
                stuck_steps=stuck_steps
            )
            print("Step 2 done.")

            # --- Concatenate the final output ---
            final_output = f"{response1.strip()}\n\n{response2.strip()}"
            # print("test:",final_output)
            return final_output
