import os
#from openai import OpenAI
import openai
#import config
from LLM.call_llm_api.call_llm import TextChatbot
import json
from project_paths import PROMPTS_DIR

SUMMARIZER_PROMPT_DIR = PROMPTS_DIR / "explainer"

#Version changed from two steps to one step
class LLMSummarizer:

    def __init__(self, layout, agent_index):

        self.summarizer_bot = TextChatbot("summarizer")

        prompt_file = SUMMARIZER_PROMPT_DIR / f'{layout}_{agent_index}.txt'
        with open(prompt_file, "r") as f:
            self.system_content = f.read()

    def summarize(self, tactics, data,stuck_steps=0):
        """
        Inputs:
            tactics: text description of the current decision tree
            data: step logs for one full episode and the final result
        Outputs:
            tree_reflexion (str): JSON string for the planner
            Contains reflection advice for only one small modifiable branch
        """

        # ============================================================
        # Step 0: Build Episode States (behavior trajectory)
        # ============================================================
        episode_states = ""
        for t, feedback in data["step"].items():
            episode_states += f"Scene {t}:\n"
            episode_states += f"{feedback['state']}\n"

            # if feedback.get("state_demo", ""):
            #     episode_states += f"[Grid]\n{feedback['state_demo']}\n"

            if feedback.get("predict", ""):
                episode_states += f"[Misprediction] {feedback['predict']}\n"

            if feedback.get("plan", ""):
                episode_states += f"[DecisionTrace] {feedback['plan']}\n"

            episode_states += "###\n"

        result_info = data.get("result", {})
        exact_final_score = result_info.get("avg", result_info)

        # ============================================================
        # Step 1: Complete behavior diagnosis + Tree Reflexion in one step
        # ============================================================
        merged_prompt = f"""
    You are analyzing the behavior of a collaborative agent controlled by a FIXED decision tree, and then generating a Tree_Reflexion for a Planner.

    You MUST reason over the full episode trajectory across multiple scenes, instead of treating each scene independently.

    IMPORTANT CONSTRAINTS:
    - In the next iteration, ONLY ONE SMALL BRANCH of the decision tree can be modified.
    - All other branches, priorities, and logic MUST remain unchanged.
    - Do NOT propose a full rewrite.
    - Do NOT suggest multiple alternative fixes.

    Your goal is to identify the SINGLE MOST DAMAGING and REPEATABLE inefficiency that:
    1) Causes clear throughput loss across multiple scenes, not just one isolated mistake,
    2) Appears as repeated wasted timesteps, redundant role overlap, premature dish handling, delayed role switching, or legal-but-low-value actions,
    3) Can still be attributed to ONE specific branch, priority rule, or missing condition in the decision tree.

    PRIORITIZATION RULE:
    - Prefer a bottleneck that repeatedly reduces soup throughput, even if all actions are technically legal.
    - Do NOT over-prioritize teammate prediction mistakes unless they clearly create repeated downstream time loss.
    - A valid-but-low-value action pattern is more important than a one-off invalid action.
    - Also consider TIMING-related inefficiencies:
      - Acting too early or too late
      - Situations where waiting would be better than acting immediately
      - Situations where the agent chooses a legal action but the wrong role for the current phase

    [Episode states]:
    {episode_states}

    [The decision tree]:
    {tactics}

    [Final score]:
    {result_info}

    Your task:
    1. Scan all episode states and identify time-wasting behaviors.
    2. Select exactly ONE primary inefficiency with the largest negative impact on efficiency or score.
    3. Identify the EXACT target branch or decision condition to modify.
    4. Generate a Tree_Reflexion that guides a Planner to perform a SINGLE, LOCAL modification:
      - describe the CURRENT behavior of this branch (BEFORE),
      - describe the DESIRED behavior after modification (AFTER), using refined conditions, priority changes, or a single added guard condition,
      - explicitly list assumptions about what logic MUST remain unchanged.
    5. The value in Decision_Tree_Summary.Final_Score MUST be exactly {exact_final_score}.

    OUTPUT REQUIREMENTS:
    - Output MUST be valid JSON.
    - Do NOT include markdown, explanations, or extra text.
    - Be concrete and implementable.
    - Output must directly contain only these top-level keys:
      - "Decision_Tree_Summary"
      - "Tree_Reflexion"

    Return JSON in the following format:

    {{
      "Decision_Tree_Summary": {{
        "Summary": "Summarize the key strategies and critical decision points here.",
        "Final_Score": "{exact_final_score}"
      }},
      "Tree_Reflexion": {{
        "Target_Branch": "",
        "Modification_Scope": "LOCAL_SINGLE_BRANCH",
        "Before_Behavior": "",
        "After_Behavior": [
          ""
        ],
        "Unchanged_Assumptions": [
          ""
        ],
        "Expected_Effect": ""
      }}
    }}
    """

        def extract_json(text):
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No valid JSON object found.")
            return json.loads(text[start:end + 1])

        # -------- Retry mechanism --------
        retries = 0
        max_retries=3
        while retries < max_retries:
            print("Retrying...")
            try:
                tree_response = self.summarizer_bot.query(
                    self.system_content,
                    merged_prompt,
                    maintain_history=True,
                    stuck_steps=stuck_steps
                )

                tree_json = extract_json(tree_response)
                break
            except Exception as e:
                print(f"[Summarizer JSON ERROR] Attempt {retries + 1}/{max_retries}")
                print(f"Error: {e}")
                retries += 1
                if retries >= max_retries:
                    print("[ERROR] Maximum retries reached, failed to extract valid JSON.")
                    print(tree_response)  # Print the returned response for further debugging
                    raise e  # Raise the error if the retry limit is exceeded

        try:
            if "Decision_Tree_Summary" not in tree_json:
                tree_json["Decision_Tree_Summary"] = {}
            tree_json["Decision_Tree_Summary"]["Final_Score"] = str(exact_final_score)
            return json.dumps(tree_json, ensure_ascii=False, indent=2)
        except Exception:
            return tree_response
