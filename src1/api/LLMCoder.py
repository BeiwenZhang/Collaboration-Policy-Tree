#from openai import OpenAI
import openai
from LLM.call_llm_api.call_llm import TextChatbot
from LLM.call_llm_api.call_llm import main_logger
import os
import re
import subprocess
from multiprocessing import Process, Queue
from project_paths import POLICY_DIR, PROMPTS_DIR


CODER_PROMPT_DIR = PROMPTS_DIR / "coder"
POLICY_DIR.mkdir(parents=True, exist_ok=True)


class LLMCoder:

    def __init__(self, layout, agent_index):

        self.coder_bot = TextChatbot("coder")

        prompt_file = CODER_PROMPT_DIR / f'{layout}_{agent_index}.txt'
        with open(prompt_file, "r") as f:
            self.system_content = f.read() 
        
        self.task_content = '''
Make sure to check whether the list variables are empty or not.
        '''

        self.policy_file = str(POLICY_DIR / f'{layout}_{agent_index}.txt')


    def generate_code(self, tactic, code='', promotion='',stuck_steps=0):
        prompt = self.task_content + '\n The tactic is: ' + str(tactic)
        
        # If code and error information are provided, add them to the prompt
        if code != '' and promotion != '':
            prompt += 'The current Python code implementation is:' + str(code)
            prompt += 'When executing this code, the following error occurred:' + str(promotion)
            prompt += 'Please carefully analyze the error message and modify the code to fix the issue.'
        else:
            prompt += '\nPlease implement the code in Python format. Please return the code wrapped with triple backticks (```python ... ```), and make sure the code is complete and syntactically correct.'

        # Try up to 5 times to obtain valid code
        for attempt in range(5):
            # Query coder_bot to generate a response
            stuck_steps_=min(10,stuck_steps+attempt)
            response = self.coder_bot.query(self.system_content, prompt, maintain_history=False, stuck_steps=stuck_steps_)
            
            # Extract code and check whether it is empty
            code = self.extract_save_code(response)
            
            # Return once valid code is successfully extracted
            if code:
                return code
            else:
                print(f"Attempt {attempt + 1}: No valid Python code found, querying again...")
        
        # Print a warning if all 5 attempts fail
        print("Warning: Failed to retrieve valid Python code after 5 attempts.")
        return ""  # Or return a default value if needed



    def extract_save_code(self, response):
        m = re.search(r"```python\s*([\s\S]*?)\s*```", response, flags=re.IGNORECASE)
        
        # Handle the case where no match is found
        if m is None:
            # Could return a special marker or raise an exception here
            print("No Python code found in the response.")
            return ""  # Return an empty string or a special marker
        
        code = m.group(1).strip()  # Extract code if a match is found
        with open(self.policy_file, "w") as f:
            f.write(code)
        return code





