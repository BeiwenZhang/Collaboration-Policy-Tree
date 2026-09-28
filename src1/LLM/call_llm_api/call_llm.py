# call_llm.py
import logging
import pdb
from logging.handlers import RotatingFileHandler
#from openai import OpenAI
import openai
import math
from configs.llm_api_config import LLMAPIConfig
from metrics import record_llm_query

def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


model_info = LLMAPIConfig.get_model_dict()
main_logger = setup_logger('main_logger', '{}#{}#{}.log'.format(model_info['planner'], model_info['coder'], model_info['summarizer']), level=logging.DEBUG)

def _response_usage(response):
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)

class BaseChatbot:
    def __init__(self, task):
        model_config = LLMAPIConfig.get_task_model(task)
        if model_config is None:
            raise ValueError(f"No model configured for task: {task}")
        if not model_config.api_key:
            model_name = LLMAPIConfig.TASK_MODELS[task]
            raise ValueError(
                f"No API key configured for {task} ({model_name}). "
                "Set the matching CO_PI_TREE_<MODEL>_API_KEY environment variable."
            )

        #self.client = OpenAI(
        #    api_key=model_config.api_key,
        #    base_url=model_config.base_url
        #)
        openai.api_key = model_config.api_key
        if model_config.base_url:
            openai.api_base = model_config.base_url
        self.client = openai

        main_logger.debug(f"{task.capitalize()} API client created")

        self.model = LLMAPIConfig.TASK_MODELS[task]
        # if self.model == 'gpt-3.5-turbo-1106':
        #     self.model = '/v1/chat/completions/gpt-3.5-turbo-1106' #"Pro/deepseek-ai/DeepSeek-R1"
        main_logger.debug(f"Model set to: {self.model}")
        self.conversation_history = []
        self.task = task

        self.log_system=False

    def query(self, system_prompt, user_input):
        raise NotImplementedError("Subclasses must implement this method")

    def clear_history(self):
        self.conversation_history = []

class TextChatbot(BaseChatbot):

    def query(self, system_prompt, user_input, maintain_history=True, stuck_steps=0):
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "user", "content": user_input})

            # ===== stuck-aware temperature =====
            T_base = 0.4
            T_max = 1.2

            # Logarithmic growth: mild early on, more exploratory later
            temperature = min(
                T_max,
                T_base * (1 + math.log(1 + stuck_steps))
            )
            print(f"Temperature used: {temperature:.3f}")
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
                top_p=0.9,
                stream=False
            )
            record_llm_query(self.task, self.model, _response_usage(response), success=True)

            assistant_response = response.choices[0].message.content

            if maintain_history:
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_response})

            main_logger.debug(f"Querying model: {self.model}")
            main_logger.debug(f"Temperature used: {temperature:.3f} (stuck_steps={stuck_steps})")

            if self.log_system is False:
                main_logger.debug(system_prompt)
                self.log_system = True

            main_logger.debug(user_input)
            main_logger.debug("Query successful")
            main_logger.debug(assistant_response)

            return assistant_response

        except Exception as e:
            record_llm_query(getattr(self, "task", "unknown"), getattr(self, "model", "unknown"), None, success=False)
            main_logger.error(f"An error occurred: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"
