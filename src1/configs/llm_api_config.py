import os


class LLMModelConfig:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def __repr__(self):
        if self.api_key:
            masked_api_key = self.api_key[:6] + "*" * max(len(self.api_key) - 6, 0)
        else:
            masked_api_key = "<unset>"
        return f"LLMModelConfig(api_key='{masked_api_key}', base_url='{self.base_url}')"


def env_config(prefix, default_base_url=""):
    return LLMModelConfig(
        api_key=os.environ.get(f"{prefix}_API_KEY", ""),
        base_url=os.environ.get(f"{prefix}_BASE_URL", default_base_url),
    )


class LLMAPIConfig:

    MODELS = {
        "deepseek-chat": env_config("CO_PI_TREE_DEEPSEEK_CHAT"),
        "deepseek-coder": env_config("CO_PI_TREE_DEEPSEEK_CODER"),
        "gpt-4": env_config("CO_PI_TREE_GPT4"),
        "Qwen2.5-72B-Instruct": env_config("CO_PI_TREE_QWEN25_72B_INSTRUCT"),
        "claude-3-5-sonnet-20240620": env_config("CO_PI_TREE_CLAUDE35_SONNET"),
        "deepseek-r1": env_config("CO_PI_TREE_DEEPSEEK_R1"),
        "gpt-3.5-turbo-1106": env_config("CO_PI_TREE_GPT35_TURBO_1106"),
        "gpt-4o": env_config("CO_PI_TREE_GPT4O"),
        "gpt-4.1": env_config("CO_PI_TREE_GPT41"),
    }

    TASK_MODELS = {
        "planner": "gpt-4o",
        "coder": "gpt-4o",
        "summarizer": "gpt-4o",
    }

    @classmethod
    def get_model_config(cls, model_name):
        return cls.MODELS.get(model_name)

    @classmethod
    def get_task_model(cls, task):
        model_name = cls.TASK_MODELS.get(task)
        return cls.get_model_config(model_name)

    @classmethod
    def get_model_dict(cls):
        return cls.TASK_MODELS


if __name__ == '__main__':
    config = LLMAPIConfig()
    print("Model config for deepseek_chat:")
    print(config.get_model_config("deepseek_chat"))
    print("\nModel config for planner task:")
    print(config.get_task_model("planner"))
