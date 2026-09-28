import os

import openai


def main():
    api_key = os.environ.get("CO_PI_TREE_OPENAI_API_KEY", "")
    base_url = os.environ.get("CO_PI_TREE_OPENAI_BASE_URL", "")
    if not api_key:
        raise RuntimeError("Set CO_PI_TREE_OPENAI_API_KEY before running this API smoke test.")

    openai.api_key = api_key
    if base_url:
        openai.api_base = base_url

    response = openai.ChatCompletion.create(
        model=os.environ.get("CO_PI_TREE_OPENAI_MODEL", "gpt-4"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False,
    )
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
