import copy
import json
import os
import threading
from datetime import datetime


_LOCK = threading.Lock()


def _empty_usage():
    return {
        "query_count": 0,
        "successful_query_count": 0,
        "failed_query_count": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "missing_usage_count": 0,
        "by_task": {},
        "by_model": {},
    }


_LLM_USAGE = _empty_usage()


def reset_llm_metrics():
    with _LOCK:
        _LLM_USAGE.clear()
        _LLM_USAGE.update(_empty_usage())


def _usage_value(usage, key, default=0):
    if usage is None:
        return default
    if isinstance(usage, dict):
        return usage.get(key, default)
    return getattr(usage, key, default)


def _nested_usage_value(usage, outer_key, inner_key, default=0):
    details = _usage_value(usage, outer_key, None)
    return _usage_value(details, inner_key, default)


def _ensure_bucket(container, name):
    if name not in container:
        container[name] = {
            "query_count": 0,
            "successful_query_count": 0,
            "failed_query_count": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "missing_usage_count": 0,
        }
    return container[name]


def _add_usage(target, prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens, has_usage, success):
    target["query_count"] += 1
    if success:
        target["successful_query_count"] += 1
    else:
        target["failed_query_count"] += 1

    target["prompt_tokens"] += prompt_tokens
    target["completion_tokens"] += completion_tokens
    target["total_tokens"] += total_tokens
    target["cached_tokens"] += cached_tokens
    target["reasoning_tokens"] += reasoning_tokens
    if not has_usage:
        target["missing_usage_count"] += 1


def record_llm_query(task, model, usage=None, success=True):
    has_usage = usage is not None
    prompt_tokens = int(_usage_value(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_usage_value(usage, "completion_tokens", 0) or 0)
    total_tokens = int(_usage_value(usage, "total_tokens", 0) or 0)
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    cached_tokens = int(_nested_usage_value(usage, "prompt_tokens_details", "cached_tokens", 0) or 0)
    reasoning_tokens = int(_nested_usage_value(usage, "completion_tokens_details", "reasoning_tokens", 0) or 0)

    task_name = str(task or "unknown")
    model_name = str(model or "unknown")

    with _LOCK:
        _add_usage(
            _LLM_USAGE,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens,
            reasoning_tokens,
            has_usage,
            success,
        )
        _add_usage(
            _ensure_bucket(_LLM_USAGE["by_task"], task_name),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens,
            reasoning_tokens,
            has_usage,
            success,
        )
        _add_usage(
            _ensure_bucket(_LLM_USAGE["by_model"], model_name),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached_tokens,
            reasoning_tokens,
            has_usage,
            success,
        )


def get_llm_metrics():
    with _LOCK:
        return copy.deepcopy(_LLM_USAGE)


def save_llm_metrics(path, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "llm_usage": get_llm_metrics(),
    }
    if metadata is not None:
        payload["metadata"] = metadata

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload
