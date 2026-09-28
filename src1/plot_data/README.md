# Plot Data

This directory stores data files consumed by plotting scripts.

- `cost_comparison.json`: query-count and latency data for `picture.py`.
- `reward_cost_comparison.json`: reward, token, query-count, and latency data for `picture_tokens.py`.
- `human_rewards_long.csv`: optional long-format human-study reward table used by `box_single.py` and `box_both.py`.

The expected CSV columns for `human_rewards_long.csv` are:

```text
layout,role_assignment,method,reward
```
