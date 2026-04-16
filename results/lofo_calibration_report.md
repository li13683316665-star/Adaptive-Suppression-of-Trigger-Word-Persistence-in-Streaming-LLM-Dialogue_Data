# Leave-one-family-out calibration audit

- Source: `data_new\results\calibration_sweep_20260411_092539.json`
- `run_group`: `aggregated_from_existing_20260411_092539`

For each model, when holding out family **F**, we merge the two other families' `recommended` tiers by taking the **more difficult** tier (higher d1<d2<d3 rank), then read the held-out family's `mean_pctp` at that tier.

| model | held_out | tier_from_other_two | mean_pctp_at_tier (held_out) |
|---|---|---|---|
| `minicpm-v` | `location` | `d1` (from `emotion`=d1, `color`=d1) | 0.1952 |
| `minicpm-v` | `emotion` | `d1` (from `location`=d1, `color`=d1) | 0.3937 |
| `minicpm-v` | `color` | `d1` (from `location`=d1, `emotion`=d1) | 0.3238 |
| `qwen2.5:7b` | `location` | `d1` (from `emotion`=d1, `color`=d1) | 0.1651 |
| `qwen2.5:7b` | `emotion` | `d1` (from `location`=d1, `color`=d1) | 0.0889 |
| `qwen2.5:7b` | `color` | `d1` (from `location`=d1, `emotion`=d1) | 0.2206 |
| `qwen3.5:4b` | `location` | `d1` (from `emotion`=d1, `color`=d1) | 0.1508 |
| `qwen3.5:4b` | `emotion` | `d1` (from `location`=d1, `color`=d1) | 0.1111 |
| `qwen3.5:4b` | `color` | `d1` (from `location`=d1, `emotion`=d1) | 0.3111 |
