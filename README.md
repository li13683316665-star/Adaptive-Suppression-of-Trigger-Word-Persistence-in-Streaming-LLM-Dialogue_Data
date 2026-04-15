# Research data — Adaptive Suppression of Trigger-Word Persistence in Streaming LLM Dialogue


## Primary snapshot (manuscript tables and figures)

Run group: `ministral_incr_full_eswa_20260414_055054_suite`.

Included JSON exports (`results/paper_*.json`) are produced by `experiments/04_build_paper_artifacts.py`
from merged manifests under `data_new/results/`. Filenames match those referenced in the manuscript
(e.g. `paper_crossmodel_baseline_*_suite.json` and held-out manifest names in Table 2).

Held-out manifests cited in the paper:

- `held_out_manifest_heldout_eswa_20260414.json`
- `held_out_manifest_ministral_incr_full_eswa_20260414_055054_heldout.json`

## Layout

```
results/     # merged paper exports + crossmodel manifest + held-out manifests
configs/     # default.yaml (paper.figure_models pin for main panels)
LICENSE        # Apache-2.0
MANIFEST.sha256
```

Publication PNG figures are here, with their filenames aligned as "Figure_X"
They can be regenerated from these `results/paper_*.json` exports using `experiments/04_build_paper_artifacts.py`
in the main project when the results directory contains this snapshot.

## Citation

See the article’s References entry for the dataset (GitHub repository) and the permanent URL given in the **Research data** statement.

## Integrity

SHA-256 checksums for included files are listed in `MANIFEST.sha256`.

## Reproducibility code (same repository)

The `code/` directory contains experiment scripts, benchmark prompts under `code/data/prompts/`, library code under `code/src/`, `code/configs/`, `code/tests/`, and `code/requirements.txt`. See `code/README.md` for environment setup.

To regenerate paper figures after cloning this repository: copy the JSON files from `results/` (at repo root) into `code/data_new/results/`, then from the `code/` folder set `PYTHONPATH` to that folder and run `python experiments/04_build_paper_artifacts.py` (PNG output under `code/Docs/Paper/figures/`).

## Software/hardware notes

THe author used a Asus ROG Strix Scar 18 (2023) Laptop with RTX4090 laptop (687 AI tops)
Python==3.12.10
ollama==0.20.6
(See all in \code\requirements.txt)
Used model: "gemma4:e4b; qwen3.5:4b openbmb/minicpm-v4.5:8b ministral-3:8b"
