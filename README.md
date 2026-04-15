# Research data — Adaptive Suppression of Trigger-Word Persistence in Streaming LLM Dialogue

This deposit supports the manuscript submitted to *Expert Systems with Applications* (ESWA),
in line with Elsevier research-data policy (Option C): public repository, citation, and link in the article.

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

Publication PNG figures are not bundled here (Elsevier uploads them as separate artwork files).
They can be regenerated from these `results/paper_*.json` exports using `experiments/04_build_paper_artifacts.py`
in the main project when the results directory contains this snapshot.

## Citation

See the article’s References entry for the dataset (GitHub repository) and the permanent URL given in the **Research data** statement.

## Integrity

SHA-256 checksums for included files are listed in `MANIFEST.sha256`.

## Push to GitHub (research-data repository)

This folder is intended to be the **root** of the public data repo (or copied into it). Using SSH (e.g. deploy key with read/write):

```text
cd eswa_data_deposit
git init
git remote add origin git@github.com:li13683316665-star/Adaptive-Suppression-of-Trigger-Word-Persistence-in-Streaming-LLM-Dialogue_Data.git
git add .
git commit -m "ESWA research data snapshot (ministral suite 20260414)"
git branch -M main
git push -u origin main
```

If GitHub already created a `README`/`LICENSE` commit, use `git pull origin main --allow-unrelated-histories` once, resolve conflicts, then `git push`.

## Full `data_new/` archive (optional)

To bundle the entire local `data_new/` tree (~500 MB, thousands of files) for byte-level replication, run from the project root:

`python scripts/build_eswa_data_deposit.py --full-data-new`


## Reproducibility code (same repository)

The `code/` directory contains experiment scripts, benchmark prompts under `code/data/prompts/`, library code under `code/src/`, `code/configs/`, `code/tests/`, and `code/requirements.txt`. See `code/README.md` for environment setup.

To regenerate paper figures after cloning this repository: copy the JSON files from `results/` (at repo root) into `code/data_new/results/`, then from the `code/` folder set `PYTHONPATH` to that folder and run `python experiments/04_build_paper_artifacts.py` (PNG output under `code/Docs/Paper/figures/`).
