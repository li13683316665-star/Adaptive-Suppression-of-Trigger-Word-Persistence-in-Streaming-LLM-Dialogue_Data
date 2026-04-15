# Code and prompts — Adaptive Suppression of Trigger-Word Persistence in Streaming LLM Dialogue

Companion **software** archive for the ESWA manuscript. The frozen **evaluation outputs** (merged JSON manifests and `paper_*.json` exports) live in the separate **research-data** GitHub repository cited in the article.

## What this folder contains

| Path | Role |
|------|------|
| `experiments/` | Experiment drivers and the paper artifact builder (`04_build_paper_artifacts.py`) |
| `data/prompts/` | Benchmark case definitions (JSON) referenced by `configs/default.yaml` (`paths.data_prompts`) |
| `src/` | Library code (detector, adaptive controller, stats, loaders) |
| `configs/` | YAML configuration (paths, `paper.figure_models`, generation defaults) |
| `tests/` | Pytest checks for detector, controller, metrics, stats |
| `requirements.txt` | Python dependencies (install PyTorch separately; see comments inside) |

## Environment

1. Install PyTorch per [pytorch.org](https://pytorch.org) for your CUDA/CPU setup.
2. From **this directory** (the parent of `experiments/`):

   ```text
   pip install -r requirements.txt
   ```

3. Run Python with the project root on `PYTHONPATH` (this folder is the root):

   **PowerShell**

   ```text
   $env:PYTHONPATH = (Get-Location).Path
   pytest tests -v
   ```

   **bash**

   ```text
   export PYTHONPATH="$(pwd)"
   pytest tests -v
   ```

## Regenerating figures / tables

1. Copy the frozen JSON from the **research-data** GitHub repository into `data_new/results/` (see `data_new/results/README.txt`).
2. From **this directory** (repository root):

   ```text
   python experiments/04_build_paper_artifacts.py
   ```

   PNGs are written under `Docs/Paper/figures/` (created automatically).

## License

Apache License 2.0 — see `LICENSE`.
