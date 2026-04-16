# Frozen archive — Adaptive Suppression of Trigger-Word Persistence in Streaming LLM Dialogue

This folder is the **single public archive repository** for the ESWA submission.
It bundles the frozen code, prompts, configs, tests, and quantitative result artifacts
so every submission component can point to one authoritative URI strategy.

## Frozen release target

- Preferred repository root: `https://github.com/li13683316665-star/Adaptive-Suppression-of-Trigger-Word-Persistence-in-Streaming-LLM-Dialogue_Data`
- Fixed release/tag name: `eswa-20260415-freeze`
- Primary benchmark run group: `minicpm_v45_incr_full_eswa_20260415_051850_suite`
- Optional DOI path: connect the repository to Zenodo after publishing the GitHub release.

## Main-paper benchmark snapshot

- `qwen3.5:4b`
- `gemma4:e4b`
- `openbmb/minicpm-v4.5:8b`
- `ministral-3:8b`

The archive includes the full `paper_*` export set and the matching cross-model manifest
for the single frozen benchmark snapshot above. Earlier `20260414` exports are not part of
the public submission-freeze narrative and should not be cited as the main-paper snapshot.

## Held-out materials

- Consolidated held-out manifest: `held_out_manifest_eswa_20260415_freeze.json`
- Legacy per-repeat held-out JSON files remain included only through the consolidated manifest.
- Expanded held-out prompt suites live under `data/prompts/` in this same archive.

## Repository layout

```
results/        # frozen paper exports + manifests + held-out freeze manifest
experiments/    # experiment runners and artifact builder
data/prompts/   # development and held-out prompt suites
src/            # detector, controller, metrics, loaders
configs/        # YAML configuration, including paper figure-model pin
tests/          # regression checks
scripts/        # archive and analysis helpers
requirements.txt
CITATION.cff
ARCHIVE_RELEASE.md
MANIFEST.sha256
```

Publication PNG figures are included with filenames aligned as `Figure_X`.
They are not the canonical evidence layer; regenerate them from the frozen `results/paper_*.json`
exports with `experiments/04_build_paper_artifacts.py` when needed.

## Citation

Use the repository root above together with the fixed release tag for the submission freeze.
If a Zenodo DOI is minted later, treat that DOI as the preferred citable archive identifier.

## Integrity

SHA-256 checksums in `MANIFEST.sha256` cover only public archive contents.
They intentionally exclude `.git/`, merge leftovers, and other repository-internal files.

## Publishing the release

See `ARCHIVE_RELEASE.md` for the exact local-to-GitHub release steps and the optional DOI workflow.

## Full `data_new/` archive (optional)

To bundle the entire local `data_new/` tree for deeper byte-level inspection, run from the project root:

`python scripts/build_eswa_data_deposit.py --full-data-new`
