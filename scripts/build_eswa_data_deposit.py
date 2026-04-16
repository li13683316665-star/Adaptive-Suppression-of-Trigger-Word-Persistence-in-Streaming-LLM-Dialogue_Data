"""
Build the frozen single-repository public archive for the ESWA submission.

This archive intentionally packages code, prompts, configs, tests, and the
frozen result artifacts in ONE repository so the manuscript, cover letter, and
research-data statement can all point to the same public URI strategy.

Primary publication-figure export snapshot:
  ``deepseek_v32_cross_20260415``

The archive also writes a consolidated held-out manifest for the submission
freeze and excludes repository-internal files from ``MANIFEST.sha256``.

Usage (from repository root):
  python scripts/build_eswa_data_deposit.py
  python scripts/build_eswa_data_deposit.py --full-data-new   # optional large local mirror
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FROZEN_RELEASE_TAG = "eswa-20260415-freeze"
ARCHIVE_REPO_URL = (
    "https://github.com/li13683316665-star/"
    "Adaptive-Suppression-of-Trigger-Word-Persistence-in-Streaming-LLM-Dialogue_Data"
)
PRIMARY_RUN_GROUP = "deepseek_v32_cross_20260415"
CONSOLIDATED_HELDOUT_NAME = "held_out_manifest_eswa_20260415_freeze.json"
RESULTS = ROOT / "data_new" / "results"
PUBLIC_FIGURES_DIR = ROOT / "Docs" / "Paper" / "figures"
PUBLIC_CODE_DIRS = (
    "experiments",
    "src",
    "configs",
    "tests",
    "scripts",
)
PUBLIC_FILES = ("requirements.txt",)
PUBLIC_PROMPT_GLOB = "data/prompts/*.json"
MANIFEST_SKIP_NAMES = {
    "MANIFEST.sha256",
    ".gitignore",
}
MANIFEST_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cursor",
    ".vscode",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_if_exists(src: Path, dest_dir: Path, missing: list[str]) -> None:
    if not src.is_file():
        missing.append(str(src.relative_to(ROOT)))
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)


def _copytree_public(src: Path, dest: Path) -> None:
    def _ignore(_dir: str, names: list[str]) -> list[str]:
        skip: list[str] = []
        for name in names:
            if name in MANIFEST_SKIP_DIRS or name.startswith(".git"):
                skip.append(name)
            elif name.endswith(".pyc"):
                skip.append(name)
        return skip

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=_ignore)


def _load_heldout_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = str(payload.get("model", "unknown"))
    family = str(payload.get("results", [{}])[0].get("trigger_word", "unknown"))
    prompt_file = str(payload.get("prompt_file", ""))
    repeat_index = int(payload.get("repeat_index", 0) or 0)
    return {
        "kind": "held_out_baseline",
        "model": model,
        "family": family,
        "repeat_index": repeat_index,
        "prompt_file": prompt_file,
        "files": [path.name],
        "source_run_group": payload.get("run_group", ""),
    }


def _collect_heldout_files() -> list[Path]:
    return sorted(RESULTS.glob("heldout_baseline_*.json"))


def _write_consolidated_heldout_manifest(dest_res: Path) -> str | None:
    heldout_files = _collect_heldout_files()
    if not heldout_files:
        return None
    artifacts = [_load_heldout_payload(path) for path in heldout_files]
    manifest = {
        "run_group": "eswa_20260415_freeze_heldout",
        "created_for_release_tag": FROZEN_RELEASE_TAG,
        "primary_benchmark_run_group": PRIMARY_RUN_GROUP,
        "families": sorted({str(item["family"]) for item in artifacts}),
        "models": sorted({str(item["model"]) for item in artifacts}),
        "artifacts": artifacts,
    }
    out = dest_res / CONSOLIDATED_HELDOUT_NAME
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out.name


def _copy_public_code_tree(out: Path, copied: list[str], missing: list[str]) -> None:
    for rel in PUBLIC_CODE_DIRS:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dest = out / rel
        _copytree_public(src, dest)
        copied.append(rel)

    prompt_dest = out / "data" / "prompts"
    prompt_dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(ROOT.glob(PUBLIC_PROMPT_GLOB)):
        shutil.copy2(path, prompt_dest / path.name)
        copied.append(str(path.relative_to(ROOT)))

    for rel in PUBLIC_FILES:
        src = ROOT / rel
        _copy_if_exists(src, out, missing)
        if src.is_file():
            copied.append(rel)


def _copy_public_figures(out: Path, copied: list[str], missing: list[str]) -> None:
    if not PUBLIC_FIGURES_DIR.is_dir():
        missing.append(str(PUBLIC_FIGURES_DIR.relative_to(ROOT)))
        return
    dest = out / "Docs" / "Paper" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(PUBLIC_FIGURES_DIR.glob("*.png")):
        shutil.copy2(path, dest / path.name)
        copied.append(str(path.relative_to(ROOT)))


def build_minimal(out: Path) -> tuple[list[str], list[str], str | None]:
    """Returns (copied_relative_paths, missing_relative_paths)."""
    copied: list[str] = []
    missing: list[str] = []
    dest_res = out / "results"
    seen: set[str] = set()

    def _track(src: Path, rel: str) -> None:
        if rel not in seen:
            seen.add(rel)
            copied.append(rel)

    for path in sorted(RESULTS.glob(f"paper_*{PRIMARY_RUN_GROUP}.json")):
        _copy_if_exists(path, dest_res, missing)
        if path.is_file():
            _track(path, str(path.relative_to(ROOT)))

    manifest = RESULTS / f"crossmodel_manifest_{PRIMARY_RUN_GROUP}.json"
    _copy_if_exists(manifest, dest_res, missing)
    if manifest.is_file():
        _track(manifest, str(manifest.relative_to(ROOT)))

    lofo_report = RESULTS / "lofo_calibration_report.md"
    _copy_if_exists(lofo_report, dest_res, missing)
    if lofo_report.is_file():
        _track(lofo_report, str(lofo_report.relative_to(ROOT)))

    # Pin config used for figure model list (reproducibility note).
    cfg = ROOT / "configs" / "default.yaml"
    _copy_if_exists(cfg, out / "configs", missing)
    if cfg.is_file():
        _track(cfg, str(cfg.relative_to(ROOT)))

    consolidated_name = _write_consolidated_heldout_manifest(dest_res)
    if consolidated_name is not None:
        copied.append(f"results/{consolidated_name}")

    _copy_public_code_tree(out, copied, missing)
    _copy_public_figures(out, copied, missing)
    return copied, missing, consolidated_name


def write_readme(out: Path, copied: list[str], full_data_new: bool, heldout_name: str | None) -> None:
    lines = [
        "# Frozen archive — Adaptive Suppression of Trigger-Word Persistence in Streaming LLM Dialogue",
        "",
        "This folder is the **single public archive repository** for the ESWA submission.",
        "It bundles the frozen code, prompts, configs, tests, and quantitative result artifacts",
        "so every submission component can point to one authoritative URI strategy.",
        "",
        "## Frozen release target",
        "",
        f"- Preferred repository root: `{ARCHIVE_REPO_URL}`",
        f"- Fixed release/tag name: `{FROZEN_RELEASE_TAG}`",
        f"- Primary publication-figure export run group: `{PRIMARY_RUN_GROUP}`",
        "- Optional DOI path: connect the repository to Zenodo after publishing the GitHub release.",
        "",
        "## Main-paper figure snapshot",
        "",
        "- `qwen3.5:4b`",
        "- `gemma4:e4b`",
        "- `openbmb/minicpm-v4.5:8b`",
        "- `ministral-3:8b`",
        "- `deepseek-chat`",
        "",
        "The archive includes the current `paper_*` export set that matches the manuscript PNG figures",
        "under `Docs/Paper/figures/`. Although the export run-group slug starts with `deepseek`, the",
        "aggregated `paper_*.json` files contain the merged five-model snapshot used by the paper figures.",
        "",
        "## Held-out materials",
        "",
    ]
    if heldout_name:
        lines.append(f"- Consolidated held-out manifest: `{heldout_name}`")
    lines.extend(
        [
            "- Legacy per-repeat held-out JSON files remain included only through the consolidated manifest.",
            "- Expanded held-out prompt suites live under `data/prompts/` in this same archive.",
            "",
            "## Repository layout",
            "",
            "```",
            "results/        # frozen paper exports + manifests + held-out freeze manifest",
            "Docs/Paper/figures/  # current manuscript PNG figures (`Figure_4`-`Figure_9`, etc.)",
            "experiments/    # experiment runners and artifact builder",
            "data/prompts/   # development and held-out prompt suites",
            "src/            # detector, controller, metrics, loaders",
            "configs/        # YAML configuration, including paper figure-model pin",
            "tests/          # regression checks",
            "scripts/        # archive and analysis helpers",
            "requirements.txt",
            "CITATION.cff",
            "ARCHIVE_RELEASE.md",
            "MANIFEST.sha256",
            "```",
            "",
            "Publication PNG figures are included under `Docs/Paper/figures/`.",
            "The paired `results/paper_*.json` files are the direct aggregated data exports behind those figures.",
            "",
        ]
    )
    if full_data_new:
        lines.extend(
            [
                "## Full local mirror",
                "",
                "This build also contains a full copy of `data_new/` from the local workspace.",
                "Use it only for byte-level inspection of intermediate files; the `results/` layer remains",
                "the public evidence layer tied to the manuscript.",
                "",
            ]
        )
    lines.extend(
        [
            "## Citation",
            "",
            "Use the repository root above together with the fixed release tag for the submission freeze.",
            "If a Zenodo DOI is minted later, treat that DOI as the preferred citable archive identifier.",
            "",
            "## Integrity",
            "",
            "SHA-256 checksums in `MANIFEST.sha256` cover only public archive contents.",
            "They intentionally exclude `.git/`, merge leftovers, and other repository-internal files.",
            "",
            "## Publishing the release",
            "",
            "See `ARCHIVE_RELEASE.md` for the exact local-to-GitHub release steps and the optional DOI workflow.",
            "",
            "## Full `data_new/` archive (optional)",
            "",
            "To bundle the entire local `data_new/` tree for deeper byte-level inspection, run from the project root:",
            "",
            "`python scripts/build_eswa_data_deposit.py --full-data-new`",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_archive_release_notes(out: Path) -> None:
    text = f"""# Archive release checklist

Canonical repository root:

`{ARCHIVE_REPO_URL}`

Submission-freeze release/tag:

`{FROZEN_RELEASE_TAG}`

## Recommended publication steps

1. Make this archive folder the root of the public repository.
2. Commit the frozen contents.
3. Create annotated tag `{FROZEN_RELEASE_TAG}`.
4. Publish a GitHub release for that tag.
5. If possible, connect the repository to Zenodo and mint a DOI for the same release.

## Notes

- The manuscript, cover letter, highlights, and bibliography should all point to the same release strategy.
- Do not describe the plain repository root as a permanent link unless the release/DOI is already live.
- `MANIFEST.sha256` must be regenerated from the clean public archive contents only.
"""
    (out / "ARCHIVE_RELEASE.md").write_text(text, encoding="utf-8")


def write_citation_cff(out: Path) -> None:
    text = f"""cff-version: 1.2.0
title: "Adaptive Suppression of Trigger-Word Persistence in Streaming LLM Dialogue"
message: "If you use this archive, please cite the frozen release or DOI."
type: software
authors:
  - family-names: "Li"
    given-names: "Jun-Xing"
version: "{FROZEN_RELEASE_TAG}"
repository-code: "{ARCHIVE_REPO_URL}"
license: "Apache-2.0"
abstract: "Single-repository frozen archive for the ESWA submission, bundling code, prompts, configs, tests, and quantitative result artifacts."
keywords:
  - "LLM reliability"
  - "streaming dialogue"
  - "trigger persistence"
  - "adaptive control"
"""
    (out / "CITATION.cff").write_text(text, encoding="utf-8")


def write_apache_license(out: Path) -> None:
    text = """Copyright 2026 Jun-Xing Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
    (out / "LICENSE").write_text(text, encoding="utf-8")


def write_manifest_sha256(out: Path) -> None:
    lines: list[str] = []
    for path in sorted(out.rglob("*")):
        rel = path.relative_to(out)
        parts = set(rel.parts)
        if not path.is_file() or path.name in MANIFEST_SKIP_NAMES:
            continue
        if parts & MANIFEST_SKIP_DIRS:
            continue
        if any(part.startswith(".git") for part in rel.parts):
            continue
        rel_posix = rel.as_posix()
        lines.append(f"{_sha256(path)}  {rel_posix}")
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build eswa_data_deposit/ for GitHub research-data repo.")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eswa_data_deposit",
        help="Output directory (default: ./eswa_data_deposit)",
    )
    ap.add_argument(
        "--full-data-new",
        action="store_true",
        help="Also copy entire data_new/ into output as data_new/ (large).",
    )
    args = ap.parse_args()
    out: Path = args.out.resolve()

    if not RESULTS.is_dir():
        print(f"Missing results directory: {RESULTS}", file=sys.stderr)
        return 1

    if out.exists():
        if (out / ".git").is_dir():
            # Refresh public archive contents but preserve the git checkout itself.
            for sub in ("results", "configs", "experiments", "src", "tests", "scripts", "data", "Docs", "code"):
                p = out / sub
                if p.is_dir():
                    shutil.rmtree(p)
            for name in ("README.md", "LICENSE", "MANIFEST.sha256", "CITATION.cff", "ARCHIVE_RELEASE.md", "requirements.txt"):
                p = out / name
                if p.exists():
                    p.unlink()
        else:
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    copied, missing, heldout_name = build_minimal(out)
    if missing:
        print("Warning: missing files (skipped):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)

    if args.full_data_new:
        src_dn = ROOT / "data_new"
        if src_dn.is_dir():
            shutil.copytree(src_dn, out / "data_new")
        else:
            print(f"Missing {src_dn}", file=sys.stderr)
            return 1

    write_readme(out, copied, args.full_data_new, heldout_name)
    write_archive_release_notes(out)
    write_citation_cff(out)
    write_apache_license(out)
    write_manifest_sha256(out)

    print(f"Wrote data deposit: {out}")
    print(f"Files copied (minimal layer): {len(copied)}")
    if args.full_data_new:
        print("Also copied full data_new/ tree.")
    return 0 if not missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
