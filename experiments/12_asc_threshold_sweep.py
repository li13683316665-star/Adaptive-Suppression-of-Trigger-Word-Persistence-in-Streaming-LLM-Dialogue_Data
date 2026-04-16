"""
Sensitivity analysis for the Adaptive Suppression Controller (ASC) thresholds.

Runs a grid of ``08_adaptive_eval.py`` jobs over guardrail, repetition, and
context-reset tiers (and optional escalation delays), and writes a manifest of
output stems. Example:
``python experiments/12_asc_threshold_sweep.py --model qwen2.5:7b --repeat 1``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASC threshold sweep using 08_adaptive_eval.py")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--repeat", type=int, default=1, help="Repeats per configuration (use >=10 for ESWA).")
    p.add_argument(
        "--families",
        nargs="*",
        default=["location"],
        help="Limit families to save time (default: location only).",
    )
    p.add_argument("--prompt-difficulty", default="d1", choices=["d1", "d2", "d3"])
    p.add_argument("--skip-quality", action="store_true", help="Pass through to 08.")
    p.add_argument(
        "--guardrail-grid",
        type=float,
        nargs="*",
        default=[0.20],
        help="Values for --asc-threshold-guardrail",
    )
    p.add_argument(
        "--repeat-tier-grid",
        type=float,
        nargs="*",
        default=[0.40],
        help="Values for --asc-threshold-repeat",
    )
    p.add_argument(
        "--reset-tier-grid",
        type=float,
        nargs="*",
        default=[0.60],
        help="Values for --asc-threshold-reset",
    )
    p.add_argument("--run-group", default="asc_threshold_sweep")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    project_root = get_project_root()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = args.model.replace(":", "_").replace(".", "_")

    manifest: dict[str, object] = {
        "created_at": ts,
        "run_group": args.run_group,
        "model": args.model,
        "configs": [],
    }

    for g in args.guardrail_grid:
        for rpt in args.repeat_tier_grid:
            for rst in args.reset_tier_grid:
                stem = f"asc_sweep_{slug}_g{g}_r{rpt}_c{rst}".replace(".", "p")
                cmd = [
                    sys.executable,
                    str(project_root / "experiments" / "08_adaptive_eval.py"),
                    "--model",
                    args.model,
                    "--host",
                    args.host,
                    "--repeat",
                    str(args.repeat),
                    "--families",
                    *args.families,
                    "--prompt-difficulty",
                    args.prompt_difficulty,
                    "--output-stem",
                    stem,
                    "--run-group",
                    args.run_group,
                    "--asc-threshold-guardrail",
                    str(g),
                    "--asc-threshold-repeat",
                    str(rpt),
                    "--asc-threshold-reset",
                    str(rst),
                ]
                if args.skip_quality:
                    cmd.append("--skip-quality")
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, cwd=project_root, check=True)
                matches = sorted(results_dir.glob(f"{stem}_*.json"), key=lambda p: p.stat().st_mtime)
                entry = {
                    "guardrail": g,
                    "repeat_penalty_tier": rpt,
                    "context_reset_tier": rst,
                    "file": matches[-1].name if matches else None,
                }
                manifest["configs"].append(entry)

    out = results_dir / f"asc_threshold_sweep_manifest_{args.run_group}_{ts}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
