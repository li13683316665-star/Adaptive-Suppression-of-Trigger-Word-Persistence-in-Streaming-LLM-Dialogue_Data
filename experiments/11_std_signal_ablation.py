"""
Leave-one-signal-out ablation for the Streaming Trigger Detector (STD).

For each lexical signal, zeroes the corresponding weight (others renormalized)
and re-runs detector evaluation to quantify marginal contribution to F1.
Requires a local inference backend. Example:
``python experiments/11_std_signal_ablation.py --model qwen2.5:7b --repeat 2``.
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

DROPS = ["", "f_freq", "f_surv", "f_leak", "f_pers", "f_lag"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STD signal ablation via 07_detector_eval.py")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument(
        "--families",
        nargs="*",
        default=["location", "emotion", "color"],
    )
    p.add_argument("--prompt-difficulty", default="d1", choices=["d1", "d2", "d3"])
    p.add_argument("--run-group", default="std_signal_ablation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    project_root = get_project_root()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest: dict[str, object] = {
        "created_at": ts,
        "run_group": args.run_group,
        "model": args.model,
        "drops": DROPS,
        "artifacts": [],
    }

    slug = args.model.replace(":", "_").replace(".", "_")

    for drop in DROPS:
        tag = drop or "full"
        stem = f"std_ablation_{slug}_{tag}"
        cmd = [
            sys.executable,
            str(project_root / "experiments" / "07_detector_eval.py"),
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
            "--std-drop-signal",
            drop,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=project_root, check=True)
        # latest json for this stem
        matches = sorted(results_dir.glob(f"{stem}_*.json"), key=lambda p: p.stat().st_mtime)
        if matches:
            manifest["artifacts"].append({"drop": tag, "file": matches[-1].name})

    out = results_dir / f"std_signal_ablation_manifest_{args.run_group}_{ts}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest {out}")


if __name__ == "__main__":
    main()
