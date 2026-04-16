"""
Deprecated compatibility wrapper.

The ESWA submission now uses a SINGLE public archive repository that bundles
code and frozen data together. This script therefore forwards users to
``scripts/build_eswa_data_deposit.py`` instead of producing a separate code-only
repository layout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPED_SCRIPT = ROOT / "scripts" / "build_eswa_data_deposit.py"


def main() -> int:
    if not WRAPPED_SCRIPT.is_file():
        print(f"Missing archive builder: {WRAPPED_SCRIPT}", file=sys.stderr)
        return 1
    print(
        "Separate code-only archives are retired for the ESWA submission. "
        "Building the single frozen archive instead...",
        file=sys.stderr,
    )
    subprocess.run([sys.executable, str(WRAPPED_SCRIPT)], cwd=ROOT, check=True)
    print(f"Wrote {ROOT / 'eswa_data_deposit'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
