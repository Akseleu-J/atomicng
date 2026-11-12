"""Building a Docs Site: Collects a Site_docs/ The MD (Single Source of Truth) → Mkdocs build.

Launch: Python scripts/build_docs.py [--serve]   (requires mkdocs-material)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_DOCS = ROOT / "site_docs"

PAGES = {
    "README.md": "index.md",
    "CALIBRATION.md": "CALIBRATION.md",
    "FIDELITY.md": "FIDELITY.md",
    "CHANGELOG.md": "CHANGELOG.md",
    "docs/vllm.md": "vllm.md",
}


def assemble() -> None:
    if SITE_DOCS.exists():
        shutil.rmtree(SITE_DOCS)
    SITE_DOCS.mkdir()
    for src, dst in PAGES.items():
        text = (ROOT / src).read_text(encoding="utf-8")
        # Links to local repo files → GitHub anchors (not available on the site))
        text = text.replace("](../FRAMEWORK_PLAN.md)", "](https://github.com/Akseleu-J/atomicng#readme)")
        (SITE_DOCS / dst).write_text(text, encoding="utf-8")
    print("site_docs assembled:", sorted(p.name for p in SITE_DOCS.iterdir()))


def main() -> int:
    assemble()
    cmd = [sys.executable, "-m", "mkdocs", *("serve".split() if "--serve" in sys.argv else ["build"])]
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
