"""Раннер тестов: делегирует pytest; без pytest — только фреймворко-независимые core-тесты.

    python tests/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        import pytest  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_core

        tests = [(n, f) for n, f in vars(test_core).items() if n.startswith("test_") and callable(f)]
        failed = 0
        for name, fn in sorted(tests):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                failed += 1
                import traceback

                traceback.print_exc()
                print(f"FAIL {name}")
        print(f"\n{len(tests) - failed}/{len(tests)} core-тестов прошло (pytest не установлен)")
        return 1 if failed else 0
    return subprocess.call([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
