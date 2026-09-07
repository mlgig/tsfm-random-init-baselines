"""Run the UniTS experiment from a selected UniTS checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-root", type=Path, required=True)
    parser.add_argument("experiment_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    units_root = args.units_root.resolve()
    if not (units_root / "models" / "UniTS.py").is_file():
        parser.error(f"{units_root} is not a compatible UniTS checkout.")

    os.chdir(units_root)
    sys.path.insert(0, str(units_root))
    sys.argv = [str(HERE / "units_random.py"), *args.experiment_args]
    runpy.run_path(str(HERE / "units_random.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
