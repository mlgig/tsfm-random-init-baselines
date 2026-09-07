"""Create a path-configured Mantis run file for Linux."""

from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "mantis_random.py"
OLD = 'MANTIS_REPO = os.path.expanduser("~/workspace_pinar/mantis")'
NEW = 'MANTIS_REPO = os.environ["MANTIS_REPO"]'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = SOURCE.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        raise RuntimeError("Could not find the expected Mantis source-path line.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
