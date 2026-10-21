"""Run the UniTS experiment from a selected UniTS checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys


HERE = Path(__file__).resolve().parent
UNITS_REVISION = "f8e5dfc71f6799164f30b090240a74ee70da01ca"


def _resolve_path_arguments(arguments: list[str], base_directory: Path) -> list[str]:
    """Resolve paths before changing into the external UniTS checkout."""
    resolved = list(arguments)
    path_flags = {"--yaml", "--ckpt", "--out"}
    for index, value in enumerate(resolved[:-1]):
        if value in path_flags:
            candidate = Path(resolved[index + 1]).expanduser()
            if not candidate.is_absolute():
                resolved[index + 1] = str((base_directory / candidate).resolve())
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-root", type=Path, required=True)
    parser.add_argument("experiment_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    original_cwd = Path.cwd()
    units_root = args.units_root.resolve()
    if not (units_root / "models" / "UniTS.py").is_file():
        parser.error(f"{units_root} is not a compatible UniTS checkout.")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(units_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = ""
    if revision != UNITS_REVISION:
        parser.error(
            f"UniTS checkout must be {UNITS_REVISION}; got {revision or 'not a git checkout'}."
        )
    clean_commands = (
        ["git", "-C", str(units_root), "diff", "--quiet"],
        ["git", "-C", str(units_root), "diff", "--cached", "--quiet"],
    )
    clean = all(
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
        for command in clean_commands
    )
    if not clean:
        parser.error(
            f"UniTS checkout has modified tracked files; restore a clean {UNITS_REVISION} checkout."
        )

    experiment_args = list(args.experiment_args)
    # ``run.sh`` uses ``--`` to protect the child script's flags from this
    # launcher. argparse retains that separator in REMAINDER on some Python
    # versions, so remove it before handing the arguments to UniTS.
    if experiment_args[:1] == ["--"]:
        experiment_args = experiment_args[1:]
    experiment_args = _resolve_path_arguments(experiment_args, original_cwd)
    os.chdir(units_root)
    sys.path.insert(0, str(units_root))
    sys.path.insert(0, str(HERE.parent))
    sys.argv = [str(HERE / "units_random.py"), *experiment_args]
    runpy.run_path(str(HERE / "units_random.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
