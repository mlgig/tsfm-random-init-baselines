"""Create an executable NuTime notebook with local run paths filled in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init", choices=["pretrained", "random"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    notebook = json.loads((HERE / "nutime_random.ipynb").read_text(encoding="utf-8"))
    replacements = {
        'INIT = "pretrained"': f"INIT = {args.init!r}",
        "SEED = 0": f"SEED = {args.seed}",
        'checkpoint_path = "ckpt/checkpoint_bias9.pth"': f"checkpoint_path = {str(args.checkpoint.resolve())!r}",
        'base_data_dir = "dataset"': f"base_data_dir = {str(args.data_root.resolve())!r}",
    }
    text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    for old in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"Expected exactly one occurrence of {old!r}.")
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            for old, new in replacements.items():
                source = source.replace(old, new)
            cell["source"] = source.splitlines(keepends=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
