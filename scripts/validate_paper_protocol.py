#!/usr/bin/env python3
"""Validate the tracked camera-ready dataset and coverage manifests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_names(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    all_datasets = read_names(ROOT / "data" / "paper_all_30_datasets.txt")
    common_datasets = read_names(ROOT / "data" / "paper_common_24_datasets.txt")
    seeds = [int(value) for value in read_names(ROOT / "data" / "paper_random_seeds.txt")]
    manifest = json.loads((ROOT / "protocol" / "paper_manifest.json").read_text(encoding="utf-8"))

    if len(all_datasets) != manifest["uea_dataset_count"] or len(set(all_datasets)) != len(all_datasets):
        raise ValueError("the full UEA dataset manifest must contain 30 unique names")
    if len(common_datasets) != 24 or not set(common_datasets).issubset(all_datasets):
        raise ValueError("the common dataset manifest must contain 24 names from the full archive")
    exclusions = set(manifest["common_cross_method_exclusions"])
    if set(all_datasets) - set(common_datasets) != exclusions:
        raise ValueError("common-subset exclusions do not match the manifest")
    if seeds != manifest["random_encoder_seeds"]:
        raise ValueError("the random seed list does not match the manifest")

    expected_total = 0
    for name, details in manifest["instances"].items():
        available = len(set(all_datasets) - set(details["unavailable"]))
        if available != details["expected_complete_datasets"]:
            raise ValueError(f"{name} has {available} available datasets, not {details['expected_complete_datasets']}")
        expected_total += available
    if expected_total != 183:
        raise ValueError(f"expected 183 paired observations, found {expected_total}")
    print("Camera-ready protocol manifests are internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
