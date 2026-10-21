"""Focused end-to-end tests for the paper-results aggregation CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_paper_results.py"
HAS_PANDAS = importlib.util.find_spec("pandas") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(HAS_PANDAS, "pandas is required for aggregation tests")
class AggregatePaperResultsTest(unittest.TestCase):
    def _write_fixture(self, input_root: Path) -> None:
        instance = input_root / "ExampleModel"
        instance.mkdir(parents=True)
        (instance / "pretrained.csv").write_text(
            "\n".join(
                [
                    "Dataset,Ridge,LogReg,RBF_SVM,RF",
                    "Alpha,0.80,0.82,0.81,0.83",
                    "Beta,0.60,0.62,0.61,0.63",
                    "Gamma,0.70,0.72,0.71,0.73",
                    "MEAN,0.70,0.72,0.71,0.73",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        for seed in (0, 7, 42, 123, 2026):
            beta_rf = "" if seed == 7 else "0.50"
            (instance / f"random_seed{seed}.csv").write_text(
                "\n".join(
                    [
                        "dataset,Ridge_acc,LogReg_acc,SVM_acc,RF_acc",
                        "Alpha,0.70,0.72,0.71,0.73",
                        f"Beta,0.50,0.52,0.51,{beta_rf}",
                        "Gamma,0.60,0.62,0.61,0.63",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    def test_cli_normalizes_wide_forms_and_excludes_incomplete_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            input_root = temporary_path / "input"
            output_dir = temporary_path / "output"
            self._write_fixture(input_root)
            command = [
                sys.executable,
                str(SCRIPT),
                "--input-root",
                str(input_root),
                "--output-dir",
                str(output_dir),
                "--bootstrap-repetitions",
                "100",
            ]
            if not HAS_SCIPY:
                command.append("--skip-inference")
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)

            import pandas as pd

            table = pd.read_csv(output_dir / "table1.csv")
            self.assertEqual(table.loc[0, "instance"], "ExampleModel")
            self.assertEqual(int(table.loc[0, "n_paired"]), 2)
            self.assertAlmostEqual(float(table.loc[0, "mean_delta_pp"]), 10.0)
            self.assertEqual(
                table.loc[0, "inference_status"],
                "computed" if HAS_SCIPY else "skipped",
            )

            classifier = pd.read_csv(output_dir / "classifier_deltas.csv")
            self.assertEqual(set(classifier["probe"]), {"Ridge", "LogReg", "RBF_SVM", "RF"})
            self.assertTrue((classifier["n_paired"] == 2).all())

            availability = pd.read_csv(output_dir / "availability.csv")
            beta = availability.loc[availability["dataset"] == "Beta"].iloc[0]
            self.assertFalse(bool(beta["included"]))
            self.assertIn("random_seed7", beta["reason"])
            self.assertTrue((output_dir / "paired_dataset_scores.csv").is_file())
            self.assertTrue((output_dir / "metadata.json").is_file())

    def test_fail_on_incomplete_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            input_root = temporary_path / "input"
            self._write_fixture(input_root)
            command = [
                sys.executable,
                str(SCRIPT),
                "--input-root",
                str(input_root),
                "--output-dir",
                str(temporary_path / "output"),
                "--bootstrap-repetitions",
                "10",
                "--fail-on-incomplete",
            ]
            if not HAS_SCIPY:
                command.append("--skip-inference")
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("incomplete", completed.stderr)


if __name__ == "__main__":
    unittest.main()
