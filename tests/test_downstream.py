"""Regression tests for the shared camera-ready downstream protocol."""

from __future__ import annotations

import unittest

import numpy as np

from common.downstream import PROBE_ORDER, evaluate_probes, probe_configurations


class DownstreamProtocolTest(unittest.TestCase):
    def test_table_five_grids_and_scaling_match_the_manifest(self) -> None:
        configurations = probe_configurations()
        self.assertEqual(tuple(configurations), PROBE_ORDER)

        ridge, ridge_grid = configurations["Ridge"]
        self.assertEqual(list(ridge.named_steps), ["scaler", "ridge"])
        self.assertEqual(ridge_grid["ridge__alpha"], [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3])

        logreg, logreg_grid = configurations["LogReg"]
        self.assertEqual(list(logreg.named_steps), ["scaler", "logreg"])
        self.assertEqual(logreg.named_steps["logreg"].max_iter, 5000)
        self.assertEqual(logreg_grid["logreg__C"], ridge_grid["ridge__alpha"])

        svm, svm_grid = configurations["RBF_SVM"]
        self.assertEqual(list(svm.named_steps), ["scaler", "svm"])
        self.assertEqual(svm_grid["svm__C"], [1e-2, 1e-1, 1.0, 10.0, 1e2])
        self.assertEqual(svm_grid["svm__gamma"], ["scale", "auto"])

        forest, forest_grid = configurations["RF"]
        self.assertEqual(list(forest.named_steps), ["rf"])
        self.assertEqual(forest.named_steps["rf"].n_estimators, 200)
        self.assertEqual(forest.named_steps["rf"].max_features, "sqrt")
        self.assertEqual(forest_grid["rf__max_depth"], [None, 10])
        self.assertEqual(forest_grid["rf__min_samples_leaf"], [1, 2])

    def test_ineligible_training_split_is_recorded_for_every_probe(self) -> None:
        train_x = np.arange(12, dtype=float).reshape(6, 2)
        train_y = np.array([0, 0, 1, 1, 1, 1])
        test_x = np.arange(8, dtype=float).reshape(4, 2)
        test_y = np.array([0, 1, 0, 1])

        results = evaluate_probes(train_x, train_y, test_x, test_y, n_jobs=1)

        self.assertEqual(tuple(results), PROBE_ORDER)
        for result in results.values():
            self.assertTrue(np.isnan(result["accuracy"]))
            self.assertIn("requires at least 3 examples", result["error"])


if __name__ == "__main__":
    unittest.main()
