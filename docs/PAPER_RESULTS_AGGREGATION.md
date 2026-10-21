# Combining paper results

Use `scripts/aggregate_paper_results.py` to combine standard runner CSVs. It
creates a Table-1-style summary for each model instance, classifier deltas,
paired-availability information, and machine-readable paired dataset scores.

Arrange the files as below. Copy
`templates/paper-results/ExampleInstance` once for each model instance.

~~~text
paper-input/
  MOMENT-Small/
    pretrained.csv
    random_seed0.csv
    random_seed7.csv
    random_seed42.csv
    random_seed123.csv
    random_seed2026.csv
  Mantis/
    ...
~~~

Each CSV must contain one result per dataset and all four probes. The preferred
wide header is:

~~~csv
Dataset,Ridge,LogReg,RBF_SVM,RF
~~~

Runner-style names such as `Ridge_acc`, `LogReg_acc`, `SVM_acc`, and `RF_acc`
are normalized automatically. The script also accepts long-form files with
dataset, probe/classifier, and accuracy/acc columns. Summary rows called
`MEAN` are ignored.

Run the aggregation in an environment with NumPy, pandas, and SciPy:

~~~bash
python scripts/aggregate_paper_results.py \
  --input-root /path/to/paper-input \
  --output-dir results/paper-summary
~~~

The default seed set is exactly `0 7 42 123 2026`. For each instance and
dataset, all four probe scores must be finite in the pretrained result and in
every random-seed result. Incomplete pairs are excluded from Table 1 and kept
with a reason in `availability.csv`. Add `--fail-on-incomplete` to stop with
an error instead.

The output files are:

- `table1.csv` — paired aggregate accuracy, random-seed standard deviation,
  paired delta in percentage points, percentile bootstrap interval, Wilcoxon
  p-value, and Holm-adjusted p-value.
- `classifier_deltas.csv` — paired mean and median deltas for Ridge, LogReg,
  RBF SVM, and random forest.
- `paired_dataset_scores.csv` — unrounded dataset-level values used in the
  statistics.
- `availability.csv` — every included or excluded dataset and its reason.
- `metadata.json` — fixed protocol details.

SciPy is required by default. The camera-ready statistics use a two-sided
Wilcoxon signed-rank test, Holm correction, and a 10,000-resample percentile
bootstrap. Use `--skip-inference` only to inspect aggregates in an environment
without SciPy. It intentionally leaves the inference fields empty.

Run the focused synthetic regression test:

~~~bash
python -m unittest tests/test_aggregate_paper_results.py
~~~
