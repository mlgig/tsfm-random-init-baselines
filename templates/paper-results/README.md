# Raw-result input template

Copy `ExampleInstance` to one directory for each paper instance, such as
`MOMENT-Small`, `Mantis`, or `NuTime`. Replace the header-only files with
runner output. Use these filenames:

~~~text
pretrained.csv
random_seed0.csv
random_seed7.csv
random_seed42.csv
random_seed123.csv
random_seed2026.csv
~~~

The preferred CSV header is:

~~~csv
Dataset,Ridge,LogReg,RBF_SVM,RF
~~~

`scripts/aggregate_paper_results.py` also accepts `*_acc` columns and maps
`SVM` to `RBF_SVM`.
