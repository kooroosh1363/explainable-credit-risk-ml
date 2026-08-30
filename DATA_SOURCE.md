# Data Source

Dataset: OpenML `credit-g`, version 1 (German Credit benchmark).

Integrity checks used by the pipeline:
- 1,000 rows
- 20 predictors
- target column: `class`
- target labels mapped as `good -> 0`, `bad -> 1`
- expected bad-credit rows: 300
- no missing values accepted

The dataset is fetched with `sklearn.datasets.fetch_openml`, cached locally under `data/raw/credit_g.csv`, and the cache is not committed to Git.

This historical benchmark is used for portfolio/educational modeling only. It is not evidence that the model is appropriate for real lending decisions, any jurisdiction, or any protected-group compliance requirement.
