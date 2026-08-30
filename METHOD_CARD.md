# Explainable Credit Risk Method Card

## Intended use
Portfolio demonstration of binary credit-risk classification plus transparent model explanation on the German Credit benchmark.

## Evaluation design
The dataset is stratified into 60% train, 20% validation, and 20% locked test. Candidate models are selected using validation ROC-AUC, with PR-AUC and recall as tie-breakers. The operating threshold is chosen on validation only and then locked for the final test.

## Candidate models
- class-weighted Logistic Regression
- class-weighted Random Forest
- Gradient Boosting

## Explainability
Global importance is reported using mean absolute SHAP values on transformed model features. Local explanation reason codes report the strongest absolute SHAP contributions for individual test examples. Validation permutation importance is also exported as an independent feature-sensitivity diagnostic.

SHAP contributions explain the fitted model, not the real-world cause of credit outcomes. Reason codes are descriptive model-attribution signals and must not be presented as causal, legal, or adverse-action explanations.

## Threshold policy
The validation threshold maximizes F1 subject to at least 70% recall for the bad-credit class. This is an illustrative modeling policy, not a real lending loss function.

## Calibration diagnostic
Brier score is reported to assess probability quality. The current project does not claim formal probability calibration unless a separate calibration procedure is explicitly fitted and validated.

## Subgroup diagnostic
The pipeline reports descriptive performance slices where the benchmark exposes relevant fields. These slices are diagnostics only. They are not fairness certification, disparate-impact analysis, or evidence of legal compliance.

## Limitations
- historical small benchmark with only 1,000 rows;
- random stratified splitting, not temporal validation;
- benchmark variables and population do not represent modern production lending systems;
- explanation stability can vary across samples and model families;
- no causal interpretation;
- no legal, regulatory, fairness, or lending-decision recommendation claim.

## Production extensions
A production design would add temporal/out-of-time validation, explicit probability calibration, cost-sensitive policy design, explanation-stability testing, protected-attribute governance, formal fairness review, drift monitoring, model/version registry, human review controls, adverse-action governance, and jurisdiction-specific compliance review.
