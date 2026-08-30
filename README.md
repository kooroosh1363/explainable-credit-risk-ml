# DS-09 — Explainable Credit Risk ML

Portfolio-grade explainable credit-risk classification system built around leakage-aware evaluation, validation-only threshold selection, global/local model explanations, decision reason codes, subgroup diagnostics, tests, and CI.

## What this project demonstrates

- public German Credit benchmark via OpenML
- strict row/target/schema integrity checks
- stratified 60/20/20 train/validation/test design
- Logistic Regression, Random Forest, and Gradient Boosting candidates
- validation ROC-AUC-first selection with PR-AUC and recall tie-breaks
- validation-only threshold tuning for the bad-credit class
- ROC-AUC, PR-AUC, Brier score, Precision, Recall, and F1
- SHAP global importance
- SHAP local explanation reason codes
- permutation importance as an independent sensitivity diagnostic
- subgroup performance diagnostics
- explicit separation between model explanation and causal explanation
- model artifacts, pytest, and GitHub Actions CI

## Explainability architecture

```text
OpenML credit-g
   -> integrity audit
   -> stratified train / validation / locked test
   -> preprocessing
        numeric -> StandardScaler
        categorical -> OneHotEncoder
   -> candidate models
        Logistic Regression
        Random Forest
        Gradient Boosting
   -> validation model selection
   -> validation-only operating threshold
   -> locked test evaluation
   -> SHAP global explanations
   -> SHAP local reason codes
   -> permutation importance
   -> subgroup diagnostics
   -> artifacts + tests + CI
```

## Why explanation is separated from causality

SHAP describes how the fitted model distributes prediction contribution across model inputs. It does not prove why a person has a particular real-world credit outcome. The exported reason codes therefore explain **model behavior**, not causal mechanisms, legal adverse-action reasons, or regulatory compliance.

## Threshold policy

The operating threshold is selected only on validation. The current policy maximizes F1 while requiring at least 70% recall for the bad-credit class. This is an illustrative modeling policy rather than a real lender's loss function.

## Calibration

Brier score is reported as a probability-quality diagnostic. The project does not claim formal probability calibration unless a separate calibration method is explicitly fitted and validated.

## Subgroup diagnostics

Where the benchmark exposes suitable fields, the pipeline reports descriptive slices of predictive performance. These are diagnostic comparisons only and are not fairness certification, disparate-impact testing, or evidence of legal compliance.

## Claim boundary

This project demonstrates offline credit-risk classification and model explainability on the German Credit benchmark. It does not recommend lending approvals, establish causal explanations, certify fairness, or establish compliance with lending law.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline
```

See `DATA_SOURCE.md` and `METHOD_CARD.md` for source, methodology, and limitations.
