from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.data import load_dataset
from src.pipeline import main

ROOT = Path(__file__).resolve().parents[1]


def test_strict_premerge_hardening():
    """Independent invariants for data, model selection, thresholding, calibration, and explanations."""
    main()
    Xtr, ytr, Xv, yv, Xte, yte, audit = load_dataset(42)

    # Data integrity / split contract.
    assert audit["rows"] == 1000
    assert audit["features"] == 20
    assert audit["bad_credit_rows"] == 300
    assert len(Xtr) == 600 and len(Xv) == 200 and len(Xte) == 200
    assert abs(ytr.mean() - 0.30) < 1e-12
    assert abs(yv.mean() - 0.30) < 1e-12
    assert abs(yte.mean() - 0.30) < 1e-12
    assert not Xtr.isna().any().any()
    assert not Xv.isna().any().any()
    assert not Xte.isna().any().any()

    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    artifact = joblib.load(ROOT / "artifacts" / "model.joblib")
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    # Selection and generalization sanity.
    rows = {r["model"]: r for r in metrics["validation_results"]}
    assert metrics["selected_model"] == "random_forest"
    assert rows["random_forest"]["roc_auc"] >= rows["logistic_regression"]["roc_auc"]
    assert rows["random_forest"]["roc_auc"] >= rows["gradient_boosting"]["roc_auc"]
    assert metrics["test_result"]["roc_auc"] > 0.70
    assert abs(metrics["test_result"]["roc_auc"] - rows["random_forest"]["roc_auc"]) < 0.10

    # Recompute locked threshold behavior independently.
    val_prob = model.predict_proba(Xv)[:, 1]
    val_pred = (val_prob >= threshold).astype(int)
    val_recall = float(((val_pred == 1) & (yv.to_numpy() == 1)).sum() / (yv.to_numpy() == 1).sum())
    assert val_recall >= 0.70
    assert abs(val_recall - metrics["threshold_policy"]["validation_recall"]) < 1e-12

    test_prob = model.predict_proba(Xte)[:, 1]
    assert np.isfinite(test_prob).all()
    assert ((0.0 <= test_prob) & (test_prob <= 1.0)).all()
    assert np.std(test_prob) > 0.05

    # Calibration diagnostics must be internally consistent and non-trivial.
    calibration = pd.read_csv(ROOT / "artifacts" / "calibration_diagnostic.csv")
    assert not calibration.empty
    assert calibration["rows"].sum() == 200
    assert calibration["mean_predicted_risk"].between(0, 1).all()
    assert calibration["observed_bad_rate"].between(0, 1).all()
    assert 0 <= metrics["test_result"]["expected_calibration_error_10bin"] <= 1
    assert metrics["test_result"]["expected_calibration_error_10bin"] < 0.20

    # SHAP artifacts: complete, finite, directional, and aligned with transformed model space.
    shap_global = pd.read_csv(ROOT / "artifacts" / "global_shap_importance.csv")
    local = pd.read_csv(ROOT / "artifacts" / "local_reason_codes.csv")
    transformed_count = int(metrics["explainability"]["transformed_feature_count"])
    assert transformed_count == len(shap_global)
    assert transformed_count > 20
    assert shap_global["feature"].is_unique
    assert np.isfinite(shap_global["mean_abs_shap"]).all()
    assert (shap_global["mean_abs_shap"] >= 0).all()
    assert shap_global["mean_abs_shap"].sum() > 0
    assert 1 <= len(local) <= 25
    assert local["top_reason_codes"].str.contains("increases|decreases", regex=True).all()
    assert np.isfinite(local["base_value"]).all()

    # Original-feature permutation importance and subgroup diagnostics must be usable.
    perm = pd.read_csv(ROOT / "artifacts" / "permutation_importance.csv")
    subgroup = pd.read_csv(ROOT / "artifacts" / "subgroup_diagnostics.csv")
    assert len(perm) == 20
    assert perm["feature"].is_unique
    assert np.isfinite(perm[["importance_mean", "importance_std"]].to_numpy()).all()
    assert not subgroup.empty
    assert subgroup["rows"].ge(10).all()
    assert subgroup["bad_rate"].between(0, 1).all()

    # Claim-boundary guards.
    assert metrics["explainability"]["causal_claim"] is False
    text = metrics["claim_boundary"].lower()
    assert "no lending approval recommendation" in text
    assert "no fairness certification" in text
