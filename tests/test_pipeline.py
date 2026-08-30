from pathlib import Path
import json
import pandas as pd

from src.pipeline import main


def test_pipeline_end_to_end():
    main()
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "artifacts" / "metrics.json").read_text())

    assert metrics["data_audit"]["rows"] == 1000
    assert metrics["data_audit"]["features"] == 20
    assert metrics["data_audit"]["bad_credit_rows"] == 300
    assert metrics["selected_model"] in {"logistic_regression", "random_forest", "gradient_boosting"}
    assert metrics["threshold_policy"]["validation_recall"] >= 0.70
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        assert 0 <= metrics["test_result"][key] <= 1
    assert metrics["test_result"]["brier"] >= 0
    assert 0 <= metrics["test_result"]["expected_calibration_error_10bin"] <= 1
    assert metrics["explainability"]["causal_claim"] is False
    assert metrics["explainability"]["transformed_feature_count"] > 20
    assert "not post-hoc probability calibrated" in metrics["calibration_note"]

    val = pd.read_csv(root / "artifacts" / "validation_metrics.csv")
    assert set(val["model"]) == {"logistic_regression", "random_forest", "gradient_boosting"}
    shap_global = pd.read_csv(root / "artifacts" / "global_shap_importance.csv")
    local = pd.read_csv(root / "artifacts" / "local_reason_codes.csv")
    subgroup = pd.read_csv(root / "artifacts" / "subgroup_diagnostics.csv")
    calibration = pd.read_csv(root / "artifacts" / "calibration_diagnostic.csv")

    assert len(shap_global) == metrics["explainability"]["transformed_feature_count"]
    assert shap_global["mean_abs_shap"].ge(0).all()
    assert 1 <= len(local) <= 25
    assert local["top_reason_codes"].str.contains("predicted bad-credit score").all()
    assert not subgroup.empty
    assert subgroup["rows"].ge(10).all()
    assert not calibration.empty
    assert calibration["rows"].sum() == metrics["data_audit"]["test_rows"]
    assert calibration["mean_predicted_risk"].between(0, 1).all()
    assert calibration["observed_bad_rate"].between(0, 1).all()
    assert (root / "artifacts" / "model.joblib").exists()
