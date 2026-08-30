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
    assert metrics["explainability"]["causal_claim"] is False

    val = pd.read_csv(root / "artifacts" / "validation_metrics.csv")
    assert set(val["model"]) == {"logistic_regression", "random_forest", "gradient_boosting"}
    shap_global = pd.read_csv(root / "artifacts" / "global_shap_importance.csv")
    local = pd.read_csv(root / "artifacts" / "local_reason_codes.csv")
    subgroup = pd.read_csv(root / "artifacts" / "subgroup_diagnostics.csv")
    assert len(shap_global) > 20
    assert shap_global["mean_abs_shap"].ge(0).all()
    assert 1 <= len(local) <= 25
    assert not subgroup.empty
    assert (root / "artifacts" / "model.joblib").exists()
