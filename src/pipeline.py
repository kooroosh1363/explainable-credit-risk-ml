from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load_dataset

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RANDOM_STATE = 42


def make_preprocessor(X: pd.DataFrame):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ], verbose_feature_names_out=False)
    return prep, num, cat


def build_models(X: pd.DataFrame):
    prep, _, _ = make_preprocessor(X)
    return {
        "logistic_regression": Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(X)[0]),
            ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", make_preprocessor(X)[0]),
            ("model", GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=2, random_state=RANDOM_STATE)),
        ]),
    }


def metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def choose_threshold(y_true, prob, min_recall=0.70):
    best = None
    for t in np.unique(prob):
        pred = prob >= t
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if rec >= min_recall:
            key = (-f1, -prec, -rec, float(t))
            if best is None or key < best[0]:
                best = (key, float(t), float(rec), float(prec), float(f1))
    if best is None:
        return 0.5, 0.0, 0.0, 0.0
    return best[1], best[2], best[3], best[4]


def explain_model(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
    prep = model.named_steps["prep"]
    clf = model.named_steps["model"]
    Xt_train = prep.transform(X_train)
    Xt_test = prep.transform(X_test)
    names = prep.get_feature_names_out().tolist()

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, Xt_train)
        sv = explainer(Xt_test)
        values = np.asarray(sv.values)
        base = np.asarray(sv.base_values)
    else:
        explainer = shap.TreeExplainer(clf)
        sv = explainer(Xt_test)
        values = np.asarray(sv.values)
        base = np.asarray(sv.base_values)
        if values.ndim == 3:
            values = values[:, :, 1]
        if base.ndim > 1:
            base = base[:, 1]

    global_imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
    global_imp = global_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    local_rows = []
    for i in range(min(25, len(X_test))):
        order = np.argsort(np.abs(values[i]))[::-1][:5]
        reasons = [f"{names[j]} ({values[i, j]:+.4f})" for j in order]
        local_rows.append({"row": i, "top_reason_codes": " | ".join(reasons), "base_value": float(np.ravel(base)[i if np.ravel(base).size > 1 else 0])})
    return global_imp, pd.DataFrame(local_rows)


def subgroup_diagnostics(X_test, y_test, prob, threshold):
    rows = []
    for feature in ["personal_status", "age"]:
        if feature not in X_test.columns:
            continue
        if feature == "age":
            groups = pd.cut(X_test[feature], bins=[0, 25, 40, 60, 120], include_lowest=True).astype(str)
        else:
            groups = X_test[feature].astype(str)
        for group in sorted(groups.unique()):
            mask = (groups == group).to_numpy()
            if mask.sum() < 10:
                continue
            m = metrics(y_test.to_numpy()[mask], prob[mask], threshold)
            rows.append({"feature": feature, "group": group, "rows": int(mask.sum()), "bad_rate": float(y_test.to_numpy()[mask].mean()), **m})
    return pd.DataFrame(rows)


def main():
    ART.mkdir(exist_ok=True)
    Xtr, ytr, Xv, yv, Xte, yte, audit = load_dataset(RANDOM_STATE)

    fitted = {}
    val_rows = []
    for name, model in build_models(Xtr).items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        prob = model.predict_proba(Xv)[:, 1]
        row = metrics(yv, prob)
        row["model"] = name
        val_rows.append(row)

    val_df = pd.DataFrame(val_rows).sort_values(["roc_auc", "pr_auc", "recall"], ascending=False).reset_index(drop=True)
    selected = str(val_df.iloc[0]["model"])
    model = fitted[selected]
    val_prob = model.predict_proba(Xv)[:, 1]
    threshold, val_recall, val_precision, val_f1 = choose_threshold(yv, val_prob, min_recall=0.70)

    test_prob = model.predict_proba(Xte)[:, 1]
    test_metrics = metrics(yte, test_prob, threshold)

    perm = permutation_importance(model, Xv, yv, scoring="roc_auc", n_repeats=15, random_state=RANDOM_STATE)
    perm_df = pd.DataFrame({"feature": Xv.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)

    global_shap, local_reasons = explain_model(model, Xtr, Xte)
    subgroup = subgroup_diagnostics(Xte, yte, test_prob, threshold)

    joblib.dump({"model": model, "threshold": threshold}, ART / "model.joblib")
    val_df.to_csv(ART / "validation_metrics.csv", index=False)
    perm_df.to_csv(ART / "permutation_importance.csv", index=False)
    global_shap.to_csv(ART / "global_shap_importance.csv", index=False)
    local_reasons.to_csv(ART / "local_reason_codes.csv", index=False)
    subgroup.to_csv(ART / "subgroup_diagnostics.csv", index=False)

    summary = {
        "data_audit": audit,
        "candidate_models": list(fitted.keys()),
        "selection_policy": "highest validation ROC-AUC; PR-AUC then recall tie-break",
        "selected_model": selected,
        "validation_results": val_df.to_dict(orient="records"),
        "threshold_policy": {
            "minimum_validation_recall": 0.70,
            "selected_threshold": float(threshold),
            "validation_recall": val_recall,
            "validation_precision": val_precision,
            "validation_f1": val_f1,
        },
        "test_result": test_metrics,
        "explainability": {
            "global_method": "mean absolute SHAP on transformed model features",
            "local_method": "top absolute SHAP contributions as reason codes",
            "permutation_importance": "validation ROC-AUC decrease under feature permutation",
            "causal_claim": False,
        },
        "claim_boundary": "offline credit-risk classification and model explainability on German Credit; no lending approval recommendation, causal explanation, legal compliance claim, or fairness certification",
    }
    (ART / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
