from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "credit_g.csv"
EXPECTED_ROWS = 1000
RANDOM_STATE = 42


def load_dataset(random_state: int = RANDOM_STATE):
    RAW.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        df = pd.read_csv(CACHE)
    else:
        bunch = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        df.to_csv(CACHE, index=False)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")
    if "class" not in df.columns:
        raise ValueError("Expected target column 'class'")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in credit-g")

    y = df["class"].astype(str).map({"good": 0, "bad": 1})
    if y.isna().any():
        raise ValueError("Unexpected target labels")
    X = df.drop(columns=["class"]).copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    audit = {
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "bad_credit_rows": int(y.sum()),
        "bad_credit_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_bad_rate": float(y_train.mean()),
        "validation_bad_rate": float(y_val.mean()),
        "test_bad_rate": float(y_test.mean()),
        "split_policy": "stratified 60/20/20 with locked test set",
    }
    return (
        X_train.reset_index(drop=True), y_train.reset_index(drop=True),
        X_val.reset_index(drop=True), y_val.reset_index(drop=True),
        X_test.reset_index(drop=True), y_test.reset_index(drop=True), audit
    )
