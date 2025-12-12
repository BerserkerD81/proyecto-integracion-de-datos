from pathlib import Path
import pandas as pd
import numpy as np

def infer_columns(sample_path: Path):
    sample = pd.read_csv(sample_path, sep=r"\s+", header=None, nrows=5)
    n_cols = sample.shape[1]
    return [f"f{i}" for i in range(1, n_cols)] + ["label"]


def load_log(path: Path, colnames, subject_id: int = 1):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=colnames)
    for col in df.columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df["subject"] = subject_id
    return df


def preprocess_for_model(df: pd.DataFrame, target_col: str = "label"):
    X = df.select_dtypes(include=[np.number]).copy()
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
    if "subject" not in X.columns:
        X["subject"] = 1
    return X


def predict_from_log(path, model, scaler):
    path = Path(path)
    colnames = infer_columns(path)
    df = load_log(path, colnames)
    X = preprocess_for_model(df, target_col="label")
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
    return {
        "predictions": y_pred.tolist(),
        "proba": proba.tolist() if proba is not None else None,
    }
