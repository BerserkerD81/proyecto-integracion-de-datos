from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib
import tempfile
from pathlib import Path
import pandas as pd
from mhealth_pipeline import predict_from_log

app = FastAPI(title="MHealth HAR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("models/random_forest_rf.pkl")
SCALER_PATH = Path("models/scaler.pkl")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise RuntimeError("Faltan modelos en carpeta models/. Ejecuta el notebook para guardarlos.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "message": "MHealth HAR service running"}


class DetectJsonPayload(BaseModel):
    rows: Optional[List[List[float]]] = None
    feature_names: Optional[List[str]] = None
    subject_id: Optional[int] = 0


@app.get("/schema")
def schema():
    return {
        "status": "ok",
        "endpoints": {
            "GET /health": "Service status",
            "GET /schema": "API schema & usage",
            "GET /models/info": "Model metadata",
            "POST /predict/file": "Predict from .log file (multipart)",
            "POST /predict/json": "Predict from JSON rows",
        },
        "json_payload": {
            "rows": "List[List[float]] — feature rows per sample/window",
            "feature_names": "Optional[List[str]] — names aligned to training",
            "subject_id": "Optional[int] — subject context",
        },
        "response": {
            "status": "ok",
            "count": 1024,
            "prediction": {
                "label": 4,
                "confidence": 0.91,
                "distribution": {"1":0.12,"4":0.78,"9":0.10}
            },
            "metadata": {"input_type": "file|json"}
        }
    }


@app.get("/models/info")
def models_info():
    classes = getattr(model, "classes_", None)
    return {
        "status": "ok",
        "model": {
            "type": type(model).__name__,
            "path": str(MODEL_PATH),
            "classes": classes.tolist() if hasattr(classes, "tolist") else classes,
        },
        "scaler": {
            "type": type(scaler).__name__,
            "path": str(SCALER_PATH),
        },
    }


@app.post("/predict/json")
async def predict_json(payload: DetectJsonPayload = Body(...)):
    try:
        df = pd.DataFrame(payload.rows, columns=payload.feature_names) if payload.feature_names else pd.DataFrame(payload.rows)
        df = df.apply(pd.to_numeric, errors="coerce")
        df["subject"] = payload.subject_id if payload.subject_id is not None else 0

        X = df.drop(columns=[c for c in ["label"] if c in df.columns])
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature mismatch: {e}")

        y_pred = model.predict(X_scaled)
        try:
            y_proba = model.predict_proba(X_scaled)
        except Exception:
            y_proba = None

        majority_label = int(pd.Series(y_pred).mode().iloc[0]) if len(y_pred) else None
        avg_conf = float(pd.Series(y_proba.max(axis=1)).mean()) if y_proba is not None else None
        distribution = pd.Series(y_pred).value_counts(normalize=True).sort_index().to_dict()
        distribution = {int(k): float(v) for k, v in distribution.items()}

        return {
            "status": "ok",
            "count": len(df),
            "prediction": {
                "label": majority_label,
                "confidence": avg_conf,
                "distribution": distribution,
            },
            "metadata": {
                "input_type": "json",
                "feature_names": payload.feature_names,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".log"):
        raise HTTPException(status_code=400, detail="Sube un archivo .log")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        raw = predict_from_log(tmp_path, model, scaler)
        return {
            "status": "ok",
            "count": len(raw.get("predictions", [])),
            "prediction": {
                "label": int(pd.Series(raw.get("predictions", [])).mode().iloc[0]) if raw.get("predictions") else None,
                "confidence": float(pd.Series(pd.DataFrame(raw.get("proba", [])).max(axis=1)).mean()) if raw.get("proba") else None,
                "distribution": {int(k): float(v) for k, v in pd.Series(raw.get("predictions", [])).value_counts(normalize=True).sort_index().to_dict().items()},
            },
            "metadata": {"input_type": "file"},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/detect")
async def detect(file: UploadFile = File(None), payload: Optional[DetectJsonPayload] = None):
    if payload and payload.rows:
        return await predict_json(payload)
    if file is None:
        raise HTTPException(status_code=400, detail="Provide a .log file or JSON payload with rows")
    return await predict_file(file)
    
