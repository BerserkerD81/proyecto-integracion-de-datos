from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
from pathlib import Path
import pandas as pd
from mhealth_pipeline import predict_from_log
from model_registry import load_artifacts, list_models, read_feature_names

app = FastAPI(title="MHealth HAR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga perezosa de artefactos con soporte a múltiples modelos
ACTIVE_NAME, model, scaler, MODEL_PATH, SCALER_PATH = load_artifacts(None)
ACTIVE_FEATURES = read_feature_names(ACTIVE_NAME)

# Allowed activities (1..12) and names
ALLOWED_LABELS = set(range(1, 13))
ACTIVITY_NAMES = {
    1: 'Standing still',
    2: 'Sitting and relaxing',
    3: 'Lying down',
    4: 'Walking',
    5: 'Climbing stairs',
    6: 'Waist bends forward',
    7: 'Frontal elevation of arms',
    8: 'Knees bending (crouching)',
    9: 'Cycling',
    10: 'Jogging',
    11: 'Running',
    12: 'Jump front & back',
}


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
        "labels": {
            "allowed": sorted(ALLOWED_LABELS),
            "names": ACTIVITY_NAMES,
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
            "name": ACTIVE_NAME,
            "classes": classes.tolist() if hasattr(classes, "tolist") else classes,
        },
        "scaler": {
            "type": type(scaler).__name__,
            "path": str(SCALER_PATH),
        },
        "features": ACTIVE_FEATURES,
    }


@app.get("/models/list")
def models_list():
    return {"status": "ok", "items": list_models(), "active": ACTIVE_NAME}


class LoadModelPayload(BaseModel):
    name: str


@app.post("/models/load")
def models_load(payload: LoadModelPayload):
    global ACTIVE_NAME, model, scaler, MODEL_PATH, SCALER_PATH, ACTIVE_FEATURES
    try:
        ACTIVE_NAME, model, scaler, MODEL_PATH, SCALER_PATH = load_artifacts(payload.name)
        ACTIVE_FEATURES = read_feature_names(ACTIVE_NAME)
        return {
            "status": "ok",
            "active": ACTIVE_NAME,
            "model_path": str(MODEL_PATH),
            "scaler_path": str(SCALER_PATH),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/models/reload")
def models_reload():
    global ACTIVE_NAME, model, scaler, MODEL_PATH, SCALER_PATH, ACTIVE_FEATURES
    try:
        ACTIVE_NAME, model, scaler, MODEL_PATH, SCALER_PATH = load_artifacts(ACTIVE_NAME)
        ACTIVE_FEATURES = read_feature_names(ACTIVE_NAME)
        return {
            "status": "ok",
            "active": ACTIVE_NAME,
            "model_path": str(MODEL_PATH),
            "scaler_path": str(SCALER_PATH),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/json")
async def predict_json(payload: DetectJsonPayload = Body(...)):
    try:
        # Preferir nombres de features del artefacto si no vienen en payload
        colnames = payload.feature_names if payload.feature_names else ACTIVE_FEATURES
        df = pd.DataFrame(payload.rows, columns=colnames) if colnames else pd.DataFrame(payload.rows)
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

        # Filter predictions to allowed labels
        y_pred_series = pd.Series([p for p in y_pred if int(p) in ALLOWED_LABELS])
        majority_label = int(y_pred_series.mode().iloc[0]) if len(y_pred_series) else None
        avg_conf = float(pd.Series(y_proba.max(axis=1)).mean()) if y_proba is not None else None
        distribution = y_pred_series.value_counts(normalize=True).sort_index().to_dict()
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
                "labels_allowed": sorted(ALLOWED_LABELS),
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
        raw = predict_from_log(
            tmp_path,
            model,
            scaler,
            feature_names=ACTIVE_FEATURES,
            # Thresholds can be tuned via env vars inside pipeline; we pass None here
            min_label_purity=None,
            min_non_nan_ratio=None,
        )
        # Filter predictions to allowed labels
        preds = [p for p in raw.get("predictions", []) if int(p) in ALLOWED_LABELS]
        dist_series = pd.Series(preds)
        # Build duration summary from timelines
        timeline = [
            t for t in (raw.get("timeline", []) or [])
            if int(t.get("label", -1)) in ALLOWED_LABELS
        ]
        timeline_bars = [
            t for t in (raw.get("timeline_bars", []) or [])
            if int(t.get("label", -1)) in ALLOWED_LABELS
        ]
        # Use non-overlapping bars to avoid double counting due to window overlap
        durations: Dict[int, float] = {}
        total_time = 0.0
        for seg in (timeline_bars if timeline_bars else timeline):
            lbl = int(seg.get("label"))
            dur = float(seg.get("end_sec", 0.0)) - float(seg.get("start_sec", 0.0))
            durations[lbl] = durations.get(lbl, 0.0) + max(0.0, dur)
            total_time += max(0.0, dur)
        percentages = {lbl: (durations[lbl] / total_time) if total_time > 0 else 0.0 for lbl in durations}
        # Validation percentage as average max-proba
        proba_list = raw.get("proba", [])
        validation_pct = float(pd.Series(pd.DataFrame(proba_list).max(axis=1)).mean()) if proba_list else None
        # Confidence by activity: duration-weighted mean using timeline_bars if available
        confidence_by_activity: Dict[int, float] = {}
        if timeline_bars and any('confidence' in t for t in timeline_bars):
            sums: Dict[int, float] = {}
            durs: Dict[int, float] = {}
            for seg in timeline_bars:
                lbl_i = int(seg.get("label"))
                dur = float(seg.get("end_sec", 0.0)) - float(seg.get("start_sec", 0.0))
                conf = seg.get("confidence")
                if conf is not None and dur > 0:
                    sums[lbl_i] = sums.get(lbl_i, 0.0) + float(conf) * dur
                    durs[lbl_i] = durs.get(lbl_i, 0.0) + dur
            for k in sums:
                if durs.get(k, 0.0) > 0:
                    confidence_by_activity[k] = sums[k] / durs[k]
        elif proba_list:
            # fallback to window average if bars not present
            proba_df = pd.DataFrame(proba_list)
            maxp = proba_df.max(axis=1)
            sums: Dict[int, float] = {}
            counts: Dict[int, int] = {}
            for i, lbl in enumerate(raw.get("predictions", [])):
                lbl_i = int(lbl)
                if lbl_i in ALLOWED_LABELS:
                    val = float(maxp.iloc[i]) if i < len(maxp) else None
                    if val is not None:
                        sums[lbl_i] = sums.get(lbl_i, 0.0) + val
                        counts[lbl_i] = counts.get(lbl_i, 0) + 1
            for k in sums:
                if counts.get(k, 0) > 0:
                    confidence_by_activity[k] = sums[k] / counts[k]
        # Main activity by duration (fallback to most frequent if empty)
        main_label = None
        if durations:
            main_label = int(max(durations.items(), key=lambda kv: kv[1])[0])
        elif len(preds):
            main_label = int(dist_series.mode().iloc[0])

        return {
            "status": "ok",
            "count": len(preds),
            "prediction": {
                "label": main_label,
                "confidence": validation_pct,
                "distribution": {int(k): float(v) for k, v in dist_series.value_counts(normalize=True).sort_index().to_dict().items()},
            },
            "timeline": timeline,
            "timeline_bars": timeline_bars,
            "duration_seconds": durations,
            "percentage_by_activity": percentages,
            "confidence_by_activity": confidence_by_activity,
            "metadata": {"input_type": "file", "labels_allowed": sorted(ALLOWED_LABELS)},
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
    
