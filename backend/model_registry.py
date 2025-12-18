from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import joblib
import os


MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _norm_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return str(name).strip().replace(" ", "_")


def model_candidates(name: Optional[str]) -> List[Path]:
    """Return possible model filenames for a given logical name.

    Supports legacy and new naming conventions.
    """
    if not name:
        return []
    name = _norm_name(name)
    cands = [
        MODEL_DIR / f"{name}.pkl",
        MODEL_DIR / f"{name}_rf.pkl",
        MODEL_DIR / f"{name}_lr.pkl",
        MODEL_DIR / f"{name}_xgb.pkl",
        MODEL_DIR / f"{name}_model.pkl",
    ]
    # Legacy known filenames
    if name in {"random_forest", "rf", "random_forest_rf"}:
        cands.insert(0, MODEL_DIR / "random_forest_rf.pkl")
    if name in {"logistic_regression", "lr", "logistic_regression_lr"}:
        cands.insert(0, MODEL_DIR / "logistic_regression_lr.pkl")
    return cands


def scaler_candidates(name: Optional[str]) -> List[Path]:
    """Return possible scaler filenames. Prefer per-model scaler, fallback to generic."""
    cands: List[Path] = []
    if name:
        name = _norm_name(name)
        cands.extend([
            MODEL_DIR / f"scaler_{name}.pkl",
            MODEL_DIR / f"{name}_scaler.pkl",
        ])
    cands.append(MODEL_DIR / "scaler.pkl")
    return cands


def features_path(name: Optional[str]) -> Path:
    nm = _norm_name(name) or "default"
    return MODEL_DIR / f"features_{nm}.json"


def save_for_backend(model, scaler, name: str = "random_forest", feature_names: Optional[List[str]] = None, overwrite: bool = True) -> Dict[str, str]:
    """Save model/scaler/feature_names into backend/models for direct backend use.

    Returns saved paths.
    """
    nm = _norm_name(name) or "model"
    model_path = MODEL_DIR / f"{nm}.pkl"
    scaler_path = MODEL_DIR / f"scaler_{nm}.pkl"
    if not overwrite and (model_path.exists() or scaler_path.exists()):
        raise FileExistsError(f"Artifacts already exist for name '{nm}'. Set overwrite=True to replace.")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    feat_path = None
    if feature_names is not None:
        feat_path = features_path(nm)
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump({"feature_names": list(feature_names)}, f, ensure_ascii=False, indent=2)
    return {
        "model": str(model_path),
        "scaler": str(scaler_path),
        "features": str(feat_path) if feat_path else "",
    }


def list_models() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for p in sorted(MODEL_DIR.glob("*.pkl")):
        if p.name.startswith("scaler"):
            continue
        nm = p.stem
        # pair with scaler candidates
        scl = None
        for s in scaler_candidates(nm):
            if s.exists():
                scl = s
                break
        items.append({
            "name": nm,
            "model": str(p),
            "scaler": str(scl) if scl else "",
        })
    return items


def load_artifacts(name: Optional[str] = None):
    """Load model and scaler for a given name. If name is None, use env BACKEND_MODEL_NAME
    or fallback to the first available model.
    """
    target = _norm_name(name) or _norm_name(os.getenv("BACKEND_MODEL_NAME"))
    if target:
        m_path = next((p for p in model_candidates(target) if p.exists()), None)
        if not m_path:
            raise FileNotFoundError(f"Model file not found for name '{target}'. Place it under {MODEL_DIR}.")
        s_path = next((p for p in scaler_candidates(target) if p.exists()), None)
        if not s_path:
            raise FileNotFoundError(f"Scaler file not found for name '{target}'. Place it under {MODEL_DIR}.")
        model = joblib.load(m_path)
        scaler = joblib.load(s_path)
        return target, model, scaler, m_path, s_path

    # No explicit target: prefer common default names, else first found in directory
    # Prefer random_forest if available to match prior behavior
    preferred = [
        "random_forest_rf",
        "random_forest",
    ]
    for pref in preferred:
        mp = next((p for p in model_candidates(pref) if p.exists()), None)
        if mp:
            sp = next((p for p in scaler_candidates(pref) if p.exists()), None)
            if not sp:
                break
            model = joblib.load(mp)
            scaler = joblib.load(sp)
            return pref, model, scaler, mp, sp

    models = list_models()
    if not models:
        raise FileNotFoundError(f"No models found under {MODEL_DIR}. Export them from the notebook.")
    first = models[0]
    model = joblib.load(Path(first["model"]))
    scaler = joblib.load(Path(first["scaler"])) if first.get("scaler") else None
    return first["name"], model, scaler, Path(first["model"]), Path(first.get("scaler") or "")


def read_feature_names(name: Optional[str]) -> Optional[List[str]]:
    p = features_path(name)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            feats = data.get("feature_names")
            if isinstance(feats, list):
                return feats
        except Exception:
            return None
    return None
