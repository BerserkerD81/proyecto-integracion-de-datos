from pathlib import Path
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

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

SAMPLING_HZ = 50
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLING_HZ * WINDOW_SECONDS)  # 100
STEP_FRACTION = 0.5
STEP_SIZE = int(WINDOW_SIZE * STEP_FRACTION)     # 50
EXCLUDE_LABELS = {0}

# Filtering thresholds
MIN_LABEL_PURITY = 0.6        # require >=60% of window to be the majority label (when labels present)
MIN_NON_NAN_RATIO = 0.8       # require >=80% non-NaN values per window across numeric columns

# Feature set used across training and inference. Keep in sync with notebook.
STAT_KEYS = [
    'mean','std','min','max','median','q25','q75','iqr','mad','rms','energy',
    'skew','kurt','ptp','slope','zcr','acf1','n_peaks'
]

def _safe(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[~np.isnan(a)] if np.isnan(a).any() else a

def _feat_stats(a: np.ndarray) -> Dict[str, float]:
    a = _safe(a)
    n = len(a)
    if n == 0:
        return {k: np.nan for k in STAT_KEYS}
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=0))
    amin = float(np.min(a))
    amax = float(np.max(a))
    median = float(np.median(a))
    q25 = float(np.percentile(a, 25))
    q75 = float(np.percentile(a, 75))
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(a - median)))
    rms = float(np.sqrt(np.mean(a**2)))
    energy = float(np.mean(a**2))
    if std > 1e-12:
        skew = float(np.mean(((a - mean) / std) ** 3))
        kurt = float(np.mean(((a - mean) / std) ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = -3.0
    ptp = float(np.ptp(a))
    if n >= 2:
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        denom = np.sum((x - x_mean) ** 2)
        slope = float(np.sum((x - x_mean) * (a - mean)) / denom) if denom > 0 else 0.0
        zcr = float(np.mean(np.sign(a[1:]) != np.sign(a[:-1])))
        # lag-1 autocorrelation (normalized)
        if std > 1e-12:
            a0 = a[:-1] - mean
            a1 = a[1:] - mean
            num = float(np.sum(a0 * a1))
            den = float(np.sum((a - mean) ** 2))
            acf1 = (num / den) if den > 0 else 0.0
        else:
            acf1 = 1.0
        # simple peak count (local maxima above mean + 0.5*std)
        if n >= 3:
            thr = mean + 0.5 * std
            d1 = np.diff(a)
            sign_change = (np.sign(d1[:-1]) > 0) & (np.sign(d1[1:]) < 0)
            mids = a[1:-1] > thr
            n_peaks = int(np.sum(sign_change & mids))
        else:
            n_peaks = 0
    else:
        slope = 0.0
        zcr = 0.0
        acf1 = 0.0
        n_peaks = 0
    return {
        'mean': mean, 'std': std, 'min': amin, 'max': amax, 'median': median,
        'q25': q25, 'q75': q75, 'iqr': iqr, 'mad': mad, 'rms': rms, 'energy': energy,
        'skew': skew, 'kurt': kurt, 'ptp': ptp, 'slope': slope, 'zcr': zcr,
        'acf1': acf1, 'n_peaks': float(n_peaks)
    }

def _window_iter(n_rows: int, window: int, step: int):
    start = 0
    while start + window <= n_rows:
        yield start, start + window
        start += step

def build_window_features(
    df: pd.DataFrame,
    label_col: str = 'label',
    subject_col: str = 'subject',
    min_label_purity: Optional[float] = None,
    min_non_nan_ratio: Optional[float] = None,
) -> pd.DataFrame:
    """Build windowed features with optional thresholds.

    Purity is computed as majority_label_count / window_size (not only among labeled rows).
    Thresholds default to module constants but can be overridden via args or env vars:
      - MHEALTH_MIN_LABEL_PURITY
      - MHEALTH_MIN_NON_NAN_RATIO
    """
    thr_purity = (
        MIN_LABEL_PURITY if min_label_purity is None else float(min_label_purity)
    )
    # allow env override when not explicitly provided
    if min_label_purity is None:
        thr_purity = float(os.getenv('MHEALTH_MIN_LABEL_PURITY', thr_purity))
    thr_non_nan = (
        MIN_NON_NAN_RATIO if min_non_nan_ratio is None else float(min_non_nan_ratio)
    )
    if min_non_nan_ratio is None:
        thr_non_nan = float(os.getenv('MHEALTH_MIN_NON_NAN_RATIO', thr_non_nan))
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c not in {label_col, subject_col}]
    rows = []
    for subj, g in df.groupby(subject_col, sort=True):
        g = g.reset_index(drop=True)
        for s, e in _window_iter(len(g), WINDOW_SIZE, STEP_SIZE):
            w = g.iloc[s:e]
            # Compute label majority and purity when labels present
            label = None
            purity = None
            if label_col in w.columns:
                lbl_series = w[label_col].dropna()
                if not lbl_series.empty:
                    counts_abs = lbl_series.value_counts()
                    label = int(counts_abs.idxmax())
                    # purity over full window size (including rows with NaN label as mismatches)
                    purity = float(counts_abs.max() / len(w))
            # Skip excluded labels or low purity (only when label available)
            if label is not None:
                if label in EXCLUDE_LABELS:
                    continue
                if purity is not None and purity < thr_purity:
                    continue
            # Missing ratio filter across numeric cols
            non_nan_ratio = float(1.0 - np.mean(np.isnan(w[numeric_cols]).to_numpy())) if len(numeric_cols) else 1.0
            if non_nan_ratio < thr_non_nan:
                continue
            # fill remaining NaNs with window medians for stability
            w_filled = w.copy()
            w_filled[numeric_cols] = w_filled[numeric_cols].apply(lambda col: col.fillna(col.median()))
            feat = {'subject': int(subj), 'start_idx': s, 'end_idx': e}
            if label is not None:
                feat[label_col] = label
                feat['purity'] = purity
            for col in numeric_cols:
                stats = _feat_stats(w_filled[col].values)
                for k in STAT_KEYS:
                    feat[f'{col}__{k}'] = stats[k]
            rows.append(feat)
    return pd.DataFrame(rows)

def predict_from_log(
    path,
    model,
    scaler,
    feature_names: Optional[List[str]] = None,
    min_label_purity: Optional[float] = None,
    min_non_nan_ratio: Optional[float] = None,
):
    path = Path(path)
    colnames = infer_columns(path)
    df = load_log(path, colnames)
    # construir ventanas y features
    feat_df = build_window_features(
        df,
        label_col='label',
        subject_col='subject',
        min_label_purity=min_label_purity,
        min_non_nan_ratio=min_non_nan_ratio,
    )
    if feat_df.empty:
        return {"predictions": [], "proba": None, "timeline": []}
    # Selección/orden de columnas: usar feature_names del backend si están disponibles
    all_feature_cols = [c for c in feat_df.columns if c not in {'subject','start_idx','end_idx','purity','label'}]
    if feature_names:
        # Reindexar a la lista dada; columnas faltantes se rellenan con 0
        X = feat_df.reindex(columns=feature_names, fill_value=0)
    else:
        X = feat_df[all_feature_cols]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)
    # Build timeline with seconds
    starts = (feat_df['start_idx'] / SAMPLING_HZ).astype(float).tolist()
    ends = (feat_df['end_idx'] / SAMPLING_HZ).astype(float).tolist()
    # per-window purity if computed during feature building
    purity_list = feat_df['purity'].tolist() if 'purity' in feat_df.columns else None
    # per-window confidence (max probability among classes)
    conf_list = None
    if proba is not None:
        conf_list = np.max(proba, axis=1).tolist()
    timeline = []
    for i, (lbl, s, e) in enumerate(zip(y_pred.tolist(), starts, ends)):
        item = {"label": int(lbl), "start_sec": float(s), "end_sec": float(e)}
        if conf_list is not None and i < len(conf_list):
            item["confidence"] = float(conf_list[i])
        if purity_list is not None and i < len(purity_list):
            pur = purity_list[i]
            if pur is not None and not (isinstance(pur, float) and np.isnan(pur)):
                item["purity"] = float(pur)
        timeline.append(item)

    # Build non-overlapping, contiguous bars using successive window starts
    last_end = ends[-1] if len(ends) else 0.0
    timeline_bars = []
    for i in range(len(starts)):
        seg_start = starts[i]
        seg_end = starts[i+1] if i+1 < len(starts) else last_end
        if seg_end < seg_start:
            seg_end = seg_start
        bar = {"label": int(y_pred[i]), "start_sec": float(seg_start), "end_sec": float(seg_end)}
        if conf_list is not None and i < len(conf_list):
            bar["confidence"] = float(conf_list[i])
        if purity_list is not None and i < len(purity_list):
            pur = purity_list[i]
            if pur is not None and not (isinstance(pur, float) and np.isnan(pur)):
                bar["purity"] = float(pur)
        timeline_bars.append(bar)
    return {
        "predictions": y_pred.tolist(),
        "proba": proba.tolist() if proba is not None else None,
        "timeline": timeline,
        "timeline_bars": timeline_bars,
    }
