"""Flask API for production ML workflow (upload → preprocess → train → CV → optimize)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ml_pipeline import (
    ALGORITHMS,
    OPT_METHODS,
    SessionState,
    SessionStore,
    apply_preprocess,
    build_regressor,
    run_choke_optimization,
    select_optimization_subset,
    split_aligned,
    test_date_bounds,
    well_summary,
)

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
SESSION_DIR = BACKEND_DIR / "sessions"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".xlsx"}

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="",
)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
CORS(app)

store = SessionStore(SESSION_DIR)


def _json_err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def _load_session():
    sid = request.headers.get("X-Session-Id") or request.args.get("session_id")
    if not sid:
        return None, _json_err("Header X-Session-Id atau query session_id wajib.", 400)
    st = store.load(sid)
    if not st:
        return None, _json_err("Sesi tidak ditemukan. Buat sesi baru atau unggah ulang.", 404)
    return st, None


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/api/session")
def create_session():
    st = SessionState()
    store.save(st)
    return jsonify({"ok": True, "session_id": st.session_id})


@app.post("/api/upload")
def upload():
    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    if "file" not in request.files:
        return _json_err("Tidak ada berkas 'file'.")
    f = request.files["file"]
    if not f.filename:
        return _json_err("Nama berkas kosong.")
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return _json_err("Hanya file Excel (.xlsx / .xls).")
    name = secure_filename(f.filename)
    path = UPLOAD_DIR / f"{st.session_id}_{name}"
    f.save(path)
    st.upload_filename = name
    try:
        import pandas as pd

        st.df_raw = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        return _json_err(f"Gagal membaca Excel: {e}")
    st.df_filtered = None
    st.df_well = None
    store.save(st)
    summary = well_summary(st.df_raw)
    return jsonify(
        {
            "ok": True,
            "filename": name,
            "columns": list(st.df_raw.columns.astype(str)),
            "n_rows": len(st.df_raw),
            "n_wells": len(summary),
            "wells": summary,
        }
    )


@app.post("/api/preprocess")
def preprocess():
    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    if st.df_raw is None:
        return _json_err("Unggah data terlebih dahulu.")
    st.df_filtered, st.preprocess_skipped = apply_preprocess(st.df_raw)
    st.df_well = None
    store.save(st)
    summary = well_summary(st.df_filtered)
    return jsonify(
        {
            "ok": True,
            "n_rows": len(st.df_filtered),
            "n_wells": len(summary),
            "wells": summary,
            "skipped_rules_missing_columns": st.preprocess_skipped,
        }
    )


@app.post("/api/select-well")
def select_well():
    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    data = request.get_json(silent=True) or {}
    well = data.get("well_name")
    if not well:
        return _json_err("well_name wajib.")
    base = st.df_filtered if st.df_filtered is not None else st.df_raw
    if base is None:
        return _json_err("Unggah dan praproses data terlebih dahulu.")
    from ml_pipeline import WELL_COL

    if WELL_COL not in base.columns:
        return _json_err(f"Kolom sumur '{WELL_COL}' tidak ada di data.")
    sub = base[base[WELL_COL].astype(str) == str(well)].copy()
    if sub.empty:
        return _json_err(f"Tidak ada data untuk sumur: {well}")
    sub = sub.reset_index(drop=True)
    st.df_well = sub
    st.well_name = str(well)
    store.save(st)
    return jsonify(
        {
            "ok": True,
            "well_name": st.well_name,
            "n_rows": len(sub),
            "columns": list(sub.columns.astype(str)),
        }
    )


@app.post("/api/features")
def set_features():
    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    data = request.get_json(silent=True) or {}
    features = data.get("feature_columns")
    t_oil = data.get("target_oil", "BORE_OIL_VOL")
    t_water = data.get("target_water", "BORE_WAT_VOL")
    if not features or not isinstance(features, list):
        return _json_err("feature_columns (array) wajib.")
    if st.df_well is None:
        return _json_err("Pilih sumur terlebih dahulu.")
    cols = list(st.df_well.columns.astype(str))
    for c in features + [t_oil, t_water]:
        if c not in cols:
            return _json_err(f"Kolom tidak ada di data sumur: {c}")
    st.feature_cols = [str(x) for x in features]
    st.target_oil = str(t_oil)
    st.target_water = str(t_water)
    store.save(st)
    return jsonify(
        {
            "ok": True,
            "feature_columns": st.feature_cols,
            "target_oil": st.target_oil,
            "target_water": st.target_water,
        }
    )


@app.post("/api/split")
def split_data():
    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    data = request.get_json(silent=True) or {}
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))
    if not 0 < test_size < 1:
        return _json_err("test_size harus antara 0 dan 1.")
    if st.df_well is None or not st.feature_cols:
        return _json_err("Pilih sumur dan fitur terlebih dahulu.")
    try:
        parts = split_aligned(
            st.df_well,
            st.feature_cols,
            st.target_oil,
            st.target_water,
            test_size,
            random_state,
        )
    except ValueError as e:
        return _json_err(str(e))
    st.test_size = test_size
    st.random_state = random_state
    st.X_train = parts["X_train"]
    st.X_test = parts["X_test"]
    st.y_oil_train = parts["y_oil_train"]
    st.y_oil_test = parts["y_oil_test"]
    st.y_water_train = parts["y_water_train"]
    st.y_water_test = parts["y_water_test"]
    st.test_meta = parts["test_meta"]
    st.model_oil = None
    st.model_water = None
    st.cv_result_oil = None
    st.cv_result_water = None
    st.optimization_rows = None
    store.save(st)
    td_min, td_max = test_date_bounds(st.test_meta)
    return jsonify(
        {
            "ok": True,
            "train_rows": len(st.X_train),
            "test_rows": len(st.X_test),
            "test_size": test_size,
            "random_state": random_state,
            "test_date_min": td_min,
            "test_date_max": td_max,
        }
    )


@app.post("/api/model")
def set_model():
    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    data = request.get_json(silent=True) or {}
    algo = (data.get("algorithm") or "").lower().strip()
    if algo not in ALGORITHMS:
        return _json_err(
            f"algorithm harus salah satu dari: {', '.join(sorted(ALGORITHMS))}"
        )
    st.algorithm = algo
    store.save(st)
    return jsonify({"ok": True, "algorithm": st.algorithm})


def _cv_payload(scores):
    scores = np.asarray(scores, dtype=float)
    return {
        "scores": [float(x) for x in scores],
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


@app.post("/api/train")
def train():
    from sklearn.model_selection import cross_val_score

    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    if st.X_train is None:
        return _json_err("Split data terlebih dahulu.")
    data = request.get_json(silent=True) or {}
    algo = (data.get("algorithm") or "").lower().strip()
    if algo:
        if algo not in ALGORITHMS:
            return _json_err(
                f"algorithm harus salah satu dari: {', '.join(sorted(ALGORITHMS))}"
            )
        st.algorithm = algo
    elif not st.algorithm:
        return _json_err("Pilih algoritma di dropdown lalu jalankan training.")
    try:
        mo = build_regressor(st.algorithm, st.random_state)
        mw = build_regressor(st.algorithm, st.random_state)
    except Exception as e:
        return _json_err(str(e))
    Xtr = st.X_train.astype(float)
    mo.fit(Xtr, st.y_oil_train)
    mw.fit(Xtr, st.y_water_train)
    st.model_oil = mo
    st.model_water = mw
    st.optimization_rows = None
    folds = max(2, int(st.cv_folds))
    try:
        s_oil = cross_val_score(
            mo, Xtr, st.y_oil_train, cv=folds, scoring="r2", n_jobs=-1
        )
        s_water = cross_val_score(
            mw, Xtr, st.y_water_train, cv=folds, scoring="r2", n_jobs=-1
        )
    except Exception as e:
        store.save(st)
        return _json_err(f"Training ok tetapi CV awal gagal: {e}")
    st.cv_result_oil = _cv_payload(s_oil)
    st.cv_result_water = _cv_payload(s_water)
    store.save(st)
    return jsonify(
        {
            "ok": True,
            "message": "Model oil & water selesai dilatih.",
            "algorithm": st.algorithm,
            "cv_folds_used": folds,
            "cross_validation_oil": st.cv_result_oil,
            "cross_validation_water": st.cv_result_water,
        }
    )


@app.post("/api/cross-validate")
def cross_validate_only():
    from sklearn.model_selection import cross_val_score

    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    if st.model_oil is None or st.X_train is None:
        return _json_err("Latih model terlebih dahulu.")
    data = request.get_json(silent=True) or {}
    folds = int(data.get("cv", st.cv_folds))
    if folds < 2:
        return _json_err("cv minimal 2.")
    st.cv_folds = folds
    Xtr = st.X_train.astype(float)
    try:
        s_oil = cross_val_score(
            st.model_oil, Xtr, st.y_oil_train, cv=folds, scoring="r2", n_jobs=-1
        )
        s_water = cross_val_score(
            st.model_water, Xtr, st.y_water_train, cv=folds, scoring="r2", n_jobs=-1
        )
    except Exception as e:
        return _json_err(str(e))
    st.cv_result_oil = _cv_payload(s_oil)
    st.cv_result_water = _cv_payload(s_water)
    store.save(st)
    return jsonify(
        {
            "ok": True,
            "cross_validation_oil": st.cv_result_oil,
            "cross_validation_water": st.cv_result_water,
        }
    )


@app.post("/api/optimize")
def optimize():
    from ml_pipeline import CHOKE_COL

    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    if st.model_oil is None or st.X_test is None or st.test_meta is None:
        return _json_err("Split data dan training harus selesai.")
    if CHOKE_COL not in st.feature_cols:
        return _json_err(
            f"Untuk optimasi choke, kolom '{CHOKE_COL}' harus dipilih sebagai fitur."
        )
    data = request.get_json(silent=True) or {}
    n_days = int(data.get("n_days", 30))
    method = (data.get("method") or "SLSQP").strip()
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    if method not in OPT_METHODS:
        return _json_err(f"method harus salah satu dari: {', '.join(OPT_METHODS)}")
    try:
        X_sub, meta_sub, subset_info = select_optimization_subset(
            st.X_test,
            st.test_meta,
            n_days,
            start_date,
            end_date,
        )
    except ValueError as e:
        return _json_err(str(e))
    if len(X_sub) < 1:
        return _json_err("Tidak ada baris test untuk disimulasikan.")
    try:
        rows = run_choke_optimization(
            st.model_oil,
            st.model_water,
            X_sub,
            meta_sub,
            st.feature_cols,
            st.target_oil,
            st.target_water,
            method=method,
            choke_col=CHOKE_COL,
        )
    except ValueError as e:
        return _json_err(str(e))
    except Exception as e:
        return _json_err(f"Optimasi gagal: {e}")
    st.optimization_rows = rows
    store.save(st)
    return jsonify(
        {
            "ok": True,
            "n_rows": len(rows),
            "method": method,
            "subset": subset_info,
            "rows": rows,
        }
    )


@app.get("/api/visualization")
def visualization():
    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    if not st.optimization_rows:
        return _json_err("Jalankan optimasi terlebih dahulu.", 400)
    return jsonify({"ok": True, "series": st.optimization_rows})


@app.get("/api/state")
def state():
    st, err = _load_session()
    if err is not None:
        return err
    assert st is not None
    td_min, td_max = (None, None)
    if st.test_meta is not None:
        td_min, td_max = test_date_bounds(st.test_meta)
    return jsonify(
        {
            "ok": True,
            "upload_filename": st.upload_filename,
            "well_name": st.well_name,
            "has_raw": st.df_raw is not None,
            "has_filtered": st.df_filtered is not None,
            "has_well": st.df_well is not None,
            "feature_columns": st.feature_cols,
            "target_oil": st.target_oil,
            "target_water": st.target_water,
            "algorithm": st.algorithm,
            "test_size": st.test_size,
            "cv_folds": st.cv_folds,
            "has_split": st.X_train is not None,
            "has_models": st.model_oil is not None,
            "has_optimization": bool(st.optimization_rows),
            "test_date_min": td_min,
            "test_date_max": td_max,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
