"""ML helpers for the production forecasting web workflow."""

from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

WELL_COL = "NPD_WELL_BORE_NAME"
DATE_COL = "DATEPRD"
CHOKE_COL = "AVG Choke size"

PREPROCESS_RULES = {
    "oil_positive": ("BORE_OIL_VOL", lambda s: s > 0),
    "flow_production": ("FLOW_KIND", lambda s: s == "production"),
    "well_op": ("WELL_TYPE", lambda s: s == "OP"),
}

ALGORITHMS = {
    "decisiontree",
    "lightgbm",
    "sgd",
    "svm",
    "xgboost",
    "gradientboosting",
    "ridgeregressor",
    "lassoregressor",
    "extratrees",
    "randomforest",
}

OPT_METHODS = ("SLSQP", "Powell", "COBYLA", "Nelder-Mead", "BFGS", "L-BFGS-B", "TNC", "trust-constr")

BOUNDED_MINIMIZE_METHODS = frozenset(
    {"SLSQP", "L-BFGS-B", "TNC", "Powell", "trust-constr"}
)


def well_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or WELL_COL not in df.columns:
        return []
    vc = df[WELL_COL].value_counts()
    return [{"well": str(k), "count": int(v)} for k, v in vc.items()]


def apply_preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply notebook filters; return (filtered_df, list of skipped rule names)."""
    out = df.copy()
    skipped: list[str] = []
    for name, (col, pred) in PREPROCESS_RULES.items():
        if col not in out.columns:
            skipped.append(name)
            continue
        before = len(out)
        if col == "FLOW_KIND":
            out = out[out[col].astype(str).str.lower() == "production"]
        elif col == "WELL_TYPE":
            out = out[out[col].astype(str) == "OP"]
        else:
            out = out[pred(out[col])]
        if len(out) == before and before > 0:
            pass
    out = out.reset_index(drop=True)
    return out, skipped


def build_regressor(name: str, random_state: int = 42):
    n = name.lower().strip()
    if n not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}")
    if n == "decisiontree":
        return DecisionTreeRegressor(random_state=random_state, max_depth=12)
    if n == "randomforest":
        return RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1, max_depth=12
        )
    if n == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1, max_depth=12
        )
    if n == "gradientboosting":
        return GradientBoostingRegressor(random_state=random_state, max_depth=5)
    if n == "ridgeregressor":
        return Ridge(alpha=1.0)
    if n == "lassoregressor":
        return Lasso(random_state=random_state, max_iter=10000)
    if n == "sgd":
        return SGDRegressor(random_state=random_state, max_iter=10000, tol=1e-3)
    if n == "svm":
        return SVR(kernel="rbf", C=1.0, epsilon=0.1)
    if n == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is not installed")
        return xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )
    if n == "lightgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed")
        return lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unhandled algorithm: {name}")


def _minimize_with_method(fun, x0, bounds, method: str):
    m = method.strip()
    if m == "BFGS":
        m = "L-BFGS-B"
    kwargs = {"fun": fun, "x0": x0, "method": m}
    if m in BOUNDED_MINIMIZE_METHODS:
        kwargs["bounds"] = bounds
    res = minimize(**kwargs)
    x = np.asarray(res.x, dtype=float).ravel()
    if m not in BOUNDED_MINIMIZE_METHODS:
        lo, hi = bounds[0]
        x[0] = float(np.clip(x[0], lo, hi))
    return res, x


def run_choke_optimization(
    model_oil,
    model_water,
    X_subset: pd.DataFrame,
    meta_subset: pd.DataFrame,
    feature_cols: list[str],
    target_oil: str,
    target_water: str,
    method: str,
    choke_col: str = CHOKE_COL,
    bounds: tuple[float, float] = (0.1, 1.0),
) -> list[dict[str, Any]]:
    if choke_col not in feature_cols:
        raise ValueError(
            f"Kolom choke '{choke_col}' harus ada di fitur agar optimasi choke berjalan."
        )
    b = [(bounds[0], bounds[1])]
    rows_out: list[dict[str, Any]] = []

    for i in range(len(X_subset)):
        row = X_subset.iloc[i]
        tanggal = meta_subset.iloc[i].get(DATE_COL, pd.NaT)
        choke_aktual = float(row[choke_col])
        x0 = np.array([choke_aktual], dtype=float)

        def objective_oil(choke_arr):
            sim = row.copy()
            sim[choke_col] = float(choke_arr[0])
            arr = sim[feature_cols].values.astype(np.float64).reshape(1, -1)
            pred = float(model_oil.predict(arr)[0])
            return -pred

        res, x_opt = _minimize_with_method(objective_oil, x0, b, method)
        choke_rekom = float(x_opt[0])

        produksi_sebelum = float(
            model_oil.predict(
                row[feature_cols].values.astype(np.float64).reshape(1, -1)
            )[0]
        )
        produksi_maksimal = float(-res.fun)

        sim_rekom = row.copy()
        sim_rekom[choke_col] = choke_rekom
        arr_rekom = sim_rekom[feature_cols].values.astype(np.float64).reshape(1, -1)
        water_pred_rekom = float(model_water.predict(arr_rekom)[0])
        water_pred_actual_choke = float(
            model_water.predict(
                row[feature_cols].values.astype(np.float64).reshape(1, -1)
            )[0]
        )

        oil_actual = float(meta_subset.iloc[i][target_oil])
        water_actual = float(meta_subset.iloc[i][target_water])

        rows_out.append(
            {
                "DATEPRD": pd.Timestamp(tanggal).isoformat()
                if pd.notna(tanggal)
                else None,
                "Choke_Aktual": choke_aktual,
                "Choke_Rekomendasi": choke_rekom,
                "Oil_Pred_ActualChoke": produksi_sebelum,
                "Oil_Pred_OptimalChoke": produksi_maksimal,
                "Oil_Actual": oil_actual,
                "Water_Pred_ActualChoke": water_pred_actual_choke,
                "Water_Pred_OptimalChoke": water_pred_rekom,
                "Water_Actual": water_actual,
            }
        )
    return rows_out


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    df_raw: pd.DataFrame | None = None
    df_filtered: pd.DataFrame | None = None
    df_well: pd.DataFrame | None = None
    well_name: str | None = None
    feature_cols: list[str] = field(default_factory=list)
    target_oil: str = "BORE_OIL_VOL"
    target_water: str = "BORE_WAT_VOL"
    test_size: float = 0.2
    random_state: int = 42
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_oil_train: pd.Series | None = None
    y_oil_test: pd.Series | None = None
    y_water_train: pd.Series | None = None
    y_water_test: pd.Series | None = None
    test_meta: pd.DataFrame | None = None
    algorithm: str | None = None
    model_oil: Any = None
    model_water: Any = None
    cv_folds: int = 5
    cv_result_oil: dict | None = None
    cv_result_water: dict | None = None
    optimization_rows: list[dict] | None = None
    preprocess_skipped: list[str] = field(default_factory=list)
    upload_filename: str | None = None


class SessionStore:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.base_dir / f"{sid}.pkl"

    def load(self, sid: str) -> SessionState | None:
        p = self._path(sid)
        if not p.is_file():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def save(self, state: SessionState) -> None:
        with open(self._path(state.session_id), "wb") as f:
            pickle.dump(state, f)


def split_aligned(
    df_well: pd.DataFrame,
    feature_cols: list[str],
    target_oil: str,
    target_water: str,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    meta_cols = [c for c in [DATE_COL, WELL_COL] if c in df_well.columns]
    need = feature_cols + [target_oil, target_water] + meta_cols
    missing = [c for c in need if c not in df_well.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    X = df_well[feature_cols].copy()
    y_oil = df_well[target_oil].copy()
    y_water = df_well[target_water].copy()
    meta = df_well[meta_cols + [target_oil, target_water]].copy()
    if CHOKE_COL in df_well.columns and CHOKE_COL not in meta.columns:
        meta[CHOKE_COL] = df_well[CHOKE_COL].values

    idx = np.arange(len(X))
    train_i, test_i = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    return {
        "X_train": X.iloc[train_i].reset_index(drop=True),
        "X_test": X.iloc[test_i].reset_index(drop=True),
        "y_oil_train": y_oil.iloc[train_i].reset_index(drop=True),
        "y_oil_test": y_oil.iloc[test_i].reset_index(drop=True),
        "y_water_train": y_water.iloc[train_i].reset_index(drop=True),
        "y_water_test": y_water.iloc[test_i].reset_index(drop=True),
        "test_meta": meta.iloc[test_i].reset_index(drop=True),
    }


def test_date_bounds(
    test_meta: pd.DataFrame | None, date_col: str = DATE_COL
) -> tuple[str | None, str | None]:
    if test_meta is None or date_col not in test_meta.columns:
        return None, None
    s = pd.to_datetime(test_meta[date_col], errors="coerce").dropna()
    if s.empty:
        return None, None
    return s.min().date().isoformat(), s.max().date().isoformat()


def select_optimization_subset(
    X_test: pd.DataFrame,
    test_meta: pd.DataFrame,
    n_days: int,
    start_date: str | None,
    end_date: str | None,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pilih baris test untuk optimasi.

    - Tanpa rentang tanggal: ambil ``n_days`` baris pertama (perilaku lama / notebook).
    - Dengan rentang: filter ``date_col`` di [start, end], urut naik tanggal, lalu ambil
      paling banyak ``n_days`` baris.
    """
    if len(X_test) != len(test_meta):
        raise ValueError("X_test dan test_meta tidak sejajar.")
    n_days = int(n_days)
    if n_days < 1:
        raise ValueError("Jumlah hari minimal 1.")

    s_raw = (start_date or "").strip() if start_date is not None else ""
    e_raw = (end_date or "").strip() if end_date is not None else ""
    has_s = bool(s_raw)
    has_e = bool(e_raw)

    if has_s ^ has_e:
        raise ValueError(
            "Isi tanggal mulai dan tanggal akhir keduanya, atau kosongkan keduanya."
        )

    if not has_s:
        n = min(n_days, len(X_test))
        Xs = X_test.iloc[:n].reset_index(drop=True)
        ms = test_meta.iloc[:n].reset_index(drop=True)
        return Xs, ms, {
            "mode": "head",
            "n_rows": n,
            "date_filter": None,
        }

    if date_col not in test_meta.columns:
        raise ValueError(
            f"Kolom '{date_col}' tidak ada di metadata test; tidak bisa filter tanggal."
        )

    d_start = pd.Timestamp(s_raw).normalize()
    d_end = pd.Timestamp(e_raw).normalize()
    if d_start > d_end:
        raise ValueError("Tanggal mulai tidak boleh setelah tanggal akhir.")

    dates = pd.to_datetime(test_meta[date_col], errors="coerce")
    norm = dates.dt.normalize()
    mask = dates.notna() & (norm >= d_start) & (norm <= d_end)
    idx = np.flatnonzero(mask.to_numpy())
    if idx.size == 0:
        raise ValueError("Tidak ada baris test dalam rentang tanggal yang dipilih.")

    sub_X = X_test.iloc[idx].reset_index(drop=True)
    sub_m = test_meta.iloc[idx].reset_index(drop=True)
    sub_d = dates.iloc[idx].reset_index(drop=True)
    order = np.argsort(sub_d.to_numpy(), kind="mergesort")
    sub_X = sub_X.iloc[order].reset_index(drop=True)
    sub_m = sub_m.iloc[order].reset_index(drop=True)

    n_take = min(n_days, len(sub_X))
    sub_X = sub_X.iloc[:n_take].reset_index(drop=True)
    sub_m = sub_m.iloc[:n_take].reset_index(drop=True)

    return sub_X, sub_m, {
        "mode": "date_range",
        "n_rows": n_take,
        "date_filter": {
            "start": d_start.date().isoformat(),
            "end": d_end.date().isoformat(),
        },
        "matched_in_range": int(idx.size),
    }
