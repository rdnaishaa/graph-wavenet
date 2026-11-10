from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs/final_64"
GWN = RESULT / "graphwavenet_audited/graphwavenet_multiseed_summary.csv"
BASELINE = RESULT / "gclstm_baseline/gclstm_fair_multiseed_summary.csv"

COLORS = {
    "Fixed GWN": "#8096ba",
    "Adaptive GWN": "#43a6cc",
    "Hybrid GWN": "#2879ed",
    "GC-LSTM-style": "#ed9860",
}

HORIZONS = ["t+1", "t+3", "t+6"]


def matching_column(df, *names):
    lookup = {str(col).lower().replace("_", "").replace(" ", ""): col for col in df.columns}
    for name in names:
        key = name.lower().replace("_", "").replace(" ", "")
        if key in lookup:
            return lookup[key]
    return None


def normalize_horizon(value):
    text = str(value).strip().lower().replace(" ", "")
    aliases = {
        "t+1": "t+1",
        "+1": "t+1",
        "1": "t+1",
        "t1": "t+1",
        "h1": "t+1",
        "t+3": "t+3",
        "+3": "t+3",
        "3": "t+3",
        "t3": "t+3",
        "h3": "t+3",
        "t+6": "t+6",
        "+6": "t+6",
        "6": "t+6",
        "t6": "t+6",
        "h6": "t+6",
    }
    return aliases.get(text)


def normalize_table(df, default_model=None):
    horizon_col = matching_column(df, "horizon", "forecast_horizon", "step")
    model_col = matching_column(df, "mode", "model_name", "model", "strategy")

    required = {
        "MAE_mean": matching_column(df, "MAE_mean", "mean_mae", "mae_avg"),
        "MAE_std": matching_column(df, "MAE_std", "std_mae", "mae_sd"),
        "RMSE_mean": matching_column(df, "RMSE_mean", "mean_rmse", "rmse_avg"),
        "RMSE_std": matching_column(df, "RMSE_std", "std_rmse", "rmse_sd"),
    }

    missing = [key for key, col in required.items() if col is None]
    if horizon_col is None or missing:
        raise ValueError(
            "Kolom ringkasan belum cocok. Kolom tersedia: " + ", ".join(map(str, df.columns))
        )

    if model_col is None and default_model is None:
        raise ValueError("Kolom model tidak ditemukan.")

    result = pd.DataFrame()
    result["Horizon"] = df[horizon_col].map(normalize_horizon)

    if default_model:
        result["Model"] = default_model
    else:

        def model_name(raw):
            value = str(raw).lower()
            if "hybrid" in value:
                return "Hybrid GWN"
            if "adaptive" in value:
                return "Adaptive GWN"
            if "fixed" in value:
                return "Fixed GWN"
            if "gclstm" in value or "gc-lstm" in value:
                return "GC-LSTM-style"
            return str(raw)

        result["Model"] = df[model_col].map(model_name)

    for key, col in required.items():
        result[key] = pd.to_numeric(df[col], errors="coerce")

    result = result[result["Horizon"].isin(HORIZONS)].dropna(subset=["MAE_mean", "RMSE_mean"])

    return result


@st.cache_data
def read_evaluation():
    if not GWN.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {GWN}")

    gwn = normalize_table(pd.read_csv(GWN))

    if gwn.duplicated(["Model", "Horizon"]).any():
        raise ValueError("Ada duplikasi model-horizon dalam GWN.")

    note = None
    if BASELINE.exists():
        raw_baseline = pd.read_csv(BASELINE)
        try:
            baseline = normalize_table(
                raw_baseline,
                default_model="GC-LSTM-style",
            )
            if len(baseline) != 3 or baseline["Horizon"].nunique() != 3:
                raise ValueError("Baseline tidak berisi tiga horizon unik.")
            result = pd.concat(
                [gwn, baseline],
                ignore_index=True,
            )
        except ValueError as exc:
            result = gwn
            note = "Baseline belum bisa dibaca otomatis: " + str(exc)
    else:
        result = gwn
        note = "File ringkasan GC-LSTM-style belum ditemukan."

    result["Horizon"] = pd.Categorical(
        result["Horizon"],
        categories=HORIZONS,
        ordered=True,
    )

    return result.sort_values(["Horizon", "Model"]), note
