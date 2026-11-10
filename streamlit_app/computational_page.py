from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "final_64"

TRAINING = BASE / "training_cost_benchmark_seed42.csv"
SCALABILITY = BASE / "scalability" / "scalability_summary_seed42.csv"
OVERALL = BASE / "model_comparison" / "final_5seed_overall_comparison.csv"

COLORS = {
    "Fixed GWN": "#8096ba",
    "Adaptive GWN": "#43a6cc",
    "Hybrid GWN": "#2879ed",
    "GC-LSTM-style": "#ed9860",
}

MODEL_ORDER = list(COLORS)
NODE_ORDER = [8, 16, 32, 64]


@st.cache_data
def load_results():
    for file in [TRAINING, SCALABILITY, OVERALL]:
        if not file.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {file}")

    train = pd.read_csv(TRAINING)
    scale = pd.read_csv(SCALABILITY)
    overall = pd.read_csv(OVERALL)

    # Samakan label model untuk visualisasi.
    aliases = {
        "GC-LSTM": "GC-LSTM-style",
        "GC-LSTM Fair": "GC-LSTM-style",
    }

    train["model"] = train["model"].replace(aliases)
    scale["model"] = scale["model"].replace(aliases)
    overall["model_clean"] = overall["model_clean"].replace(aliases)

    required_train = [
        "model",
        "mean_sec_per_epoch",
        "total_training_time_sec",
        "best_epoch_benchmark",
    ]
    required_scale = [
        "model",
        "num_nodes",
        "avg_mae_kmh",
        "avg_rmse_kmh",
        "parameters",
        "mean_sec_per_epoch",
        "inference_ms_per_sample",
    ]
    required_overall = ["model_clean", "avg_mae", "avg_rmse"]

    for df, cols, name in [
        (train, required_train, "Training"),
        (scale, required_scale, "Scalability"),
        (overall, required_overall, "Overall"),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom {name} belum sesuai: {missing}")

    if train["model"].duplicated().any():
        raise ValueError("Model duplikat pada benchmark training.")

    if scale.duplicated(["model", "num_nodes"]).any():
        raise ValueError("Model-node duplikat pada scalability summary.")

    return train, scale, overall
