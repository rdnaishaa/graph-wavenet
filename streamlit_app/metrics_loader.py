from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = ROOT / "outputs" / "final_64" / "graphwavenet_audited"

SUMMARY_FILE = RESULT_DIR / "graphwavenet_multiseed_summary.csv"

ALL_RESULTS_FILE = RESULT_DIR / "graphwavenet_multiseed_all_results.csv"


# ============================================================
# DISPLAY NAMES
# ============================================================

MODEL_DISPLAY_NAMES = {
    "fixed": "Fixed Graph WaveNet",
    "adaptive": "Adaptive Graph WaveNet",
    "hybrid": "Hybrid Graph WaveNet",
}


HORIZON_ORDER = {
    "t+1": 1,
    "t+3": 3,
    "t+6": 6,
}


# ============================================================
# LOAD FILES
# ============================================================


def load_multiseed_summary():

    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {SUMMARY_FILE}")

    df = pd.read_csv(SUMMARY_FILE)

    df["mode"] = df["mode"].astype(str).str.lower()

    df["model_name"] = df["mode"].map(MODEL_DISPLAY_NAMES)

    df["horizon_order"] = df["horizon"].map(HORIZON_ORDER)

    return df.sort_values(
        [
            "horizon_order",
            "mode",
        ]
    ).reset_index(drop=True)


def load_multiseed_results():

    if not ALL_RESULTS_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {ALL_RESULTS_FILE}")

    return pd.read_csv(ALL_RESULTS_FILE)


# ============================================================
# PERFORMANCE TABLE
# ============================================================


def get_performance_table():

    df = load_multiseed_summary()

    result = df[
        [
            "model_name",
            "horizon",
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "RMSE_std",
        ]
    ].copy()

    result = result.rename(
        columns={
            "model_name": "Model",
            "horizon": "Horizon",
            "MAE_mean": "MAE Mean",
            "MAE_std": "MAE Std",
            "RMSE_mean": "RMSE Mean",
            "RMSE_std": "RMSE Std",
        }
    )

    return result


# ============================================================
# FORMATTED TABLE
# ============================================================


def get_formatted_performance_table():

    df = load_multiseed_summary()

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "Model": row["model_name"],
                "Horizon": row["horizon"],
                "MAE": (f"{row['MAE_mean']:.3f} ± {row['MAE_std']:.3f}"),
                "RMSE": (f"{row['RMSE_mean']:.3f} ± {row['RMSE_std']:.3f}"),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MODEL SUMMARY
# ============================================================


def get_model_average_metrics():

    df = load_multiseed_summary()

    result = df.groupby(
        [
            "mode",
            "model_name",
        ],
        as_index=False,
    ).agg(
        MAE_mean=("MAE_mean", "mean"),
        RMSE_mean=("RMSE_mean", "mean"),
    )

    return result


# ============================================================
# RECOMMENDED MODEL
# ============================================================


def get_recommended_model():

    df = get_model_average_metrics()

    df["combined_rank"] = df["MAE_mean"].rank(method="min") + df["RMSE_mean"].rank(method="min")

    best = df.sort_values(
        [
            "combined_rank",
            "MAE_mean",
            "RMSE_mean",
        ]
    ).iloc[0]

    return {
        "mode": best["mode"],
        "model_name": best["model_name"],
        "average_mae": float(best["MAE_mean"]),
        "average_rmse": float(best["RMSE_mean"]),
    }


# ============================================================
# CHECK IF MODEL LOWEST AT EVERY HORIZON
# ============================================================


def get_horizon_winners():

    df = load_multiseed_summary()

    rows = []

    for horizon in [
        "t+1",
        "t+3",
        "t+6",
    ]:
        subset = df[df["horizon"] == horizon]

        mae_row = subset.sort_values("MAE_mean").iloc[0]

        rmse_row = subset.sort_values("RMSE_mean").iloc[0]

        rows.append(
            {
                "horizon": horizon,
                "lowest_mae_model": mae_row["model_name"],
                "lowest_mae": float(mae_row["MAE_mean"]),
                "lowest_rmse_model": rmse_row["model_name"],
                "lowest_rmse": float(rmse_row["RMSE_mean"]),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STATUS
# ============================================================


def get_metrics_status():

    return {
        "summary_exists": SUMMARY_FILE.exists(),
        "all_results_exists": ALL_RESULTS_FILE.exists(),
        "summary_path": str(SUMMARY_FILE),
        "all_results_path": str(ALL_RESULTS_FILE),
    }
