from time import perf_counter

import numpy as np
import pandas as pd
import tensorflow as tf

from streamlit_app.data_loader import (
    get_corridor_metadata,
    load_scaler,
    load_test_data,
)
from streamlit_app.model_loader import (
    load_graphwavenet_model,
    normalize_model_mode,
)

HORIZONS = [1, 3, 6]


# ============================================================
# INVERSE SCALE
# ============================================================


def inverse_scale_prediction(prediction_std):

    scaler = load_scaler()

    return prediction_std * scaler["scale"] + scaler["mean"]


# ============================================================
# VALIDATE SAMPLE
# ============================================================


def validate_sample_index(sample_idx):

    data = load_test_data()

    num_samples = data["X"].shape[0]

    if sample_idx < 0 or sample_idx >= num_samples:
        raise IndexError(f"sample_idx harus 0 sampai {num_samples - 1}")


# ============================================================
# RUN FULL FORECAST
# ============================================================


def run_forecast(model_name, sample_idx):

    validate_sample_index(sample_idx)

    mode = normalize_model_mode(model_name)

    data = load_test_data()

    X = data["X"][sample_idx : sample_idx + 1].astype(np.float32)

    model = load_graphwavenet_model(mode)

    input_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    # warm-up call
    _ = model(input_tensor, training=False)

    start = perf_counter()

    prediction_std = model(input_tensor, training=False).numpy()

    elapsed_ms = (perf_counter() - start) * 1000.0

    prediction_real = inverse_scale_prediction(prediction_std)

    return {
        "mode": mode,
        "sample_idx": int(sample_idx),
        "prediction_std": prediction_std[0],
        "prediction": prediction_real[0],
        "inference_ms": float(elapsed_ms),
    }


# ============================================================
# ONE CORRIDOR FORECAST
# ============================================================


def forecast_corridor(model_name, sample_idx, node_idx):

    result = run_forecast(model_name=model_name, sample_idx=sample_idx)

    prediction = result["prediction"]

    data = load_test_data()

    rows = []

    for horizon_idx, horizon in enumerate(HORIZONS):
        target_time = pd.Timestamp(data["target_times"][sample_idx, horizon_idx])

        rows.append(
            {
                "horizon": horizon,
                "label": f"+{horizon} jam",
                "target_time": target_time,
                "predicted_speed": float(prediction[horizon_idx, node_idx]),
            }
        )

    return {
        "model": result["mode"],
        "sample_idx": int(sample_idx),
        "node_idx": int(node_idx),
        "inference_ms": result["inference_ms"],
        "forecast": pd.DataFrame(rows),
    }


# ============================================================
# FORECAST BY CORRIDOR NAME
# ============================================================


def forecast_corridor_by_name(model_name, sample_idx, corridor_name):

    metadata = get_corridor_metadata()

    match = metadata[metadata["corridor_name"] == corridor_name]

    if match.empty:
        raise ValueError(f"Corridor tidak ditemukan: {corridor_name}")

    node_idx = int(match.iloc[0]["node_index"])

    result = forecast_corridor(model_name=model_name, sample_idx=sample_idx, node_idx=node_idx)

    result["corridor_name"] = corridor_name

    return result


# ============================================================
# TRIP ORIGIN + DESTINATION FORECAST
# ============================================================


def forecast_trip(model_name, sample_idx, origin_node, destination_node):

    result = run_forecast(model_name=model_name, sample_idx=sample_idx)

    prediction = result["prediction"]

    data = load_test_data()

    rows = []

    for horizon_idx, horizon in enumerate(HORIZONS):
        target_time = pd.Timestamp(data["target_times"][sample_idx, horizon_idx])

        origin_speed = float(prediction[horizon_idx, origin_node])

        destination_speed = float(prediction[horizon_idx, destination_node])

        mean_speed = (origin_speed + destination_speed) / 2.0

        rows.append(
            {
                "horizon": horizon,
                "label": f"+{horizon} jam",
                "target_time": target_time,
                "origin_speed": origin_speed,
                "destination_speed": destination_speed,
                "mean_endpoint_speed": mean_speed,
            }
        )

    return {
        "model": result["mode"],
        "sample_idx": int(sample_idx),
        "origin_node": int(origin_node),
        "destination_node": int(destination_node),
        "inference_ms": result["inference_ms"],
        "forecast": pd.DataFrame(rows),
    }


# ============================================================
# COMPARE ALL MODELS
# ============================================================


def compare_models(sample_idx, node_idx):

    frames = []

    inference_times = {}

    for mode in ["fixed", "adaptive", "hybrid"]:
        result = forecast_corridor(model_name=mode, sample_idx=sample_idx, node_idx=node_idx)

        df = result["forecast"].copy()

        df["model"] = mode.title()

        frames.append(df)

        inference_times[mode] = result["inference_ms"]

    comparison = pd.concat(frames, ignore_index=True)

    return {
        "comparison": comparison,
        "inference_ms": inference_times,
    }
