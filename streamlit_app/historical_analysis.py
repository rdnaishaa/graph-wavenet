import numpy as np

from streamlit_app.data_loader import (
    get_historical_series,
)

# ============================================================
# BASIC STATISTICS
# ============================================================


def get_history_statistics(
    sample_idx,
    node_idx,
):

    df = get_historical_series(
        sample_idx=sample_idx,
        node_idx=node_idx,
    )

    speeds = df["speed"].astype(float)

    return {
        "latest": float(speeds.iloc[-1]),
        "average": float(speeds.mean()),
        "minimum": float(speeds.min()),
        "maximum": float(speeds.max()),
        "std": float(speeds.std()),
        "trend": get_speed_trend(speeds),
    }


# ============================================================
# TREND
# ============================================================


def get_speed_trend(speeds):

    speeds = np.asarray(speeds, dtype=float)

    if len(speeds) < 2:
        return "Stabil"

    x = np.arange(len(speeds))

    slope = np.polyfit(x, speeds, 1)[0]

    if slope > 0.5:
        return "Cenderung meningkat"

    if slope < -0.5:
        return "Cenderung menurun"

    return "Relatif stabil"


# ============================================================
# RELATIVE CONDITION
# ============================================================


def get_relative_condition(
    speed,
    historical_average,
):

    if historical_average is None or historical_average <= 0:
        return "Data historis tidak tersedia"

    ratio = float(speed) / float(historical_average)

    if ratio >= 1.10:
        return "Di atas rata-rata historis"

    if ratio >= 0.90:
        return "Sekitar rata-rata historis"

    if ratio >= 0.70:
        return "Di bawah rata-rata historis"

    return "Jauh di bawah rata-rata historis"


# ============================================================
# HISTORY TABLE
# ============================================================


def get_history_table(
    sample_idx,
    node_idx,
):

    df = get_historical_series(
        sample_idx=sample_idx,
        node_idx=node_idx,
    ).copy()

    stats = get_history_statistics(
        sample_idx=sample_idx,
        node_idx=node_idx,
    )

    average = stats["average"]

    df["relative_condition"] = df["speed"].apply(
        lambda value: get_relative_condition(value, average)
    )

    return df
