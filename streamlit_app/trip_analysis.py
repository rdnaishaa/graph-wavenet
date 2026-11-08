from streamlit_app.forecasting import (
    forecast_trip,
)
from streamlit_app.historical_analysis import (
    get_history_statistics,
    get_relative_condition,
)
from streamlit_app.trip_utils import (
    get_trip_nodes,
)

# ============================================================
# TRIP FORECAST ANALYSIS
# ============================================================


def analyze_trip(
    model_name,
    sample_idx,
    origin_name,
    destination_name,
):

    nodes = get_trip_nodes(
        origin_name=origin_name,
        destination_name=destination_name,
    )

    origin_node = nodes["origin_node"]

    destination_node = nodes["destination_node"]

    prediction = forecast_trip(
        model_name=model_name,
        sample_idx=sample_idx,
        origin_node=origin_node,
        destination_node=destination_node,
    )

    origin_history = get_history_statistics(
        sample_idx=sample_idx,
        node_idx=origin_node,
    )

    destination_history = get_history_statistics(
        sample_idx=sample_idx,
        node_idx=destination_node,
    )

    df = prediction["forecast"].copy()

    # ========================================================
    # RELATIVE CONDITIONS
    # ========================================================

    df["origin_condition"] = df["origin_speed"].apply(
        lambda speed: get_relative_condition(
            speed,
            origin_history["average"],
        )
    )

    df["destination_condition"] = df["destination_speed"].apply(
        lambda speed: get_relative_condition(
            speed,
            destination_history["average"],
        )
    )

    # ========================================================
    # RELATIVE SCORE
    # ========================================================

    origin_avg = origin_history["average"]

    destination_avg = destination_history["average"]

    df["origin_relative_score"] = df["origin_speed"] / origin_avg

    df["destination_relative_score"] = df["destination_speed"] / destination_avg

    df["trip_relative_score"] = (
        df["origin_relative_score"] + df["destination_relative_score"]
    ) / 2.0

    # ========================================================
    # BEST RELATIVE HORIZON
    # ========================================================

    best_row = df.sort_values("trip_relative_score", ascending=False).iloc[0]

    return {
        "model": prediction["model"],
        "origin_name": origin_name,
        "destination_name": destination_name,
        "origin_node": origin_node,
        "destination_node": destination_node,
        "origin_history": origin_history,
        "destination_history": destination_history,
        "forecast": df,
        "suggested_horizon": int(best_row["horizon"]),
        "suggested_time": best_row["target_time"],
        "suggested_score": float(best_row["trip_relative_score"]),
        "inference_ms": prediction["inference_ms"],
    }


# ============================================================
# USER-FRIENDLY SUMMARY
# ============================================================


def build_trip_summary(result):

    horizon = result["suggested_horizon"]

    origin = result["origin_name"]

    destination = result["destination_name"]

    return (
        f"Berdasarkan prediksi kondisi relatif "
        f"pada corridor {origin} dan {destination}, "
        f"horizon +{horizon} jam menunjukkan kondisi "
        f"endpoint yang relatif lebih baik dibandingkan "
        f"horizon prediksi lainnya."
    )
