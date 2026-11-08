"""Validate recent hourly snapshots and prepare input in training node order."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data/collected/tomtom_flow_final64.csv"
OUTPUT_PATH = ROOT / "data/collected/latest_12h_final64.csv"


def prepare_window(df, stations, now=None):
    df = df.copy()
    df["obs_time_utc"] = pd.to_datetime(df.obs_time_utc, utc=True, format="mixed")
    df = df.drop_duplicates(["obs_time_utc", "station_id"], keep="last")
    end = df.obs_time_utc.max()
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if pd.isna(end) or not pd.Timedelta(0) <= now - end < pd.Timedelta(hours=2):
        raise RuntimeError("Latest snapshot is stale or in the future.")
    hours = pd.date_range(end=end, periods=12, freq="h")
    window = df[df.obs_time_utc.isin(hours)]
    if len(stations) != 64 or len(set(stations)) != 64:
        raise ValueError("Expected exactly 64 training stations.")
    for hour in hours:
        if set(window.loc[window.obs_time_utc == hour, "station_id"]) != set(stations):
            raise RuntimeError(
                f"Missing or mismatched corridor observations at {hour}. Need 12 consecutive complete hours."
            )
    index = pd.MultiIndex.from_product([hours, stations], names=["obs_time_utc", "station_id"])
    window = window.set_index(["obs_time_utc", "station_id"]).loc[index].reset_index()
    values = window.current_speed.to_numpy(dtype=np.float32).reshape(12, 64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("Speeds must be finite and nonnegative.")
    return window, values


def build_latest_12h():
    nodes = pd.read_csv(ROOT / "outputs/final_64/fixed_graph/fixed_graph_64_nodes.csv").sort_values(
        "node_index"
    )
    stations = nodes.corridor_file.str.removesuffix(".csv").tolist()
    window, values = prepare_window(pd.read_csv(INPUT_PATH), stations)
    scaler = np.load(ROOT / "data/final_64/stgnn_scaler.npz")
    mean, scale = float(scaler["mean"].reshape(-1)[0]), float(scaler["scale"].reshape(-1)[0])
    x = ((values - mean) / scale)[None, :, :, None].astype(np.float32)
    window.to_csv(OUTPUT_PATH, index=False)
    np.savez(
        ROOT / "data/collected/live_input_final64.npz",
        X=x,
        station_ids=np.asarray(stations),
        forecast_origin_utc=str(window.obs_time_utc.max()),
    )
    print(f"READY: input {x.shape}, origin {window.obs_time_utc.max()}")
    return x, stations, window.obs_time_utc.max()


if __name__ == "__main__":
    build_latest_12h()
