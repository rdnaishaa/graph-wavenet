"""Collect one current hourly snapshot; never reconstruct missed history."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data/collected/tomtom_flow_final64.csv"
NODES_PATH = ROOT / "outputs/final_64/fixed_graph/fixed_graph_64_nodes.csv"


def load_final_64():
    nodes = pd.read_csv(NODES_PATH).sort_values("node_index")
    nodes["station_id"] = nodes.corridor_file.str.removesuffix(".csv")
    source = ROOT / "corridors.csv"
    if not source.exists():
        source = OUTPUT_PATH
    metadata = pd.read_csv(source)[["station_id", "name", "city", "lat", "lon"]].drop_duplicates(
        "station_id"
    )
    result = nodes.merge(metadata, on="station_id", how="left", validate="one_to_one")
    if len(result) != 64 or result[["lat", "lon", "name", "city"]].isna().any().any():
        raise ValueError("Metadata must contain all 64 training nodes.")
    if not np.allclose(
        result[["lat", "lon"]], result[["latitude", "longitude"]], atol=1e-8, rtol=0
    ):
        raise ValueError("Collector coordinates do not match the training graph.")
    return result


def fetch_flow(lat, lon):
    load_dotenv(ROOT / ".env")
    key = os.getenv("TOMTOM_API_KEY")
    if not key:
        raise RuntimeError("TOMTOM_API_KEY is missing.")
    for attempt in range(3):
        try:
            response = requests.get(
                "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
                params={"point": f"{lat},{lon}", "unit": "KMPH", "key": key},
                timeout=20,
            )
            response.raise_for_status()
            flow = response.json()["flowSegmentData"]
            speed = float(flow["currentSpeed"])
            if not np.isfinite(speed) or speed < 0:
                raise ValueError("Invalid speed")
            return flow
        except (requests.RequestException, ValueError, KeyError):
            if attempt == 2:
                # Do not expose a request URL containing the API key.
                raise RuntimeError("TomTom request failed after 3 attempts.") from None
            time.sleep(1)


def collect_once():
    corridors = load_final_64()
    bucket = pd.Timestamp.now(tz="UTC").floor("h")
    old = pd.read_csv(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame()
    if not old.empty:
        old["obs_time_utc"] = pd.to_datetime(old.obs_time_utc, utc=True, format="mixed")
    done = (
        set(old.loc[(old.obs_time_utc == bucket) & old.current_speed.notna(), "station_id"])
        if not old.empty
        else set()
    )
    rows = []
    failures = []
    for row in corridors.itertuples():
        if row.station_id in done:
            continue
        if pd.Timestamp.now(tz="UTC").floor("h") != bucket:
            break
        try:
            flow = fetch_flow(row.lat, row.lon)
            actual = pd.Timestamp.now(tz="UTC")
            if actual.floor("h") != bucket:
                break
            speed, free = flow["currentSpeed"], flow.get("freeFlowSpeed")
            rows.append(
                dict(
                    obs_time_utc=bucket,
                    collected_at_utc=actual,
                    station_id=row.station_id,
                    name=row.name,
                    city=row.city,
                    lat=row.lat,
                    lon=row.lon,
                    current_speed=speed,
                    free_flow_speed=free,
                    current_travel_time=flow.get("currentTravelTime"),
                    free_flow_travel_time=flow.get("freeFlowTravelTime"),
                    congestion_ratio=1 - speed / free if free else None,
                    road_closure=flow.get("roadClosure"),
                    confidence=flow.get("confidence"),
                )
            )
        except RuntimeError:
            failures.append(row.station_id)
        time.sleep(0.1)
    combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(["obs_time_utc", "station_id"], keep="last").sort_values(
        ["obs_time_utc", "station_id"]
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(OUTPUT_PATH)
    count = len(done) + len(rows)
    print(f"{bucket}: {count}/64 corridors saved; failures={failures}", flush=True)
    if count != 64:
        raise RuntimeError(
            "Incomplete snapshot; successful rows were preserved. Retry within this hour."
        )
    return bucket


if __name__ == "__main__":
    collect_once()
