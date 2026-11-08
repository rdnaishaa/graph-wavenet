from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "final_64"

TRAIN_FILE = DATA_DIR / "forecasting_train_preprocessed.csv"

VAL_FILE = DATA_DIR / "forecasting_val_preprocessed.csv"

TEST_FILE = DATA_DIR / "forecasting_test_preprocessed.csv"


# ============================================================
# LOAD FULL FINAL64 HISTORY
# ============================================================


def load_full_history():

    frames = []

    for split_name, path in [
        ("train", TRAIN_FILE),
        ("validation", VAL_FILE),
        ("test", TEST_FILE),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        df = pd.read_csv(path)

        df["split"] = split_name

        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)

    full_df["obs_time_utc"] = pd.to_datetime(full_df["obs_time_utc"], utc=True, errors="coerce")

    full_df = (
        full_df.dropna(subset=["obs_time_utc"])
        .sort_values(
            [
                "obs_time_utc",
                "corridor_file",
            ]
        )
        .reset_index(drop=True)
    )

    return full_df


# ============================================================
# DATASET PERIOD
# ============================================================


def get_history_period():

    df = load_full_history()

    return {
        "start": df["obs_time_utc"].min(),
        "end": df["obs_time_utc"].max(),
    }


# ============================================================
# ONE CORRIDOR
# ============================================================


def get_corridor_full_history(corridor_file):

    df = load_full_history()

    return df[df["corridor_file"] == corridor_file].copy().reset_index(drop=True)


# ============================================================
# DATE RANGE
# ============================================================


def get_corridor_date_range(
    corridor_file,
    start_date,
    end_date,
):

    df = get_corridor_full_history(corridor_file)

    start_ts = pd.Timestamp(start_date)

    end_ts = pd.Timestamp(end_date)

    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")

    else:
        start_ts = start_ts.tz_convert("UTC")

    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    else:
        end_ts = end_ts.tz_convert("UTC")

    end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    result = df[(df["obs_time_utc"] >= start_ts) & (df["obs_time_utc"] <= end_ts)].copy()

    return result.reset_index(drop=True)


# ============================================================
# HOURLY PATTERN
# ============================================================


def get_hourly_pattern(corridor_file):

    df = get_corridor_full_history(corridor_file).dropna(subset=["current_speed"]).copy()

    df["hour_utc"] = df["obs_time_utc"].dt.hour

    result = (
        df.groupby("hour_utc", as_index=False)
        .agg(
            average_speed=("current_speed", "mean"),
            minimum_speed=("current_speed", "min"),
            maximum_speed=("current_speed", "max"),
            observations=("current_speed", "count"),
        )
        .sort_values("hour_utc")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# DAILY PATTERN
# ============================================================


def get_daily_pattern(corridor_file):

    df = get_corridor_full_history(corridor_file).dropna(subset=["current_speed"]).copy()

    df["date"] = df["obs_time_utc"].dt.date

    result = (
        df.groupby("date", as_index=False)
        .agg(
            average_speed=("current_speed", "mean"),
            minimum_speed=("current_speed", "min"),
            maximum_speed=("current_speed", "max"),
            observations=("current_speed", "count"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================


def get_history_quality(
    corridor_file,
    start_date=None,
    end_date=None,
):

    if start_date is not None and end_date is not None:
        df = get_corridor_date_range(
            corridor_file=corridor_file,
            start_date=start_date,
            end_date=end_date,
        )

    else:
        df = get_corridor_full_history(corridor_file)

    observed = int(df["is_observed"].fillna(False).astype(bool).sum())

    imputed = int(df["was_imputed"].fillna(False).astype(bool).sum())

    return {
        "rows": int(len(df)),
        "observed": observed,
        "imputed": imputed,
    }
