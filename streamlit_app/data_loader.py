from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# PATHS
# ============================================================

DATA_DIR = ROOT / "data" / "final_64"

TEST_FILE = DATA_DIR / "stgnn_test.npz"

SCALER_FILE = DATA_DIR / "stgnn_scaler.npz"

CORRIDOR_METADATA_FILE = ROOT / "outputs" / "final_64" / "fixed_graph" / "fixed_graph_64_nodes.csv"


# ============================================================
# DATASET
# ============================================================


def load_test_data():

    npz = np.load(TEST_FILE, allow_pickle=True)

    return {
        "X": npz["X"],
        "y": npz["y"],
        "mask": npz["mask"],
        "target_times": npz["target_times"],
        "input_observed_mask": npz["input_observed_mask"],
        "input_imputed_mask": npz["input_imputed_mask"],
    }


# ============================================================
# SCALER
# ============================================================


def load_scaler():

    npz = np.load(SCALER_FILE)

    mean = float(np.asarray(npz["mean"]).reshape(-1)[0])

    scale = float(np.asarray(npz["scale"]).reshape(-1)[0])

    return {
        "mean": mean,
        "scale": scale,
    }


def inverse_scale(
    values,
    mean,
    scale,
):

    return values * scale + mean


def get_real_test_data():

    data = load_test_data()

    scaler = load_scaler()

    X_real = inverse_scale(
        data["X"],
        scaler["mean"],
        scaler["scale"],
    )

    y_real = inverse_scale(
        data["y"],
        scaler["mean"],
        scaler["scale"],
    )

    return {
        **data,
        "X_real": X_real,
        "y_real": y_real,
        "scaler_mean": scaler["mean"],
        "scaler_scale": scaler["scale"],
    }


# ============================================================
# CORRIDOR NAMES
# ============================================================


def clean_corridor_name(corridor_file):

    name = str(corridor_file)

    name = name.replace("tt_", "")

    name = name.replace(".csv", "")

    custom_names = {
        "kelapagading": "Kelapa Gading",
        "kebonjeruk": "Kebon Jeruk",
        "tbsimatupang": "TB Simatupang",
        "daanmogot": "Daan Mogot",
        "bsd": "BSD",
        "gatsu": "Gatot Subroto",
    }

    if name in custom_names:
        return custom_names[name]

    return name.replace("_", " ").title()


# ============================================================
# CORRIDOR METADATA
# ============================================================


def get_corridor_metadata():

    df = pd.read_csv(CORRIDOR_METADATA_FILE)

    df = df.sort_values("node_index").reset_index(drop=True)

    df["corridor_name"] = df["corridor_file"].apply(clean_corridor_name)

    return df[
        [
            "node_index",
            "corridor_file",
            "corridor_name",
            "latitude",
            "longitude",
            "degree",
        ]
    ]


def get_corridor_options():

    df = get_corridor_metadata()

    return [
        {
            "node_idx": int(row["node_index"]),
            "name": row["corridor_name"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }
        for _, row in df.iterrows()
    ]


# ============================================================
# REFERENCE TIMES
# ============================================================


def get_reference_times():

    data = load_test_data()

    target_times = data["target_times"]

    results = []

    for sample_idx in range(target_times.shape[0]):
        t1 = pd.Timestamp(target_times[sample_idx, 0])

        reference_time = t1 - pd.Timedelta(hours=1)

        results.append(
            {
                "sample_idx": sample_idx,
                "reference_time": reference_time,
                "label": reference_time.strftime("%d %b %Y, %H:%M UTC"),
            }
        )

    return results


# ============================================================
# HISTORICAL SERIES
# ============================================================


def get_historical_series(
    sample_idx,
    node_idx,
):

    data = get_real_test_data()

    t1 = pd.Timestamp(data["target_times"][sample_idx, 0])

    reference_time = t1 - pd.Timedelta(hours=1)

    history_times = pd.date_range(
        end=reference_time,
        periods=12,
        freq="h",
    )

    speed = data["X_real"][
        sample_idx,
        :,
        node_idx,
        0,
    ]

    return pd.DataFrame(
        {
            "time": history_times,
            "speed": speed,
        }
    )


# ============================================================
# TARGET DATA
# ============================================================


# ============================================================
# TARGET VALUES
# ============================================================


def get_target_values(
    sample_idx,
    node_idx,
):

    data = get_real_test_data()

    horizons = [
        1,
        3,
        6,
    ]

    rows = []

    for horizon_idx, horizon in enumerate(horizons):
        valid = bool(
            data["mask"][
                sample_idx,
                horizon_idx,
                node_idx,
            ]
        )

        actual_speed = None

        if valid:
            actual_speed = float(
                data["y_real"][
                    sample_idx,
                    horizon_idx,
                    node_idx,
                ]
            )

        target_time = pd.Timestamp(
            data["target_times"][
                sample_idx,
                horizon_idx,
            ]
        )

        rows.append(
            {
                "horizon": horizon,
                "target_time": target_time,
                "actual_speed": actual_speed,
                "valid": valid,
            }
        )

    return pd.DataFrame(rows)
