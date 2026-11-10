from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = ROOT / "outputs" / "final_64" / "fixed_graph" / "fixed_graph_64.npz"

# Matriks adaptive hanya ditampilkan jika memang ada
# dalam file hasil eksperimen yang teridentifikasi.
ADAPTIVE_CANDIDATES = [
    ROOT / "outputs/final_64/graphwavenet_audited/graphwavenet_adaptive_seed42_adjacency.npy",
]


def find_col(df, options):
    columns = {str(col).strip().lower(): col for col in df.columns}
    for option in options:
        if option in columns:
            return columns[option]
    return None


@st.cache_data
def load_fixed_graph():
    with np.load(GRAPH_FILE) as data:
        if "directed_knn_adjacency" not in data.files:
            raise ValueError("NPZ tidak memiliki directed_knn_adjacency.")
        matrix = data["directed_knn_adjacency"].copy()

    if matrix.shape != (64, 64):
        raise ValueError(f"Ukuran graph tidak sesuai Final64: {matrix.shape}")
    return matrix


@st.cache_data
def load_adaptive_graph():
    for path in ADAPTIVE_CANDIDATES:
        if not path.exists():
            continue

        if path.suffix == ".npy":
            matrix = np.load(path, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as data:
                keys = [
                    key
                    for key in (
                        "adaptive_adjacency",
                        "adaptive_graph",
                        "adjacency",
                    )
                    if key in data.files
                ]
                if not keys:
                    continue
                matrix = data[keys[0]].copy()

        if matrix.shape == (64, 64) and np.isfinite(matrix).all():
            return matrix.astype(float), str(path)

    return None, None


def prepare_nodes(metadata):
    df = metadata.copy()

    lat_col = find_col(df, ["latitude", "lat", "start_lat", "start_latitude"])
    lon_col = find_col(df, ["longitude", "lon", "lng", "start_lon", "start_longitude"])
    node_col = find_col(df, ["node_index", "node_id", "index"])
    name_col = find_col(df, ["corridor_name", "name", "corridor_file"])

    if not all([lat_col, lon_col, node_col]):
        raise ValueError(
            "Kolom koordinat atau node_index belum ditemukan "
            f"pada metadata. Kolom tersedia: {list(df.columns)}"
        )

    nodes = pd.DataFrame(
        {
            "node": pd.to_numeric(df[node_col], errors="coerce"),
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
            "corridor": (df[name_col].astype(str) if name_col else df[node_col].astype(str)),
        }
    ).dropna(subset=["node", "latitude", "longitude"])

    nodes["node"] = nodes["node"].astype(int)
    nodes = nodes.drop_duplicates("node").set_index("node").reindex(range(64)).reset_index()

    if nodes[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Koordinat tidak lengkap untuk 64 node.")

    return nodes


def make_edges(nodes, adjacency, selected_node=None):
    lookup = nodes.set_index("node")
    edges = []

    for src, dst in np.argwhere(adjacency > 0):
        source = lookup.loc[int(src)]
        target = lookup.loc[int(dst)]

        is_selected = selected_node is not None and (
            int(src) == selected_node or int(dst) == selected_node
        )

        edges.append(
            {
                "source_lon": float(source["longitude"]),
                "source_lat": float(source["latitude"]),
                "target_lon": float(target["longitude"]),
                "target_lat": float(target["latitude"]),
                "color": ([242, 80, 93, 230] if is_selected else [33, 86, 164, 110]),
                "width": 3 if is_selected else 1.5,
            }
        )

    return pd.DataFrame(edges)
