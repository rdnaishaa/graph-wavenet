from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
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


def draw_map(nodes, adjacency, selected_node):
    view_nodes = nodes.copy()

    view_nodes["color"] = view_nodes["node"].apply(
        lambda node: [244, 74, 91, 255] if node == selected_node else [42, 121, 234, 235]
    )
    view_nodes["radius"] = view_nodes["node"].apply(
        lambda node: 240 if node == selected_node else 135
    )

    edges = make_edges(nodes, adjacency, selected_node)

    edge_layer = pdk.Layer(
        "LineLayer",
        data=edges,
        get_source_position=["source_lon", "source_lat"],
        get_target_position=["target_lon", "target_lat"],
        get_color="color",
        get_width="width",
        width_min_pixels=1,
        pickable=False,
    )

    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=view_nodes,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=4,
        radius_max_pixels=12,
        pickable=True,
    )

    selected = nodes.loc[nodes["node"] == selected_node].iloc[0]

    # Fokus pada jaringan corridor Jabodetabek
    zoom = 10.0

    deck = pdk.Deck(
        layers=[edge_layer, node_layer],
        initial_view_state=pdk.ViewState(
            latitude=float(nodes["latitude"].mean()),
            longitude=float(nodes["longitude"].mean()),
            zoom=zoom,
            pitch=0,
        ),
        map_style=("https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"),
        tooltip={
            "html": "<b>Node {node}</b><br/>{corridor}",
            "style": {
                "backgroundColor": "#102655",
                "color": "white",
            },
        },
    )

    st.pydeck_chart(
        deck,
        height=450,
        width="stretch",
    )

    st.caption(
        f"🔵 Node corridor · 🔴 Node terpilih: "
        f"{selected_node} ({selected['corridor']}) · "
        "Garis merah = koneksi terkait node terpilih."
    )


def draw_matrix(matrix, heading, weighted=False):
    indices = np.indices(matrix.shape)

    table = pd.DataFrame(
        {
            "Source": indices[0].ravel(),
            "Target": indices[1].ravel(),
            "Weight": matrix.ravel().astype(float),
        }
    )

    chart = (
        alt.Chart(table)
        .mark_rect()
        .encode(
            x=alt.X(
                "Target:O",
                title="Target Node",
                axis=alt.Axis(labels=False, ticks=False),
            ),
            y=alt.Y(
                "Source:O",
                title="Source Node",
                axis=alt.Axis(labels=False, ticks=False),
            ),
            color=alt.Color(
                "Weight:Q",
                scale=alt.Scale(
                    scheme="blues",
                    domainMin=0,
                ),
                title="Bobot" if weighted else "Edge",
            ),
            tooltip=[
                alt.Tooltip("Source:O"),
                alt.Tooltip("Target:O"),
                alt.Tooltip("Weight:Q", format=".4f"),
            ],
        )
        .properties(height=350, title=heading)
        .configure_view(stroke=None)
    )

    st.altair_chart(chart, width="stretch")


def render_graph_explorer(corridor_metadata):
    st.markdown(
        """
        <div style="padding:10px 0 18px">
          <div style="color:#2879ed;font-weight:800;
                      font-size:13px;letter-spacing:1px">
            GRAPH EXPLORER
          </div>
          <div style="color:#102655;font-weight:800;
                      font-size:clamp(26px,3vw,38px);
                      margin-top:8px">
            Eksplorasi Graph Jaringan Jalan
          </div>
          <div style="color:#5b7090;font-size:16px;
                      margin-top:6px">
            Analisis representasi spasial 64 corridor
            Jabodetabek pada Graph WaveNet.
          </div>
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )

    if not GRAPH_FILE.exists():
        st.error(f"File Fixed Graph tidak ditemukan: {GRAPH_FILE.relative_to(ROOT)}")
        return

    try:
        fixed = load_fixed_graph()
        nodes = prepare_nodes(corridor_metadata)
    except (ValueError, KeyError) as exc:
        st.error(f"Data Graph Explorer belum cocok: {exc}")
        return

    adaptive, adaptive_source = load_adaptive_graph()

    section1, section2, section3 = st.tabs(
        [
            "🕸️ Visualisasi Graph",
            "▦ Adjacency Matrix",
            "📘 Perbandingan Strategi",
        ]
    )

    with section1:
        a, b = st.columns([1, 1], gap="medium")

        with a:
            strategy = st.selectbox(
                "Strategi Graph",
                ["Fixed", "Adaptive", "Hybrid"],
                key="graph_strategy",
            )

        with b:
            selected_node = st.selectbox(
                "Pilih Node",
                nodes["node"].tolist(),
                index=5,
                format_func=lambda node: (
                    f"Node {node} — {nodes.loc[nodes['node'] == node, 'corridor'].iloc[0]}"
                ),
                key="graph_node",
            )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Jumlah Node", "64")
        k2.metric("K Tetangga", "4")
        k3.metric(
            "Directed Edges",
            f"{int(np.count_nonzero(fixed))}",
        )
        k4.metric(
            "Self Loops",
            f"{int(np.count_nonzero(np.diag(fixed)))}",
        )

        left, right = st.columns([1.65, 1], gap="medium")

        with left:
            with st.container(border=True):
                st.markdown("#### City Graph — Jabodetabek")

                if strategy == "Fixed":
                    st.caption("Koneksi geografis berdasarkan Haversine dan directed KNN.")
                elif strategy == "Adaptive":
                    st.caption(
                        "Peta menunjukkan posisi corridor dan "
                        "referensi geografis, bukan learned "
                        "adaptive edges."
                    )
                else:
                    st.caption(
                        "Koneksi geografis Fixed sebagai salah "
                        "satu support Hybrid. Learned adaptive "
                        "edges tidak digambarkan pada peta ini."
                    )

                draw_map(nodes, fixed, selected_node)

        with right:
            with st.container(border=True):
                st.markdown("#### Detail Node Terpilih")

                selected = nodes.loc[nodes["node"] == selected_node].iloc[0]

                st.markdown(f"**{selected['corridor']}**")

                st.write(f"Node index: **{selected_node}**")

                st.write(f"Latitude: {selected['latitude']:.6f}")
                st.write(f"Longitude: {selected['longitude']:.6f}")

                neighbors = np.flatnonzero(fixed[selected_node] > 0)

                st.markdown("**Outgoing Neighbors (Fixed KNN)**")

                neighbor_table = nodes.loc[
                    nodes["node"].isin(neighbors),
                    ["node", "corridor"],
                ].copy()

                st.dataframe(
                    neighbor_table,
                    hide_index=True,
                    width="stretch",
                )

                st.caption(
                    "Neighbor ditentukan oleh graph Fixed, bukan oleh pemilihan Adaptive/Hybrid."
                )

        if strategy == "Fixed":
            st.info(
                "**Fixed Graph:** hubungan geografis "
                "ditetapkan sebelum training. Graph WaveNet "
                "menggunakan support forward dan reverse."
            )

        elif strategy == "Adaptive":
            if adaptive is None:
                st.warning(
                    "Matriks learned adaptive belum ditemukan "
                    "pada lokasi hasil eksperimen yang dikenali. "
                    "Peta di atas hanya menunjukkan Fixed graph "
                    "sebagai referensi lokasi, bukan visualisasi "
                    "hubungan adaptif."
                )
            else:
                st.success(
                    "Matriks adaptive ditemukan. Detail "
                    "bobotnya tersedia pada tab Adjacency Matrix."
                )

        else:
            st.info(
                "**Hybrid Graph:** memanfaatkan support "
                "geografis forward, reverse, dan learned adaptive "
                "secara bersamaan. Garis pada peta hanya "
                "merepresentasikan komponen geografis."
            )

    with section2:
        option = st.radio(
            "Pilih Matriks",
            ["Fixed KNN", "Adaptive Learned"],
            horizontal=True,
        )

        if option == "Fixed KNN":
            draw_matrix(
                (fixed > 0).astype(float),
                "Fixed Directed KNN Adjacency (64 × 64)",
            )
            st.caption(
                "Kotak berwarna menandakan directed edge. "
                "Tidak ada self-loop pada directed KNN dasar."
            )

        elif adaptive is not None:
            draw_matrix(
                adaptive,
                "Adaptive GWN · seed 42 (64 × 64)",
                weighted=True,
            )
            st.caption(f"Matriks dibaca dari file hasil eksperimen: {adaptive_source}")

        else:
            st.warning(
                "File bobot adaptive belum tersedia pada "
                "lokasi yang dikonfigurasi. Matriks ini "
                "tidak bisa dibuat hanya dari Fixed Graph."
            )

    with section3:
        st.markdown("### Perbandingan Strategi Graph")

        comparison = pd.DataFrame(
            [
                {
                    "Strategi": "Fixed",
                    "Sumber Hubungan": "Jarak geografis",
                    "Support": "Forward + Reverse",
                    "Learned Adjacency": "Tidak",
                },
                {
                    "Strategi": "Adaptive",
                    "Sumber Hubungan": "Embedding node",
                    "Support": "Adaptive",
                    "Learned Adjacency": "Ya",
                },
                {
                    "Strategi": "Hybrid",
                    "Sumber Hubungan": "Geografis + Learned",
                    "Support": "Forward + Reverse + Adaptive",
                    "Learned Adjacency": "Ya",
                },
            ]
        )

        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            with st.container(border=True):
                st.markdown("#### Fixed")
                st.write(
                    "Hubungan antar-corridor dibentuk "
                    "berdasarkan jarak geografis menggunakan "
                    "Haversine dan KNN, dengan K=4."
                )

        with c2:
            with st.container(border=True):
                st.markdown("#### Adaptive")
                st.write(
                    "Model mempelajari hubungan laten antar-node "
                    "dari data selama proses pelatihan, "
                    "tanpa harus mengikuti jarak geografis."
                )

        with c3:
            with st.container(border=True):
                st.markdown("#### Hybrid")
                st.write(
                    "Menggabungkan hubungan geografis "
                    "dan hubungan adaptif yang dipelajari "
                    "untuk memodelkan dependensi spasial."
                )

        st.caption(
            "Visualisasi ini menjelaskan struktur spatial support, bukan hasil prediksi langsung."
        )
