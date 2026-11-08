import math

import pandas as pd
import pydeck as pdk
import streamlit as st

from streamlit_app.data_loader import (
    get_corridor_metadata,
    get_historical_series,
    get_reference_times,
    get_target_values,
)
from streamlit_app.forecasting import (
    compare_models,
    forecast_corridor,
)
from streamlit_app.full_history_loader import (
    get_corridor_date_range,
    get_history_period,
)
from streamlit_app.metrics_loader import (
    get_recommended_model,
)
from streamlit_app.model_info import (
    get_model_info,
    get_model_names,
)
from streamlit_app.routing_utils import (
    get_route_between_corridors,
)
from streamlit_app.trip_analysis import (
    analyze_trip,
)
from streamlit_app.trip_utils import (
    build_trip_map_data,
    get_map_center,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Traffic Forecast Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# RESPONSIVE UI
# ============================================================

st.markdown(
    """
    <style>

    /* -----------------------------
       MAIN PAGE WIDTH
       ----------------------------- */

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }


    /* -----------------------------
       TITLE RESPONSIVE
       ----------------------------- */

    h1 {
        font-size: clamp(2rem, 4vw, 3.6rem) !important;
        line-height: 1.12 !important;
        margin-bottom: 0.6rem !important;
    }

    h2 {
        font-size: clamp(1.5rem, 2.8vw, 2.2rem) !important;
    }

    h3 {
        font-size: clamp(1.2rem, 2vw, 1.65rem) !important;
    }


    /* -----------------------------
       TABS / NAVBAR
       ----------------------------- */

    [data-baseweb="tab-list"] {
        gap: 0.35rem;
        flex-wrap: wrap !important;
        justify-content: flex-start;
        border-bottom: 1px solid rgba(128,128,128,0.25);
        padding-bottom: 0.2rem;
    }

    [data-baseweb="tab"] {
        height: auto !important;
        min-height: 42px;
        padding: 0.55rem 0.85rem !important;
        border-radius: 8px 8px 0 0;
        white-space: nowrap;
        font-size: 0.95rem;
    }

    [data-baseweb="tab"] p {
        font-size: 0.95rem !important;
        margin: 0 !important;
    }


    /* -----------------------------
       METRICS
       ----------------------------- */

    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }


    /* -----------------------------
       DATAFRAMES
       ----------------------------- */

    [data-testid="stDataFrame"] {
        width: 100% !important;
    }


    /* -----------------------------
       MAP
       ----------------------------- */

    iframe {
        max-width: 100% !important;
    }


    /* ========================================================
       TABLET
       ======================================================== */

    @media (max-width: 900px) {

        .block-container {
            padding-left: 1.1rem;
            padding-right: 1.1rem;
            padding-top: 1.2rem;
        }

        [data-baseweb="tab-list"] {
            gap: 0.25rem;
        }

        [data-baseweb="tab"] {
            padding: 0.5rem 0.65rem !important;
            font-size: 0.88rem;
        }

        [data-baseweb="tab"] p {
            font-size: 0.88rem !important;
        }
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 640px) {

        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-top: 0.9rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        [data-baseweb="tab-list"] {
            display: grid !important;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            width: 100%;
            gap: 0.35rem;
            border-bottom: none;
        }

        [data-baseweb="tab"] {
            width: 100% !important;
            justify-content: center !important;
            text-align: center;
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 9px !important;
            min-height: 44px;
            padding: 0.45rem 0.35rem !important;
            white-space: normal !important;
        }

        [data-baseweb="tab"] p {
            font-size: 0.82rem !important;
            line-height: 1.15 !important;
            white-space: normal !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
        }

        .stButton > button {
            width: 100%;
        }
    }

    
    /* ========================================================
       PAGE SPACING IMPROVEMENT
       ======================================================== */

    .block-container {
        padding-top: 2.4rem !important;
        padding-bottom: 3rem !important;
    }

    /* Jarak judul halaman */
    h1, h2, h3 {
        margin-top: 0.35rem !important;
    }

    /* Tab internal halaman */
    [data-baseweb="tab-list"] {
        margin-top: 0.9rem !important;
        margin-bottom: 1.25rem !important;
        gap: 0.55rem !important;
        border-bottom: 1px solid rgba(120,120,120,0.18);
        padding-bottom: 0.3rem !important;
    }

    [data-baseweb="tab"] {
        padding: 0.6rem 0.9rem !important;
        min-height: 42px !important;
        border-radius: 10px 10px 0 0 !important;
    }

    /* Jarak antar section */
    .stSelectbox, .stTextInput, .stDateInput {
        margin-bottom: 0.35rem !important;
    }

    [data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.15rem;
    }

    /* Card dataframe/chart biar ga sesak */
    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"],
    iframe {
        margin-top: 0.3rem !important;
    }

    /* Divider section */
    hr {
        margin-top: 1.6rem !important;
        margin-bottom: 1.4rem !important;
    }

</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NEW DASHBOARD STYLE
# ============================================================

st.markdown(
    """
<style>
:root {
    --navy: #102655;
    --blue: #2779ed;
    --muted: #647594;
    --border: #e6edf7;
}

.stApp {
    background: #f7faff;
    color: var(--navy);
}

[data-testid="stAppViewContainer"] .block-container {
    max-width: 1550px;
    padding: 1.5rem 1.7rem 3rem;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.35rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.72rem 0.8rem;
    border-radius: 10px;
    transition: background 0.15s;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #e4efff;
    color: #1161d7;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #f0f5ff;
}

[data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    font-weight: 550;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: var(--border);
}

[data-testid="stExpander"] {
    border-color: var(--border);
}

.hero-overview {
    position: relative;
    overflow: hidden;
    min-height: 210px;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 16px;
    background-size: cover;
    background-position: center;
    background-color: #eaf2ff;
}

.hero-content {
    position: relative;
    z-index: 1;
    max-width: 670px;
}

.hero-tag {
    font-weight: 750;
    color: #2479f0;
    font-size: 0.78rem;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.hero-overview h1 {
    color: #0d2350 !important;
    font-size: clamp(2rem, 3vw, 3.05rem) !important;
    line-height: 1.08 !important;
    margin: 0 0 10px !important;
}

.hero-subtitle {
    font-size: 1.08rem;
    color: #203961;
    margin-bottom: 12px;
}

.hero-description {
    font-size: 0.91rem;
    max-width: 560px;
    line-height: 1.65;
    color: #435777;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin: 12px 0 20px;
}

.kpi-card {
    border-radius: 14px;
    padding: 20px 16px;
    min-height: 116px;
    display: flex;
    align-items: center;
    gap: 13px;
    border: 1px solid rgba(255,255,255,.8);
}

.kpi-icon {
    font-size: 1.9rem;
    flex-shrink: 0;
}

.kpi-number {
    font-size: 1.65rem;
    font-weight: 800;
    color: #102655;
    line-height: 1.15;
}

.kpi-label {
    font-size: 0.84rem;
    color: #203961;
    margin-top: 3px;
}

.kpi-detail {
    font-size: 0.71rem;
    color: #60738f;
    margin-top: 5px;
}

.kpi-blue { background: #eaf4ff; }
.kpi-green { background: #e8f8f2; }
.kpi-yellow { background: #fff7e5; }
.kpi-pink { background: #fff0f3; }
.kpi-purple { background: #f2efff; }

.section-title {
    font-size: 1.13rem;
    font-weight: 750;
    color: #102655;
    margin: 5px 0 10px;
}

.info-pill {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px 5px 6px 0;
    border-radius: 8px;
    background: #eef5ff;
    font-size: 0.75rem;
    color: #244a86;
}

.research-flow {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin-top: 12px;
}

.flow-card {
    border: 1px solid #e2ebf8;
    background: #fff;
    border-radius: 13px;
    padding: 14px;
    min-height: 128px;
}

.flow-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 29px;
    height: 29px;
    border-radius: 50%;
    color: #196ce3;
    background: #dfedff;
    font-weight: 800;
    margin-bottom: 10px;
}

.flow-title {
    font-weight: 750;
    font-size: 0.82rem;
    color: #102655;
    margin-bottom: 7px;
}

.flow-detail {
    font-size: 0.74rem;
    line-height: 1.5;
    color: #647594;
}

@media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    .research-flow { grid-template-columns: repeat(3, 1fr); }
}

@media (max-width: 650px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .research-flow { grid-template-columns: repeat(2, 1fr); }
    .hero-overview { padding: 22px 18px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# GLOBAL UI POLISH V1

st.markdown(
    """
<style>

/* ==========================================================
   COLORS & TYPOGRAPHY
   ========================================================== */

:root {
    --ui-primary: #2879ed;
    --ui-primary-dark: #155ec7;
    --ui-navy: #102655;
    --ui-muted: #647594;
    --ui-border: #e2eaf5;
}

.stApp {
    background-color: #f7faff;
}

[data-testid="stAppViewContainer"] {
    background-color: #f7faff;
}

[data-testid="stAppViewContainer"] .block-container {
    max-width: 1550px;
    padding-top: 1.5rem;
    padding-left: 1.7rem;
    padding-right: 1.7rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--ui-navy);
    letter-spacing: -0.015em;
}

/* ==========================================================
   SIDEBAR
   ========================================================== */

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--ui-border);
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 6px;
}

/* Hide radio circle, keep navigation text */

[data-testid="stSidebar"] [role="radiogroup"]
[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100%;
    min-height: 41px;
    border-radius: 10px;
    padding: 10px 12px !important;
    margin: 0;
    cursor: pointer;
    transition: background-color .16s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background-color: #f0f6ff;
}

[data-testid="stSidebar"] [role="radiogroup"]
label:has(input:checked) {
    background: #e5efff !important;
    color: #1165de !important;
    box-shadow: inset 3px 0 0 #2879ed;
}

[data-testid="stSidebar"] [role="radiogroup"]
label:has(input:checked) p {
    color: #1165de !important;
    font-weight: 700 !important;
}

/* Sidebar form */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    border-radius: 10px;
    border-color: var(--ui-border);
}

/* ==========================================================
   PRIMARY BUTTONS — BLUE
   ========================================================== */

button[kind="primary"],
[data-testid="stBaseButton-primary"],
.stButton button[kind="primary"],
.stFormSubmitButton button[kind="primary"] {
    background: #2879ed !important;
    border: 1px solid #2879ed !important;
    color: #ffffff !important;
    border-radius: 9px !important;
    font-weight: 650 !important;
}

button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
.stButton button[kind="primary"]:hover {
    background: #155ec7 !important;
    border-color: #155ec7 !important;
    color: #ffffff !important;
}

button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    border-radius: 9px !important;
}

/* ==========================================================
   TABS — BLUE ACTIVE COLOR
   ========================================================== */

[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--ui-border);
    gap: 10px;
}

[data-baseweb="tab"] {
    color: #657693 !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    color: #2879ed !important;
    font-weight: 700 !important;
}

[data-baseweb="tab-highlight"] {
    background-color: #2879ed !important;
}

[data-baseweb="tab-border"] {
    background-color: var(--ui-border) !important;
}

/* ==========================================================
   INPUTS
   ========================================================== */

[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stDateInput"] [data-baseweb="input"] {
    border-radius: 9px !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"]
> div:focus-within {
    border-color: #2879ed !important;
}

/* ==========================================================
   CARDS, METRICS, TABLES
   ========================================================== */

[data-testid="stMetric"] {
    border: 1px solid var(--ui-border);
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stMetricValue"] {
    color: var(--ui-navy);
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: var(--ui-border);
    border-radius: 12px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 900px) {
    [data-testid="stAppViewContainer"] .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }
}

@media (max-width: 640px) {
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
    }

    h1 {
        font-size: 1.9rem !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

if "forecast_context" not in st.session_state:
    st.session_state.forecast_context = None

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

if "trip_result" not in st.session_state:
    st.session_state.trip_result = None

if "trip_context" not in st.session_state:
    st.session_state.trip_context = None


if "route_result" not in st.session_state:
    st.session_state.route_result = None

if "route_context" not in st.session_state:
    st.session_state.route_context = None


# ============================================================
# DISPLAY HELPERS
# ============================================================


def pretty_corridor_name(name):

    mapping = {
        "Gatsu": "Gatot Subroto",
        "Kelapagading": "Kelapa Gading",
        "Kebonjeruk": "Kebon Jeruk",
        "Tbsimatupang": "TB Simatupang",
        "Daanmogot": "Daan Mogot",
        "Bsd": "BSD",
    }

    return mapping.get(name, name)


def model_label(mode):

    mapping = {
        "fixed": "Fixed Graph WaveNet",
        "adaptive": "Adaptive Graph WaveNet",
        "hybrid": "Hybrid Graph WaveNet",
    }

    return mapping.get(mode, mode)


def display_time(timestamp):

    ts = pd.Timestamp(timestamp)

    return ts.strftime("%d %b %Y, %H:%M UTC")


# ============================================================
# LOAD STATIC DATA
# ============================================================

corridor_df = get_corridor_metadata()
reference_times = get_reference_times()
model_names = get_model_names()

recommended = get_recommended_model()
recommended_name = recommended["model_name"]


corridor_records = corridor_df.to_dict(orient="records")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:8px 0 20px">
        <div style="font-size:22px;font-weight:800;color:#102655">
            🔷 TomTom STGNN
        </div>
        <div style="font-size:13px;color:#6480a6">
            Research Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page_options = [
    "🏠 Overview",
    "🗃️ Dataset Explorer",
    "🕸️ Graph Explorer",
    "📈 Traffic Forecasting",
    "📊 Model Comparison",
    "🗺️ Analisis Perjalanan",
    "⚙️ Computational & Scalability",
]

active_page = st.sidebar.radio(
    "Navigasi",
    page_options,
    label_visibility="collapsed",
)

st.sidebar.divider()

# Pengaturan tetap tersedia untuk fungsi website lama.
with st.sidebar.expander("⚙️ Pengaturan Prediksi", expanded=False):
    selected_model = st.selectbox(
        "Model Graph WaveNet",
        model_names,
        index=(model_names.index(recommended_name) if recommended_name in model_names else 0),
    )

    selected_corridor_record = st.selectbox(
        "Corridor",
        corridor_records,
        format_func=lambda item: pretty_corridor_name(item["corridor_name"]),
    )

    selected_reference = st.selectbox(
        "Waktu Acuan",
        reference_times,
        format_func=lambda item: item["label"],
    )

selected_node = int(selected_corridor_record["node_index"])
selected_corridor_raw = selected_corridor_record["corridor_name"]
selected_corridor_display = pretty_corridor_name(selected_corridor_raw)
selected_sample = int(selected_reference["sample_idx"])
selected_model_info = get_model_info(selected_model)

st.sidebar.caption("Historical Test-Set Demo · Final64 · 2026")


if active_page == "🏠 Overview":
    import base64
    import html
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data" / "final_64"

    # HERO: optional local Jakarta photograph
    hero_path = ROOT / "streamlit_app" / "assets" / "jakarta_hero.jpeg"

    background = (
        "linear-gradient(100deg, "
        "rgba(247,250,255,1) 0%, "
        "rgba(247,250,255,.97) 44%, "
        "rgba(247,250,255,.15) 100%)"
    )

    if hero_path.exists():
        encoded = base64.b64encode(hero_path.read_bytes()).decode("ascii")

        background += f",url('data:image/jpeg;base64,{encoded}')"
    else:
        background += ",linear-gradient(115deg,#e9f3ff,#d2e6ff)"

    st.markdown(
        f"""
        <div class="hero-overview"
             style="background-image:{background}">
            <div class="hero-content">
                <div class="hero-tag">OVERVIEW</div>
                <h1>Traffic Speed Forecasting<br>
                    in Jabodetabek</h1>
                <div class="hero-subtitle">
                    Spatio-Temporal Graph Neural Network
                    with TomTom Data (Final64)
                </div>
                <div class="hero-description">
                    Dashboard hasil penelitian prediksi kecepatan
                    lalu lintas pada 64 corridor Jabodetabek
                    menggunakan Graph WaveNet dengan strategi
                    Fixed, Adaptive, dan Hybrid.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI CARDS
    kpis = [
        ("🛣️", "64", "Corridor", "Node jaringan", "blue"),
        ("🗃️", "600", "Sampel", "415 train · 92 val · 93 test", "green"),
        ("🕒", "12 jam", "Historis input", "Interval 1 jam", "yellow"),
        ("🎯", "3 horizon", "Prediksi", "+1, +3, +6 jam", "pink"),
        ("🧊", "4 model", "Eksperimen", "3 GWN + GC-LSTM-style", "purple"),
    ]

    kpi_html = '<div class="kpi-grid">'

    for icon, value, label, detail, color in kpis:
        kpi_html += f"""
        <div class="kpi-card kpi-{color}">
            <div class="kpi-icon">{icon}</div>
            <div>
                <div class="kpi-number">{value}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-detail">{detail}</div>
            </div>
        </div>
        """

    kpi_html += "</div>"

    st.markdown("".join(line.strip() for line in kpi_html.splitlines()), unsafe_allow_html=True)

    # MAIN VISUALS
    left, right = st.columns([1, 1.2], gap="small")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">City Graph 64 Corridor</div>',
                unsafe_allow_html=True,
            )

            graph_file = ROOT / "fixed_graph_64_nodes.png"

            import numpy as np

            graph_npz = ROOT / "outputs" / "final_64" / "fixed_graph" / "fixed_graph_64.npz"

            def find_column(df, candidates):
                lookup = {col.lower().strip(): col for col in df.columns}
                for candidate in candidates:
                    if candidate in lookup:
                        return lookup[candidate]
                return None

            # Sumber koordinat: metadata corridor asli
            map_df = corridor_df.copy()

            lat_col = find_column(map_df, ["latitude", "lat", "start_lat", "start_latitude"])

            lon_col = find_column(
                map_df, ["longitude", "lon", "lng", "start_lon", "start_longitude"]
            )

            index_col = find_column(map_df, ["node_index", "node_id", "index"])

            if lat_col is None or lon_col is None:
                st.warning(
                    "Koordinat belum ditemukan pada metadata. "
                    "Sementara menampilkan visualisasi graph asli."
                )
                if graph_file.exists():
                    st.image(
                        str(graph_file),
                        width="stretch",
                    )

                st.caption("Kolom metadata: " + ", ".join(map_df.columns.astype(str)))

            elif not graph_npz.exists():
                st.warning("File fixed_graph_64.npz belum ditemukan.")
                if graph_file.exists():
                    st.image(
                        str(graph_file),
                        width="stretch",
                    )

            else:
                if index_col is None:
                    st.warning(
                        "Metadata tidak memiliki node_index. "
                        "Peta graph belum bisa dicocokkan secara aman."
                    )
                    if graph_file.exists():
                        st.image(
                            str(graph_file),
                            width="stretch",
                        )
                else:
                    map_df["node_index_map"] = pd.to_numeric(map_df[index_col], errors="coerce")

                    map_df["latitude_map"] = pd.to_numeric(map_df[lat_col], errors="coerce")

                    map_df["longitude_map"] = pd.to_numeric(map_df[lon_col], errors="coerce")

                    map_df = map_df.dropna(
                        subset=[
                            "node_index_map",
                            "latitude_map",
                            "longitude_map",
                        ]
                    )

                    map_df["node_index_map"] = map_df["node_index_map"].astype(int)

                    with np.load(graph_npz) as graph_data:
                        A = graph_data["directed_knn_adjacency"]

                    coords = map_df.drop_duplicates("node_index_map").set_index("node_index_map")

                    # Pastikan urutan node sesuai adjacency
                    valid_ids = set(range(A.shape[0]))

                    if len(coords.index.intersection(valid_ids)) != A.shape[0]:
                        st.error(
                            "Koordinat belum lengkap untuk 64 node. Periksa mapping node_index."
                        )
                    else:
                        edges = []

                        for src, dst in np.argwhere(A > 0):
                            source = coords.loc[int(src)]
                            target = coords.loc[int(dst)]

                            edges.append(
                                {
                                    "source_lon": float(source["longitude_map"]),
                                    "source_lat": float(source["latitude_map"]),
                                    "target_lon": float(target["longitude_map"]),
                                    "target_lat": float(target["latitude_map"]),
                                }
                            )

                        edge_df = pd.DataFrame(edges)

                        nodes = map_df.copy()

                        nodes["node_color"] = nodes["node_index_map"].apply(
                            lambda idx: [240, 65, 70, 240] if idx == 5 else [31, 111, 233, 220]
                        )

                        nodes["node_radius"] = nodes["node_index_map"].apply(
                            lambda idx: 230 if idx == 5 else 130
                        )

                        name_col = find_column(nodes, ["corridor_name", "name", "corridor_file"])

                        nodes["label"] = (
                            nodes[name_col].astype(str)
                            if name_col
                            else nodes["node_index_map"].astype(str)
                        )

                        edge_layer = pdk.Layer(
                            "LineLayer",
                            data=edge_df,
                            get_source_position=["source_lon", "source_lat"],
                            get_target_position=["target_lon", "target_lat"],
                            get_color=[31, 62, 115, 115],
                            get_width=2,
                            width_min_pixels=1,
                            pickable=False,
                        )

                        node_layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=nodes,
                            get_position=["longitude_map", "latitude_map"],
                            get_fill_color="node_color",
                            get_radius="node_radius",
                            radius_min_pixels=4,
                            radius_max_pixels=12,
                            pickable=True,
                        )

                        view = pdk.ViewState(
                            latitude=float(nodes["latitude_map"].mean()),
                            longitude=float(nodes["longitude_map"].mean()),
                            zoom=10.2,
                            pitch=0,
                        )

                        deck = pdk.Deck(
                            layers=[edge_layer, node_layer],
                            initial_view_state=view,
                            map_style=(
                                "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
                            ),
                            tooltip={
                                "html": ("<b>Node {node_index_map}</b><br/>{label}"),
                                "style": {
                                    "backgroundColor": "#102655",
                                    "color": "white",
                                },
                            },
                        )

                        st.pydeck_chart(
                            deck,
                            width="stretch",
                            height=400,
                        )

                        st.caption(
                            "🔵 Corridor node &nbsp;&nbsp; "
                            "🔴 Sudirman (node 5) &nbsp;&nbsp; "
                            "— Koneksi KNN",
                            unsafe_allow_html=True,
                        )

            st.markdown(
                """
                <span class="info-pill">64 node</span>
                <span class="info-pill">K = 4</span>
                <span class="info-pill">256 directed edges</span>
                <span class="info-pill">0 self-loop</span>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "Jaringan Fixed Graph dibentuk berdasarkan "
                "jarak geografis Haversine dan K-Nearest Neighbors."
            )

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title">Contoh Pola Temporal: Corridor Sudirman</div>',
                unsafe_allow_html=True,
            )

            paths = [
                DATA / "forecasting_train_preprocessed.csv",
                DATA / "forecasting_val_preprocessed.csv",
                DATA / "forecasting_test_preprocessed.csv",
            ]

            if all(path.exists() for path in paths):
                history = pd.concat(
                    [pd.read_csv(path) for path in paths],
                    ignore_index=True,
                )

                sudirman = history.loc[history["corridor_file"] == "tt_sudirman.csv"].copy()

                sudirman["is_observed_flag"] = (
                    sudirman["is_observed"].astype(str).str.lower().isin(["true", "1", "yes"])
                )

                observed = sudirman.loc[
                    sudirman["is_observed_flag"] & sudirman["current_speed"].notna()
                ].copy()

                observed["hour_wib"] = (
                    pd.to_datetime(
                        observed["obs_time_utc"],
                        utc=True,
                    )
                    .dt.tz_convert("Asia/Jakarta")
                    .dt.hour
                )

                hourly = observed.groupby("hour_wib")["current_speed"].mean().reindex(range(24))

                st.caption("Rata-rata kecepatan berdasarkan jam WIB · data observasi asli")

                st.line_chart(
                    pd.DataFrame({"Kecepatan (km/jam)": hourly}),
                    width="stretch",
                    height=290,
                )

                st.markdown(
                    '<div class="section-title">Temuan Utama (Corridor Sudirman)</div>',
                    unsafe_allow_html=True,
                )

                imputed_flag = (
                    sudirman["was_imputed"].astype(str).str.lower().isin(["true", "1", "yes"])
                )

                valid = (sudirman["is_observed_flag"] | imputed_flag) & sudirman[
                    "current_speed"
                ].notna()

                total = int(valid.sum())
                n_observed = int((sudirman["is_observed_flag"] & valid).sum())
                n_imputed = int((imputed_flag & valid).sum())

                fastest_hour = int(hourly.idxmax())
                slowest_hour = int(hourly.idxmin())

                cards = st.columns(4, gap="small")

                with cards[0]:
                    st.metric(
                        "Kecepatan tertinggi",
                        f"{hourly.max():.2f}",
                    )
                    st.caption(f"{fastest_hour:02d}.00 WIB · km/jam")

                with cards[1]:
                    st.metric(
                        "Kecepatan terendah",
                        f"{hourly.min():.2f}",
                    )
                    st.caption(f"{slowest_hour:02d}.00 WIB · km/jam")

                with cards[2]:
                    st.metric(
                        "Rentang",
                        f"{hourly.max() - hourly.min():.2f}",
                    )
                    st.caption("km/jam")

                with cards[3]:
                    st.metric(
                        "Observed",
                        f"{n_observed}",
                    )
                    st.caption(f"{n_observed / total * 100:.2f}% · imputasi {n_imputed}")

            else:
                st.warning("File historical Final64 belum ditemukan.")

    # RESEARCH WORKFLOW
    st.markdown(
        '<div class="section-title">Alur Penelitian</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("Data TomTom (Final64)", "64 corridor Jabodetabek · data historis kecepatan"),
        ("City Graph", "Fixed geografis · Adaptive learned · Hybrid"),
        ("Graph WaveNet", "Temporal convolution · spatial diffusion"),
        ("Eksperimen & Evaluasi", "3 horizon · 5 seed · MAE dan RMSE"),
        ("Analisis & Insight", "Perbandingan model · komputasi · skalabilitas"),
    ]

    flow_html = '<div class="research-flow">'

    for i, (name, detail) in enumerate(steps, start=1):
        flow_html += f"""
        <div class="flow-card">
            <div class="flow-num">{i}</div>
            <div class="flow-title">{html.escape(name)}</div>
            <div class="flow-detail">{html.escape(detail)}</div>
        </div>
        """

    flow_html += "</div>"

    st.markdown("".join(line.strip() for line in flow_html.splitlines()), unsafe_allow_html=True)

    st.caption(
        "Seluruh prediksi dan evaluasi yang ditampilkan "
        "menggunakan data historis/test set, bukan prediksi live."
    )


# ============================================================
# FORECAST
# ============================================================


if active_page == "📈 Traffic Forecasting":
    import altair as alt
    import numpy as np

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="padding:10px 0 18px">
            <div style="color:#2879ed;font-size:13px;
                        font-weight:800;letter-spacing:1px">
                TRAFFIC FORECASTING
            </div>
            <div style="font-size:clamp(27px,3vw,39px);
                        font-weight:800;color:#102655;
                        line-height:1.2;margin:8px 0">
                Prediksi Kecepatan Lalu Lintas
            </div>
            <div style="color:#5b7090;font-size:16px">
                Eksplorasi prediksi multi-horizon
                menggunakan Graph WaveNet pada
                dataset historis Final64.
            </div>
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )

    st.info(
        "Historical Test-Set Demo — Model menggunakan "
        "12 jam riwayat untuk memprediksi kecepatan "
        "+1, +3, dan +6 jam. Bukan prediksi real-time."
    )

    # --------------------------------------------------------
    # SELECTED SETTINGS
    # --------------------------------------------------------

    st.markdown("### Konfigurasi Prediksi")

    with st.container(border=True):
        st.caption(
            "Ubah model, corridor, dan waktu acuan melalui menu Pengaturan Prediksi pada sidebar."
        )

        a, b, c = st.columns(3, gap="medium")

        with a:
            st.markdown("**Model Graph WaveNet**")
            st.markdown(f"#### {selected_model}")

        with b:
            st.markdown("**Corridor Terpilih**")
            st.markdown(f"#### {selected_corridor_display}")

        with c:
            st.markdown("**Node Index**")
            st.markdown(f"#### {selected_node}")

        st.caption("Waktu acuan: " + display_time(selected_reference["reference_time"]))

        horizon_choice = st.selectbox(
            "Horizon yang Ingin Dianalisis",
            [
                "Semua Horizon",
                "+1 jam",
                "+3 jam",
                "+6 jam",
            ],
            key="forecast_horizon_filter",
        )

        run_forecast = st.button(
            "▶ Jalankan Prediksi",
            type="primary",
            width="stretch",
            key="forecast_button",
        )

    # --------------------------------------------------------
    # INFERENCE — REUSE EXISTING MODEL
    # --------------------------------------------------------

    if run_forecast:
        with st.spinner("Menjalankan model pada test set Final64..."):
            result = forecast_corridor(
                model_name=selected_model,
                sample_idx=selected_sample,
                node_idx=selected_node,
            )

        st.session_state.forecast_result = result

        st.session_state.forecast_context = {
            "corridor": selected_corridor_raw,
            "corridor_display": selected_corridor_display,
            "model": selected_model,
            "sample_idx": selected_sample,
            "node_idx": selected_node,
            "reference_time": selected_reference["reference_time"],
        }

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    result = st.session_state.get("forecast_result")
    context = st.session_state.get("forecast_context")

    if result is None or context is None:
        st.markdown("### Hasil Prediksi")

        with st.container(border=True):
            st.info("Pilih konfigurasi, kemudian klik **Jalankan Prediksi** untuk melihat hasil.")

    else:
        settings_changed = (
            context["model"] != selected_model
            or context["sample_idx"] != selected_sample
            or context["node_idx"] != selected_node
        )

        if settings_changed:
            st.warning(
                "Konfigurasi sekarang berbeda dari hasil "
                "prediksi yang tersimpan. Klik "
                "**Jalankan Prediksi** untuk memperbarui hasil."
            )

        st.markdown("### Hasil Prediksi Multi-Horizon")

        st.caption(
            "Model: "
            + str(context["model"])
            + " · Corridor: "
            + str(context["corridor_display"])
            + " · Waktu acuan: "
            + display_time(context["reference_time"])
        )

        # ----------------------------------------------------
        # HISTORICAL INPUT — 12 HOURS
        # ----------------------------------------------------

        st.markdown("### Input Historis — 12 Jam")

        st.caption(
            "Data input asli yang digunakan model pada "
            f"corridor {context['corridor_display']} "
            "untuk sampel test terpilih."
        )

        history_df = get_historical_series(
            sample_idx=context["sample_idx"],
            node_idx=context["node_idx"],
        ).copy()

        history_df["time"] = pd.to_datetime(history_df["time"], utc=True).dt.tz_convert(
            "Asia/Jakarta"
        )

        history_df["speed"] = pd.to_numeric(history_df["speed"], errors="coerce")

        with st.container(border=True):
            st.markdown("#### Riwayat Kecepatan Sebelum Prediksi")

            input_chart = (
                alt.Chart(history_df)
                .mark_line(
                    color="#2879ed",
                    strokeWidth=3,
                    point=alt.OverlayMarkDef(
                        filled=True,
                        size=65,
                    ),
                )
                .encode(
                    x=alt.X(
                        "time:T",
                        title="Waktu (WIB)",
                        axis=alt.Axis(
                            format="%d %b %H:%M",
                            labelAngle=-35,
                        ),
                    ),
                    y=alt.Y(
                        "speed:Q",
                        title="Kecepatan (km/jam)",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "time:T",
                            title="Waktu WIB",
                            format="%d %b %Y %H:%M",
                        ),
                        alt.Tooltip(
                            "speed:Q",
                            title="Kecepatan",
                            format=".2f",
                        ),
                    ],
                )
                .properties(height=270)
                .configure_view(stroke=None)
            )

            st.altair_chart(
                input_chart,
                width="stretch",
            )

            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Rata-rata Input",
                f"{history_df['speed'].mean():.2f} km/jam",
            )

            k2.metric(
                "Kecepatan Minimum",
                f"{history_df['speed'].min():.2f} km/jam",
            )

            k3.metric(
                "Kecepatan Maksimum",
                f"{history_df['speed'].max():.2f} km/jam",
            )

            with st.expander("Lihat 12 Data Historis"):
                history_table = history_df.copy()

                history_table["time"] = history_table["time"].dt.strftime("%d %b %Y %H:%M")

                history_table.columns = [
                    "Waktu (WIB)",
                    "Kecepatan (km/jam)",
                ]

                st.dataframe(
                    history_table,
                    hide_index=True,
                    width="stretch",
                )

        st.caption(
            "Grafik menampilkan 12 nilai input dari tensor "
            "X_real pada test set Final64. "
            "Data dapat mencakup nilai hasil preprocessing."
        )

        prediction = result["forecast"].copy()

        prediction["predicted_speed"] = pd.to_numeric(
            prediction["predicted_speed"],
            errors="coerce",
        )

        # Read the actual targets from the original test set.
        actual = get_target_values(
            sample_idx=context["sample_idx"],
            node_idx=result["node_idx"],
        ).copy()

        actual["actual_speed"] = pd.to_numeric(
            actual["actual_speed"],
            errors="coerce",
        )

        actual["valid"] = actual["valid"].astype(str).str.lower().isin(["true", "1", "yes"])

        actual.loc[~actual["valid"], "actual_speed"] = np.nan

        # Merge using horizon, not row position.
        comparison = prediction.merge(
            actual[["horizon", "actual_speed", "valid"]],
            on="horizon",
            how="left",
            validate="one_to_one",
        )

        comparison["absolute_error"] = (
            comparison["predicted_speed"] - comparison["actual_speed"]
        ).abs()

        comparison["time_display"] = pd.to_datetime(comparison["target_time"]).dt.strftime(
            "%d %b %Y %H:%M"
        )

        if horizon_choice != "Semua Horizon":
            horizon_map = {
                "+1 jam": 1,
                "+3 jam": 3,
                "+6 jam": 6,
            }

            selected_horizon = horizon_map[horizon_choice]

            visible = comparison.loc[
                pd.to_numeric(
                    comparison["horizon"],
                    errors="coerce",
                )
                == selected_horizon
            ].copy()

        else:
            visible = comparison.copy()

        if visible.empty:
            st.warning("Tidak ada data hasil untuk horizon terpilih.")
        else:
            # --------------------------------------------
            # PREDICTION KPI
            # --------------------------------------------

            metric_columns = st.columns(len(visible))

            for col, (_, row) in zip(metric_columns, visible.iterrows()):
                with col:
                    with st.container(border=True):
                        st.caption(f"Prediksi {row['label']}")

                        st.metric(
                            "Kecepatan",
                            f"{row['predicted_speed']:.2f} km/jam",
                        )

                        st.caption(str(row["time_display"]))

            st.markdown("### Prediksi vs Aktual")

            left, right = st.columns(
                [1.55, 1],
                gap="medium",
            )

            with left:
                with st.container(border=True):
                    st.markdown("#### Perbandingan Kecepatan")

                    plot_df = visible[
                        [
                            "label",
                            "predicted_speed",
                            "actual_speed",
                        ]
                    ].melt(
                        id_vars="label",
                        value_vars=[
                            "predicted_speed",
                            "actual_speed",
                        ],
                        var_name="series",
                        value_name="speed",
                    )

                    plot_df["series"] = plot_df["series"].map(
                        {
                            "predicted_speed": "Prediksi",
                            "actual_speed": "Aktual",
                        }
                    )

                    plot_df = plot_df.dropna(subset=["speed"])

                    chart = (
                        alt.Chart(plot_df)
                        .mark_bar(
                            cornerRadiusTopLeft=5,
                            cornerRadiusTopRight=5,
                        )
                        .encode(
                            x=alt.X(
                                "label:N",
                                title="Horizon Prediksi",
                                sort=list(visible["label"]),
                            ),
                            xOffset="series:N",
                            y=alt.Y(
                                "speed:Q",
                                title="Kecepatan (km/jam)",
                                scale=alt.Scale(zero=True),
                            ),
                            color=alt.Color(
                                "series:N",
                                scale=alt.Scale(
                                    domain=[
                                        "Prediksi",
                                        "Aktual",
                                    ],
                                    range=[
                                        "#2879ed",
                                        "#f46b78",
                                    ],
                                ),
                                legend=alt.Legend(
                                    title=None,
                                    orient="top",
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "label:N",
                                    title="Horizon",
                                ),
                                alt.Tooltip(
                                    "series:N",
                                    title="Data",
                                ),
                                alt.Tooltip(
                                    "speed:Q",
                                    title="Kecepatan",
                                    format=".2f",
                                ),
                            ],
                        )
                        .properties(height=315)
                        .configure_view(stroke=None)
                    )

                    st.altair_chart(
                        chart,
                        width="stretch",
                    )

                    st.caption(
                        "Biru menunjukkan prediksi model, "
                        "merah menunjukkan target aktual "
                        "test set yang tersedia."
                    )

            with right:
                with st.container(border=True):
                    st.markdown("#### Ringkasan Evaluasi Sampel")

                    valid_error = visible["absolute_error"].dropna()

                    if not valid_error.empty:
                        st.metric(
                            "Mean Absolute Error (sampel)",
                            f"{valid_error.mean():.2f} km/jam",
                        )

                        st.metric(
                            "Absolute Error Tertinggi",
                            f"{valid_error.max():.2f} km/jam",
                        )

                        st.caption(f"{len(valid_error)} horizon memiliki target aktual valid.")

                    else:
                        st.info("Target aktual tidak tersedia untuk pilihan ini.")

                    st.divider()

                    st.metric(
                        "Inference Time",
                        f"{result['inference_ms']:.2f} ms",
                    )

                    st.caption(
                        "Pengukuran pada mesin yang menjalankan aplikasi, setelah warm-up model."
                    )

            # --------------------------------------------
            # RESULT TABLE
            # --------------------------------------------

            st.markdown("### Detail Prediksi")

            table = visible[
                [
                    "label",
                    "time_display",
                    "predicted_speed",
                    "actual_speed",
                    "absolute_error",
                ]
            ].copy()

            table.columns = [
                "Horizon",
                "Waktu Target",
                "Prediksi (km/jam)",
                "Aktual (km/jam)",
                "Absolute Error (km/jam)",
            ]

            st.dataframe(
                table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Prediksi (km/jam)": st.column_config.NumberColumn(format="%.2f"),
                    "Aktual (km/jam)": st.column_config.NumberColumn(format="%.2f"),
                    "Absolute Error (km/jam)": st.column_config.NumberColumn(format="%.2f"),
                },
            )

            st.caption(
                "Absolute Error = |prediksi − aktual|. "
                "Nilai kosong berarti target aktual "
                "tidak tersedia atau tidak valid."
            )

    st.divider()

    st.markdown("### Cara Kerja Prediksi")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        with st.container(border=True):
            st.markdown("**01 · Input**")
            st.caption("12 jam data kecepatan historis.")

    with s2:
        with st.container(border=True):
            st.markdown("**02 · Spatial**")
            st.caption("Relasi antar-node berdasarkan strategi graph.")

    with s3:
        with st.container(border=True):
            st.markdown("**03 · Temporal**")
            st.caption("Graph WaveNet mempelajari pola waktu.")

    with s4:
        with st.container(border=True):
            st.markdown("**04 · Output**")
            st.caption("Prediksi +1, +3, dan +6 jam.")

    st.caption(
        "Halaman ini mengevaluasi satu sampel historis. "
        "Hasilnya tidak menggantikan metrik agregat "
        "multi-seed pada halaman Model Comparison."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

if active_page == "📊 Model Comparison":
    from streamlit_app.model_comparison_page import render_model_comparison

    render_model_comparison(
        compare_models=compare_models,
        selected_sample=selected_sample,
        selected_node=selected_node,
        selected_corridor_display=selected_corridor_display,
        display_time=display_time,
        model_label=model_label,
    )


# ============================================================
# HISTORICAL
# ============================================================


# ============================================================
# TRIP INSIGHT
# ============================================================

if active_page == "🗺️ Analisis Perjalanan":
    st.markdown(
        """
        <div style="padding:8px 0 19px">
            <div style="color:#2879ed;font-size:13px;
                        font-weight:800;letter-spacing:1px">
                TRIP ANALYSIS
            </div>
            <div style="color:#102655;font-weight:800;
                        font-size:clamp(27px,3vw,39px);
                        line-height:1.2;margin-top:8px">
                Analisis Perjalanan Jabodetabek
            </div>
            <div style="font-size:16px;color:#5b7090;
                        margin-top:9px">
                Eksplorasi rute jalan dan prediksi kondisi
                lalu lintas pada corridor asal dan tujuan.
            </div>
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )

    st.info(
        "TomTom Routing API menyediakan rute dan estimasi "
        "waktu tempuh saat permintaan dilakukan. "
        "Graph WaveNet menampilkan prediksi historis "
        "test set untuk +1, +3, dan +6 jam. "
        "Keduanya bukan satu prediksi waktu tempuh masa depan."
    )

    st.markdown("### Konfigurasi Perjalanan")
    st.caption("Pilih corridor asal, tujuan, waktu acuan, dan model Graph WaveNet.")

    # ========================================================
    # SELECT ORIGIN / DESTINATION
    # ========================================================

    trip_col1, trip_col2 = st.columns(2)

    with trip_col1:
        origin_record = st.selectbox(
            "Corridor Asal",
            corridor_records,
            index=next(
                (i for i, item in enumerate(corridor_records) if item["corridor_name"] == "Depok"),
                0,
            ),
            format_func=lambda item: pretty_corridor_name(item["corridor_name"]),
            key="trip_origin",
        )

    with trip_col2:
        destination_record = st.selectbox(
            "Corridor Tujuan",
            corridor_records,
            index=next(
                (i for i, item in enumerate(corridor_records) if item["corridor_name"] == "Bekasi"),
                1,
            ),
            format_func=lambda item: pretty_corridor_name(item["corridor_name"]),
            key="trip_destination",
        )

    trip_col3, trip_col4 = st.columns(2)

    with trip_col3:
        trip_reference = st.selectbox(
            "Waktu Acuan Forecast",
            reference_times,
            format_func=lambda item: item["label"],
            key="trip_reference",
        )

    with trip_col4:
        trip_model = st.selectbox(
            "Model Graph WaveNet",
            model_names,
            index=(model_names.index(recommended_name) if recommended_name in model_names else 0),
            key="trip_model",
        )

    origin_raw = origin_record["corridor_name"]

    destination_raw = destination_record["corridor_name"]

    origin_display = pretty_corridor_name(origin_raw)

    destination_display = pretty_corridor_name(destination_raw)

    if trip_model == "Hybrid Graph WaveNet":
        st.success(
            "Hybrid Graph WaveNet digunakan sebagai konfigurasi "
            "utama karena memperoleh mean MAE dan RMSE terendah "
            "pada t+1, t+3, dan t+6 dalam evaluasi multi-seed."
        )

    # ========================================================
    # RUN ROUTE + FORECAST
    # ========================================================

    if st.button(
        "Tampilkan Rute & Analisis Perjalanan",
        type="primary",
        key="trip_button",
    ):
        if origin_raw == destination_raw:
            st.error("Corridor asal dan tujuan harus berbeda.")

        else:
            with st.spinner("Mengambil rute TomTom dan menjalankan Graph WaveNet..."):
                try:
                    route_result = get_route_between_corridors(
                        origin_record=origin_record,
                        destination_record=destination_record,
                        traffic=True,
                    )

                    trip_result = analyze_trip(
                        model_name=trip_model,
                        sample_idx=int(trip_reference["sample_idx"]),
                        origin_name=origin_raw,
                        destination_name=destination_raw,
                    )

                    st.session_state.route_result = route_result

                    st.session_state.route_context = {
                        "origin": origin_raw,
                        "destination": destination_raw,
                        "origin_display": origin_display,
                        "destination_display": destination_display,
                    }

                    st.session_state.trip_result = trip_result

                    st.session_state.trip_context = {
                        "origin_display": origin_display,
                        "destination_display": destination_display,
                        "model": trip_model,
                        "reference_time": trip_reference["reference_time"],
                    }

                except Exception as exc:
                    st.error(f"Gagal memuat rute atau forecast: {exc}")

    # ========================================================
    # DISPLAY ROUTE
    # ========================================================

    if st.session_state.route_result is not None and st.session_state.trip_result is not None:
        route_result = st.session_state.route_result

        route_context = st.session_state.route_context

        trip_result = st.session_state.trip_result

        trip_context = st.session_state.trip_context

        # ====================================================
        # ROUTE SUMMARY
        # ====================================================

        st.markdown(
            f"### {route_context['origin_display']} → {route_context['destination_display']}"
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Jarak Rute", f"{route_result['distance_km']:.2f} km")

        with r2:
            st.metric("Estimasi Waktu Tempuh", f"{route_result['travel_time_minutes']:.1f} menit")

        with r3:
            st.metric("Traffic Delay TomTom", f"{route_result['traffic_delay_minutes']:.1f} menit")

        # ====================================================
        # MAP DATA
        # ====================================================

        st.markdown("### Peta Rute Perjalanan")

        route_df = route_result["route_df"]

        route_path = route_df[
            [
                "longitude",
                "latitude",
            ]
        ].values.tolist()

        path_data = [
            {
                "path": route_path,
                "name": (
                    f"{route_context['origin_display']} → {route_context['destination_display']}"
                ),
            }
        ]

        corridor_map_df = build_trip_map_data(
            origin_name=route_context["origin"],
            destination_name=route_context["destination"],
        )

        show_all_corridors = st.checkbox(
            "Tampilkan titik seluruh corridor",
            value=False,
            key="show_all_corridors",
        )

        if not show_all_corridors:
            corridor_map_df = corridor_map_df[
                corridor_map_df["status"].isin(
                    [
                        "Asal",
                        "Tujuan",
                    ]
                )
            ].copy()

        corridor_map_df["display_name"] = corridor_map_df["corridor_name"].apply(
            pretty_corridor_name
        )

        corridor_map_df["color"] = corridor_map_df["status"].map(
            {
                "Asal": [22, 163, 74, 230],
                "Tujuan": [220, 38, 38, 230],
                "Corridor": [60, 100, 220, 150],
            }
        )

        corridor_map_df["radius"] = corridor_map_df["status"].map(
            {
                "Asal": 1100,
                "Tujuan": 1100,
                "Corridor": 300,
            }
        )

        path_layer = pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color=[40, 121, 237, 240],
            get_width=8,
            width_min_pixels=5,
            rounded=True,
            pickable=True,
        )

        point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=corridor_map_df,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
        )

        map_center = get_map_center(
            origin_name=route_context["origin"],
            destination_name=route_context["destination"],
        )

        # Fokus otomatis berdasarkan bounding box rute.
        min_lat = route_df["latitude"].min()
        max_lat = route_df["latitude"].max()
        min_lon = route_df["longitude"].min()
        max_lon = route_df["longitude"].max()

        lat_span = max(max_lat - min_lat, 0.01)
        lon_span = max(max_lon - min_lon, 0.01)

        # Estimasi zoom agar seluruh rute tetap terlihat.
        zoom_lat = math.log2(150 / lat_span)
        zoom_lon = math.log2(150 / lon_span)
        route_zoom = min(zoom_lat, zoom_lon)
        route_zoom = max(9.0, min(12.0, route_zoom))

        map_view = pdk.ViewState(
            latitude=float((min_lat + max_lat) / 2),
            longitude=float((min_lon + max_lon) / 2),
            zoom=route_zoom,
            pitch=0,
        )

        tooltip = {
            "html": "<b>{display_name}</b><br>Status: {status}<br>Node: {node_index}",
            "style": {
                "backgroundColor": "rgba(20,20,20,0.9)",
                "color": "white",
            },
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[
                    path_layer,
                    point_layer,
                ],
                initial_view_state=map_view,
                tooltip=tooltip,
                map_style=("https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"),
            ),
            width="stretch",
        )

        st.caption(
            "Garis pada peta mengikuti geometri jalan yang dikembalikan "
            "TomTom Routing API, bukan garis lurus antar titik."
        )

        # ====================================================
        # GRAPH WAVENET FORECAST
        # ====================================================

        st.markdown("### Prediksi Kondisi Corridor Asal & Tujuan")

        f1, f2, f3 = st.columns(3)

        trip_forecast = trip_result["forecast"]

        for idx, (_, row) in enumerate(trip_forecast.iterrows()):
            with [
                f1,
                f2,
                f3,
            ][idx]:
                st.metric(
                    row["label"],
                    (f"{row['mean_endpoint_speed']:.2f}"),
                )

                st.caption("Rata-rata prediksi endpoint")

        trip_display = trip_forecast[
            [
                "label",
                "target_time",
                "origin_speed",
                "destination_speed",
                "origin_condition",
                "destination_condition",
            ]
        ].rename(
            columns={
                "label": "Horizon",
                "target_time": "Waktu Target",
                "origin_speed": route_context["origin_display"],
                "destination_speed": route_context["destination_display"],
                "origin_condition": "Kondisi Asal",
                "destination_condition": "Kondisi Tujuan",
            }
        )

        trip_display["Waktu Target"] = trip_display["Waktu Target"].apply(display_time)

        with st.expander("Lihat detail prediksi endpoint"):
            st.dataframe(
                trip_display,
                width="stretch",
                hide_index=True,
            )

        st.markdown(
            f"**Insight kondisi endpoint:** Horizon "
            f"+{trip_result['suggested_horizon']} jam "
            f"({display_time(trip_result['suggested_time'])}) "
            "memiliki kondisi relatif terbaik di antara "
            "horizon yang diuji untuk corridor asal dan tujuan."
        )
        st.caption(
            "Insight berdasarkan prediksi dua endpoint, bukan prediksi waktu tempuh seluruh rute."
        )


# ============================================================
# FOOTER
# ============================================================


# ============================================================
# NEW PAGES — NEXT IMPLEMENTATION STAGES
# ============================================================

if active_page == "🗃️ Dataset Explorer":
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    data_tab, traffic_tab = st.tabs(
        [
            "🗃️ Data & Quality",
            "🚦 Traffic Patterns",
        ]
    )

    with data_tab:
        from streamlit_app.dataset_explorer import render_dataset_explorer

        render_dataset_explorer()

    with traffic_tab:
        from streamlit_app.history_page import render_history_page

        render_history_page(
            corridor_records=corridor_records,
            get_history_period=get_history_period,
            get_corridor_date_range=get_corridor_date_range,
            pretty_corridor_name=pretty_corridor_name,
        )

if active_page == "🕸️ Graph Explorer":
    from streamlit_app.graph_explorer import render_graph_explorer

    render_graph_explorer(corridor_df)

if active_page == "⚙️ Computational & Scalability":
    from streamlit_app.computational_page import render_computational_page

    render_computational_page()


st.divider()

st.caption(
    "Traffic Forecast & Trip Insight Dashboard · "
    "Graph WaveNet Fixed, Adaptive, dan Hybrid · "
    "Final64 Traffic Dataset"
)
